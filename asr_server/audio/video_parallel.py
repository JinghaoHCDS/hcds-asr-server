"""One retained video task, two bounded encode lanes and in-flight validation.

Only used for zero-origin, unrotated inputs. Audio is encoded continuously once;
video seams are placed on whole seconds and fixed-rate frames. Final packet
hashes prove the faststart MP4 contains exactly the fully decoded lane outputs.
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from collections.abc import Awaitable, Callable

from asr_server.audio.preprocess import INPUT_OPTIONS, _guard_operation
from asr_server.audio.retained import _check_cancel, _event, _hash, _command
from asr_server.audio.video import NVENC_OPTIONS, VIDEO_TIMEOUT, VIDEO_PROFILE, VideoProgress, _read_block, _read_progress, _file_command, probe_video
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext


async def prepare_parallel_video(source: Path, target: Path, info: dict[str, Any], *,
                                 context: ExecutionContext, gpu_input: list[str], max_fps: int,
                                 ensure_capacity: Callable[[int], Awaitable[None]],
                                 written: Callable[[int], Awaitable[None]], progress: VideoProgress | None) -> dict[str, Any]:
    started = perf_counter()
    duration = float(info['duration'])
    paths = [target.with_name(f'video-{i}.partial.mp4') for i in range(3)]
    concat = target.with_name('video.concat.partial.txt')
    merged = target.with_name('video.faststart.partial.mp4')
    totals = [0, 0, 0]
    processed = [0., 0.]
    admitted = reported = 0
    last_progress = 0.
    accounting = asyncio.Lock()
    v, a = info['video'], info['audio']
    factor = min(1., 1920 / max(v['width'],v['height']), 1080 / min(v['width'],v['height']))
    width,height = max(2,int(v['width']*factor)//2*2),max(2,int(v['height']*factor)//2*2)
    try:
        numerator, _, denominator = str(v.get('avg_frame_rate','30/1')).partition('/')
        source_rate = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        source_rate = float(max_fps)
    rate = min(max_fps, source_rate) if math.isfinite(source_rate) and source_rate > 0 else float(max_fps)
    # At high source rates, reference frames alone still provide the interview
    # playback cadence. Skip decoding disposable intermediate H.264 pictures;
    # all encoded output frames are independently decoded below.
    skip_nonref = (v.get('codec_name') == 'h264' and source_rate >= 4 * rate
                   and v.get('r_frame_rate') == v.get('avg_frame_rate'))
    boundary = max(1., math.floor(duration * (.55 if skip_nonref else .5)))
    windows = [(0., boundary), (boundary, duration)]
    lane_seconds = [0., 0., 0.]
    cpu_threads = min(12, os.cpu_count() or 2)

    async def report(index: int, value: dict[str, Any]) -> None:
        nonlocal last_progress
        if index >= 2:
            return
        processed[index] = float(value['processed_seconds'])
        moment = perf_counter()
        if progress and moment-last_progress >= 1:
            last_progress = moment
            seconds = min(duration,sum(processed))
            await progress({'phase':'encoding_video','percent':round(seconds/duration*100,1),
                'processed_seconds':seconds,'duration_seconds':duration,'elapsed_seconds':round(moment-started,1)})

    async def account(index: int, block: bytes, file: Any) -> None:
        nonlocal admitted, reported
        async with accounting:
            total = sum(totals) + len(block)
            if total > admitted:
                admitted = max(total,admitted + 8*1024*1024)
                await ensure_capacity(admitted)
            _check_cancel(context)
            file.write(block)
            totals[index] += len(block)
            if total-reported >= 4*1024*1024:
                await written(total)
                reported = total

    async def lane(index: int) -> None:
        lane_started = perf_counter()
        is_video = index < 2
        command = ['ffmpeg','-v','error','-nostdin','-nostats','-stats_period','1','-progress','pipe:2','-xerror','-err_detect','explode']
        if index == 1:
            command += gpu_input
        elif index == 0:
            command += ['-threads',str(cpu_threads)]
            if skip_nonref:
                command += ['-skip_frame','noref']
        if is_video:
            start,end = windows[index]
            command += ['-ss',str(start)]
        command += [*INPUT_OPTIONS,'-i',str(source.resolve())]
        if is_video:
            command += ['-t',str(end-start),'-map',f"0:{v['index']}",'-an','-sn','-dn']
            filters = f'fps={rate:.9g}'
            if index == 0:
                filters += f',format=nv12,hwupload_cuda,scale_cuda=w={width}:h={height}'
            command += ['-vf',filters,'-filter_threads','1',*NVENC_OPTIONS,'-profile:v','high','-fps_mode','vfr']
        else:
            command += ['-map',f"0:{a['index']}",'-vn','-sn','-dn','-c:a','aac','-b:a','128k','-ac','2','-threads','1']
        command += ['-map_metadata','-1','-frag_duration','2000000','-movflags','+frag_keyframe+delay_moov+default_base_moof','-f','mp4','pipe:1']
        process = await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        async def collect() -> None:
            assert process.stdout is not None and process.stderr is not None
            verifier = await asyncio.create_subprocess_exec('ffmpeg','-v','error','-nostdin','-xerror','-err_detect','explode',
                *(['-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v','h264_cuvid'] if is_video else ['-threads','1']),
                '-protocol_whitelist','pipe','-format_whitelist','mov','-i','pipe:0','-map','0:v:0' if is_video else '0:a:0','-f','null','-',
                stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.PIPE)
            assert verifier.stdin is not None and verifier.stderr is not None
            errors = asyncio.create_task(_read_progress(process.stderr,'encoding_video',end-start if is_video else duration,
                lambda value: report(index,value),rate if is_video else None))
            validation = asyncio.create_task(_read_progress(verifier.stderr,'verifying_video',duration,None))
            try:
                with paths[index].open('xb') as output:
                    while block := await _read_block(process.stdout):
                        if errors.done():errors.result()
                        await account(index,block,output)
                        try:
                            verifier.stdin.write(block)
                            await verifier.stdin.drain()
                        except (BrokenPipeError,ConnectionResetError) as error:
                            raise AsrError(400,'video_integrity_failed','encoded media failed full decode') from error
                    output.flush();os.fsync(output.fileno())
                await process.wait()
                verifier.stdin.close()
                await verifier.wait()
                if process.returncode or await errors:
                    raise AsrError(400,'video_encode_failed','parallel video encoding failed')
                if verifier.returncode or await validation:
                    raise AsrError(400,'video_integrity_failed','encoded media failed full decode')
            finally:
                if verifier.returncode is None:verifier.kill()
                await verifier.wait()
                for task in (errors,validation):
                    if not task.done():task.cancel()
                await asyncio.gather(errors,validation,return_exceptions=True)
        await _guard_operation(process,collect(),timeout_seconds=VIDEO_TIMEOUT,
            timeout_code='video_preprocess_timeout',cancel_event=_event(context))
        lane_seconds[index] = round(perf_counter()-lane_started,3)

    tasks: list[asyncio.Task[None]] = []
    try:
        if progress:await progress({'phase':'encoding_video','percent':0.,'processed_seconds':0.,'duration_seconds':duration,'elapsed_seconds':0.})
        tasks = [asyncio.create_task(lane(index)) for index in range(3)]
        await asyncio.gather(*tasks)
        await written(sum(totals))
        encode_seconds = perf_counter()-started
        bound = sum(totals)*2+16*1024*1024
        await ensure_capacity(bound)
        concat.write_text(''.join(f"file '{path.name}'\n" for path in paths[:2]))
        inputs = ['-protocol_whitelist','file','-f','concat','-safe','1','-auto_convert','0','-i',str(concat.resolve()),
                  *INPUT_OPTIONS,'-i',str(paths[2].resolve())]
        mapping = ['-map','0:v:0','-map','1:a:0','-c','copy']
        await _file_command(['ffmpeg','-v','error','-nostdin','-xerror',*inputs,*mapping,'-movflags','+faststart',
            '-fs',str(bound-sum(totals)),'-n',str(merged.resolve())],context,phase='finalizing_video',duration=duration,progress=progress,frame_rate=rate)
        await written(sum(totals)+merged.stat().st_size)
        remux_seconds = perf_counter()-started-encode_seconds
        if progress:await progress({'phase':'verifying_video','percent':0.,'processed_seconds':0.,'duration_seconds':duration})
        expected,_ = await _command('ffmpeg','-v','error','-nostdin',*inputs,*mapping,'-f','streamhash','-hash','sha256','-',context=context,timeout_seconds=VIDEO_TIMEOUT)
        actual,_ = await _command('ffmpeg','-v','error','-nostdin',*INPUT_OPTIONS,'-i',str(merged.resolve()),'-map','0:v:0','-map','0:a:0',
            '-c','copy','-f','streamhash','-hash','sha256','-',context=context,timeout_seconds=VIDEO_TIMEOUT)
        if expected != actual:raise AsrError(400,'video_integrity_failed','remux changed validated media packets')
        output = await probe_video(merged,context)
        if output is None:raise AsrError(400,'video_integrity_failed','encoded video stream missing')
        encoded = output['video']
        if (encoded.get('codec_name')!='h264' or encoded.get('pix_fmt')!='yuv420p'
                or encoded['width']!=width or encoded['height']!=height
                or abs(output['duration']-duration)>.15
                or abs(output['audio_start']-output['video_start'])>.15):
            raise AsrError(400,'video_timeline_invalid','parallel output specifications or timeline differ')
        _check_cancel(context)
        digest = await asyncio.to_thread(_hash,merged)
        with merged.open('rb') as file:os.fsync(file.fileno())
        metadata = {**VIDEO_PROFILE,'version':'h264-1080-hybrid-v1','preset':'p1','cq':23,'rate_control':'vbr','max_fps':max_fps,
            'state':'ready','encoder':'h264_nvenc','hardware_decode':True,'decode_mode':'cpu_cuda_parallel','encode_lanes':2,
            'cpu_decode_threads':cpu_threads,'skip_nonreference_frames':skip_nonref,'split_seconds':boundary,
            'sha256':digest,'size_bytes':merged.stat().st_size,'width':width,'height':height,'duration_seconds':output['duration'],
            'audio_offset_seconds':output['video_start'],'source_stream_index':v['index'],'source_audio_stream_index':a['index'],
            'source_width':v['width'],'source_height':v['height'],'preparation_seconds':round(perf_counter()-started,3),
            'timings_seconds':{'encoding_and_full_decode':round(encode_seconds,3),'cpu_lane':lane_seconds[0],
                'gpu_lane':lane_seconds[1],'audio_lane':lane_seconds[2],'remux':round(remux_seconds,3),
                'packet_verification':round(perf_counter()-started-encode_seconds-remux_seconds,3)}}
        metadata.pop('crf',None)
        merged.replace(target)
        directory=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        if progress:await progress({'phase':'verifying_video','percent':100.,'processed_seconds':duration,'duration_seconds':duration})
        return metadata
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        for path in [*paths,concat,merged]:path.unlink(missing_ok=True)
