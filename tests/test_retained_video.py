from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from asr_server.audio.video import prepare_video, probe_video, validate_video
from asr_server.observability import ExecutionContext
from asr_server.errors import AsrError

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")


def make_video(path: Path, size: str = "320x180", *, offset: float = 0) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration=1.5",
        "-itsoffset", str(offset), "-f", "lavfi", "-i", "sine=frequency=660:duration=1.5",
        "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", str(path)], check=True)


@pytest.mark.parametrize("size,offset", [("320x180", 0), ("180x320", 0), ("3840x2160", 0), ("320x180", 0.5)])
async def test_video_encode_validate_and_timeline(tmp_path: Path, size: str, offset: float) -> None:
    source, target = tmp_path / "source.mp4", tmp_path / "video.mp4"
    make_video(source, size, offset=offset)
    info = await probe_video(source)
    assert info is not None
    admitted = 0
    progress = []
    async def report(value: dict[str, Any]) -> None:
        progress.append(value)
    async def capacity(total: int) -> None:
        nonlocal admitted
        admitted = max(admitted, total)
    async def written(total: int) -> None:
        assert total <= admitted
    metadata = await prepare_video(source, target, info, context=ExecutionContext(trace_id="test-video"), ensure_capacity=capacity, written=written, progress=report)
    validate_video(target, metadata)
    assert max(metadata["width"], metadata["height"]) <= 1920
    assert min(metadata["width"], metadata["height"]) <= 1080
    assert (metadata["width"] > metadata["height"]) == (int(size.split('x')[0]) > int(size.split('x')[1]))
    assert abs(metadata["audio_offset_seconds"] - offset) < 0.15
    assert source.exists() and not list(tmp_path.glob("*.partial.mp4"))
    assert metadata['encoder'] == 'libx264'
    assert {p['phase'] for p in progress} == {'checking_video', 'encoding_video', 'finalizing_video', 'verifying_video'}
    for phase in {p['phase'] for p in progress}:
        measured = [p['processed_seconds'] for p in progress if p['phase'] == phase]
        assert measured == sorted(measured) and measured[0] == 0
        assert measured[-1] > 0 and measured[-1] <= info['duration']
    assert all(0 <= p['percent'] <= 100 for p in progress)


async def test_encoder_selection_requires_actual_driver_and_honors_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.audio import video
    calls = []
    async def available(*args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
        calls.append(args)
        return b'', b''
    monkeypatch.setattr(video, '_command', available)
    context = ExecutionContext(trace_id='encoder-choice')
    assert await video.select_video_encoder('cpu', context) == 'libx264' and not calls
    assert await video.select_video_encoder('auto', context) == 'h264_nvenc'
    async def unavailable(*args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
        raise AsrError(400, 'audio_decode_failed', 'synthetic unavailable driver')
    monkeypatch.setattr(video, '_command', unavailable)
    assert await video.select_video_encoder('auto', context) == 'libx264'
    with pytest.raises(AsrError) as error:
        await video.select_video_encoder('nvenc', context)
    assert error.value.code == 'video_encoder_unavailable'
    async def cancelled(*args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
        raise asyncio.CancelledError
    monkeypatch.setattr(video, '_command', cancelled)
    with pytest.raises(asyncio.CancelledError):
        await video.select_video_encoder('auto', context)


async def test_progress_ignores_unknown_times_and_retains_errors() -> None:
    from asr_server.audio.video import _read_progress
    stream = asyncio.StreamReader()
    stream.feed_data(b'out_time_us=N/A\nprogress=continue\nout_time_us=inf\nprogress=continue\n'
                     b'out_time_us=2500000\nprogress=continue\nframe=75\nout_time_us=5000000\nprogress=end\n'
                     b'[decoder] damaged packet\n')
    stream.feed_eof()
    values = []
    async def report(value: dict[str, Any]) -> None:
        values.append(value)
    errors = await _read_progress(stream, 'encoding_video', 5, report)
    assert [p['percent'] for p in values] == [50, 100]
    assert errors == b'[decoder] damaged packet\n'


async def test_video_storage_rejection_cleans_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    make_video(source)
    info = await probe_video(source)
    assert info is not None
    async def reject(_: int) -> None:
        raise AsrError(503, "storage_unavailable", "test budget exhausted")
    async def written(_: int) -> None:
        pass
    with pytest.raises(AsrError, match="budget"):
        await prepare_video(source, tmp_path / "video.mp4", info, context=ExecutionContext(trace_id="test-video"), ensure_capacity=reject, written=written)
    assert source.exists() and not list(tmp_path.glob("*.partial.mp4"))

async def test_nonvideo_cover_corruption_and_missing_audio(tmp_path: Path) -> None:
    from tests.helpers import make_wav
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(make_wav(1))
    assert await probe_video(audio) is None
    cover = tmp_path / 'cover.mp3'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(audio), '-f', 'lavfi', '-i', 'color=size=160x90',
        '-map', '0:a', '-map', '1:v', '-c:a', 'libmp3lame', '-c:v', 'mjpeg', '-frames:v', '1',
        '-disposition:v', 'attached_pic', '-y', str(cover)], check=True)
    assert await probe_video(cover) is None
    silent = tmp_path / 'silent.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=160x90:duration=0.5', '-c:v', 'libx264', '-threads', '2', '-y', str(silent)], check=True)
    with pytest.raises(AsrError, match='audio'):
        await probe_video(silent)
    source = tmp_path / 'source.mp4'
    make_video(source)
    source.write_bytes(source.read_bytes()[:200])
    with pytest.raises(AsrError):
        await probe_video(source)


async def test_rotation_metadata_and_damaged_tail(tmp_path: Path) -> None:
    source, rotated, target = tmp_path / 'source.mp4', tmp_path / 'rotated.mp4', tmp_path / 'video.mp4'
    make_video(source)
    modern_rotation = b'-display_rotation' in subprocess.check_output(['ffmpeg', '-hide_banner', '-h', 'full'], stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-v', 'error', *(['-display_rotation:v:0', '90'] if modern_rotation else []),
        '-i', str(source), '-c', 'copy', *([] if modern_rotation else ['-metadata:s:v:0', 'rotate=90']),
        '-movflags', '+faststart', '-y', str(rotated)], check=True)
    info = await probe_video(rotated)
    assert info is not None
    async def capacity(_: int) -> None:
        pass
    metadata = await prepare_video(rotated, target, info, context=ExecutionContext(trace_id='rotation'), ensure_capacity=capacity, written=capacity)
    assert (metadata['width'], metadata['height']) == (180, 320)
    # The moov box remains readable while final media packets are truncated.
    rotated.write_bytes(rotated.read_bytes()[:-5000])
    info = await probe_video(rotated)
    assert info is not None
    with pytest.raises(AsrError):
        await prepare_video(rotated, tmp_path / 'damaged.mp4', info, context=ExecutionContext(trace_id='damaged-tail'), ensure_capacity=capacity, written=capacity)
    assert not (tmp_path / 'damaged.mp4').exists() and not list(tmp_path.glob('*.partial.mp4'))


async def test_cancelled_video_keeps_source_and_cleans(tmp_path: Path) -> None:
    source = tmp_path / 'source.mp4'
    make_video(source)
    info = await probe_video(source)
    assert info is not None
    context = ExecutionContext(trace_id='cancel-video')
    async def capacity(_: int) -> None:
        context.cancel_event.set()
    async def wrote(_: int) -> None:
        pass
    with pytest.raises(asyncio.CancelledError):
        await prepare_video(source, tmp_path / 'video.mp4', info, context=context, ensure_capacity=capacity, written=wrote)
    assert source.exists() and not (tmp_path / 'video.mp4').exists()
    assert not list(tmp_path.glob('*.partial.mp4'))

async def test_flash_tone_markers_and_variable_frame_rate(tmp_path: Path) -> None:
    """Measure decoded signals at three times, not only container metadata."""
    import array
    source, target = tmp_path / 'markers.mp4', tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        "color=black:size=160x90:rate=50:duration=4,drawbox=color=white:t=fill:enable='between(t,0.9,1.0)+between(t,1.9,2.0)+between(t,3.4,3.5)',select='if(lt(t,2),not(mod(n,2)),1)'",
        '-itsoffset', '0.4', '-f', 'lavfi', '-i',
        "sine=frequency=660:duration=3.6,volume=0:enable='not(between(t,0.5,0.6)+between(t,1.5,1.6)+between(t,3.0,3.1))'",
        '-fps_mode', 'vfr', '-c:v', 'libx264', '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', '-y', str(source)], check=True)
    info = await probe_video(source)
    assert info is not None
    async def capacity(_: int) -> None:
        pass
    metadata = await prepare_video(source, target, info, context=ExecutionContext(trace_id='markers'), ensure_capacity=capacity, written=capacity)
    # Make absolute presentation time explicit in decoded probes.
    picture = subprocess.check_output(['ffmpeg', '-v', 'error', '-copyts', '-i', str(target),
        '-vf', 'fps=100:start_time=0,scale=1:1', '-pix_fmt', 'gray', '-f', 'rawvideo', '-an', 'pipe:1'])
    sound = subprocess.check_output(['ffmpeg', '-v', 'error', '-copyts', '-i', str(target),
        '-af', 'aresample=8000:async=1:first_pts=0', '-ar', '8000', '-ac', '1', '-f', 's16le', '-vn', 'pipe:1'])
    samples = array.array('h', sound)
    energy = [max(abs(x) for x in samples[i:i+80]) for i in range(0, len(samples)-80, 80)]
    def starts(values: list[int], threshold: int) -> list[float]:
        return [i / 100 for i, value in enumerate(values) if value > threshold and (i == 0 or values[i-1] <= threshold)]
    flashes, tones = starts(list(picture), 150), starts(energy, 500)
    assert len(flashes) == len(tones) == 3, (flashes, tones)
    assert all(abs(v - a) <= .15 for v, a in zip(flashes, tones, strict=True)), (flashes, tones)
    for local, absolute in zip([.5, 1.5, 3.0], tones, strict=True):
        assert abs(local + metadata['audio_offset_seconds'] - absolute) <= .15
    output = await probe_video(target)
    assert output is not None
    numerator, denominator = output['video']['avg_frame_rate'].split('/')
    assert float(numerator) / float(denominator) <= 30.01


@pytest.mark.skipif(__import__('os').environ.get('ASR_TEST_NVENC') != '1', reason='requires authorized NVIDIA node')
@pytest.mark.parametrize('variable', [False, True])
async def test_parallel_gpu_full_decode_seam_and_markers(tmp_path: Path, variable: bool) -> None:
    import array
    from asr_server.audio.video_parallel import prepare_parallel_video
    source, target = tmp_path / 'markers.mp4', tmp_path / 'video.mp4'
    selection = ",select='if(lt(t,2),not(mod(n,2)),1)'" if variable else ''
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        "color=black:size=320x180:rate=60:duration=4,drawbox=color=white:t=fill:enable='between(t,0.8,0.95)+between(t,1.96,2.12)+between(t,3.4,3.55)'" + selection,
        '-f', 'lavfi', '-i', "sine=frequency=660:duration=4,volume=0:enable='not(between(t,0.8,0.95)+between(t,1.96,2.12)+between(t,3.4,3.55))'",
        '-fps_mode', 'vfr', '-c:v', 'libx264', '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', '-y', str(source)], check=True)
    info = await probe_video(source)
    assert info is not None
    async def capacity(_: int) -> None: pass
    metadata = await prepare_parallel_video(source, target, info, context=ExecutionContext(trace_id='gpu-seam'),
        gpu_input=['-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v','h264_cuvid'], max_fps=15,
        ensure_capacity=capacity, written=capacity, progress=None)
    # Independent CPU decoder checks the final file, including the join.
    subprocess.run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-i',str(target),'-f','null','-'], check=True)
    picture = subprocess.check_output(['ffmpeg','-v','error','-i',str(target),'-vf','fps=100:start_time=0,scale=1:1',
        '-pix_fmt','gray','-f','rawvideo','-an','pipe:1'])
    sound = subprocess.check_output(['ffmpeg','-v','error','-i',str(target),'-af','aresample=8000:async=1:first_pts=0',
        '-ar','8000','-ac','1','-f','s16le','-vn','pipe:1'])
    samples = array.array('h', sound)
    energy = [max(abs(x) for x in samples[i:i+80]) for i in range(0,len(samples)-80,80)]
    def starts(values: list[int], threshold: int) -> list[float]:
        return [i/100 for i,value in enumerate(values) if value>threshold and (i==0 or values[i-1]<=threshold)]
    flashes,tones = starts(list(picture),150),starts(energy,500)
    assert len(flashes)==len(tones)==3, (flashes,tones)
    assert all(abs(v-a)<=.15 for v,a in zip(flashes,tones,strict=True)), (flashes,tones)
    assert metadata['skip_nonreference_frames'] is (not variable)
    assert not list(tmp_path.glob('*.partial*'))
    damaged = tmp_path / 'damaged.mp4'
    damaged.write_bytes(source.read_bytes()[:-256])
    with pytest.raises(AsrError):
        await prepare_parallel_video(damaged, tmp_path / 'broken.mp4', info, context=ExecutionContext(trace_id='gpu-tail'),
            gpu_input=['-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v','h264_cuvid'], max_fps=15,
            ensure_capacity=capacity, written=capacity, progress=None)
    assert not (tmp_path / 'broken.mp4').exists() and not list(tmp_path.glob('*.partial*'))
