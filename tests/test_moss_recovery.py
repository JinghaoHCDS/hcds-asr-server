from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from asr_server.adapters.base import AudioComposition, TranscriptionResult, TranscriptionSegment, TranscriptionTimings
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.audio.splitter import ChunkDescriptor
from asr_server.config import Settings
from asr_server.diarization.anchor_replay import AnchorReplaySequence
from asr_server.diarization.recovery import MossCheckpoints, _patch_gap, run_moss_chunks
from asr_server.errors import AsrError, error_payload
from asr_server.main import create_app
from asr_server.transcription import validate_transcription_request
from asr_server.workers.codec import error_to_payload, result_from_payload
from tests.helpers import make_wav


def result(*segments: tuple[float, float, str | None, str], duration: float = 100) -> TranscriptionResult:
    return TranscriptionResult('\n'.join(item[3] for item in segments), duration, 'zh', [],
        segments=[TranscriptionSegment(a,b,text,speaker) for a,b,speaker,text in segments])


async def test_resume_rebuilds_anchor_identity_without_regenerating_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app=create_app(Settings(adapter='mock',enable_moss=True,workbench_data_dir=tmp_path/'data'))
    request=app.state.workbench.submission_request({'model':'moss-transcribe-diarize-0.9b','quality_profile':'legacy'})
    manager=app.state.manager; checked=validate_transcription_request(manager,request)
    path=tmp_path/'audio.wav';path.write_bytes(make_wav(300));asset=NormalizedAudioAsset.from_path(path)
    chunks=[ChunkDescriptor(i,i*100,(i+1)*100,path) for i in range(3)]
    calls=[];fail=True
    async def recognize(inputs: Any, **kw: Any) -> Any:
        nonlocal fail
        audio=inputs[0];calls.append(audio)
        if len(calls)==2 and fail:
            fail=False
            raise AsrError(503,'worker_timeout','synthetic interruption')
        if isinstance(audio,AudioComposition):
            # The old S01 person is now S02; references must restore mapping.
            value=result((0,8,'S02','甲参考'),(8.5,16.5,'S01','乙参考'),
                         (17.5,24,'S02','甲继续'),(25,31,'S01','乙继续'),duration=117)
        else:
            value=result((1,9,'S01','甲'),(12,20,'S02','乙'))
        return 'transformers',[value],TranscriptionTimings()
    monkeypatch.setattr(manager,'transcribe_chunks',recognize)
    async def run() -> Any:
        return await run_moss_chunks(manager=manager,request=request,checked=checked,chunks=chunks,asset=asset,
            directory=tmp_path/'checkpoints',max_new_tokens=65536,sequence=AnchorReplaySequence(chunks),
            cancel_event=threading.Event(),before_chunk=None,after_chunk=None)
    with pytest.raises(AsrError) as failure:
        await run()
    assert failure.value.code=='worker_timeout' and len(calls)==2
    _,results,_,gaps,_=await run()
    assert len(calls)==4 and not gaps
    assert results[0].segments[0].speaker==results[1].segments[0].speaker==results[2].segments[0].speaker
    await run()
    assert len(calls)==4  # Entire successful retry is a verified cache hit.
    request=replace(request,max_new_tokens=24000)
    with pytest.raises(AsrError) as failure:
        await run()
    assert failure.value.code=='checkpoint_version_mismatch'
    app.state.workbench.close()


@pytest.mark.parametrize('covered',[True,False])
async def test_local_review_preserves_raw_and_charges_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, covered: bool) -> None:
    app=create_app(Settings(adapter='mock',enable_moss=True,workbench_data_dir=tmp_path/'data'))
    request=app.state.workbench.submission_request({'model':'moss-transcribe-diarize-0.9b','quality_profile':'legacy'})
    manager=app.state.manager;checked=validate_transcription_request(manager,request)
    path=tmp_path/'audio.wav';path.write_bytes(make_wav(100));asset=NormalizedAudioAsset.from_path(path)
    chunks=[ChunkDescriptor(0,0,100,path)];calls=0
    original=result((1,9,'S01','已有正文'),(30,40,'S02','另一人'),(90,99,'S01','尾部正文'))
    async def recognize(inputs: Any, **kw: Any) -> Any:
        nonlocal calls
        calls+=1
        if calls==1:
            error=AsrError(422,'incomplete_transcript','candidate activity',{'uncovered_speech_intervals':[{'start':20.,'end':20.5}]})
            error.partial_transcription=original
            raise error
        audio=inputs[0];assert isinstance(audio,AudioComposition)
        # Reference local labels are deliberately swapped.
        value=result((0,8,'S02','甲参考'),(8.5,16.5,'S01','乙参考'),
                     (audio.prefix_duration+4,audio.prefix_duration+4.5,'S01','嗯'),duration=audio.duration)
        if not covered:value=replace(value,segments=value.segments[:2])
        return 'transformers',[value],TranscriptionTimings()
    monkeypatch.setattr(manager,'transcribe_chunks',recognize)
    async def run() -> Any:
        return await run_moss_chunks(manager=manager,request=request,checked=checked,chunks=chunks,asset=asset,
            directory=tmp_path/'checkpoints',max_new_tokens=65536,sequence=None,cancel_event=threading.Event(),
            before_chunk=None,after_chunk=None)
    _,items,_,gaps,quality=await run()
    assert bool(gaps) is not covered
    assert items[0].segments[0]==original.segments[0]
    if covered:
        inserted=next(s for s in items[0].segments if s.text=='嗯')
        assert inserted.speaker=='S02' and inserted.start==20
    _,again,_,again_gaps,again_quality=await run()
    assert calls==2 and again==items and again_gaps==gaps
    assert again_quality['review_audio_seconds']==quality['review_audio_seconds']>0
    import json
    raw=json.loads(next((tmp_path/'checkpoints').rglob('chunk-00000.json')).read_text())
    assert raw['payload']['result']['text']==original.text
    app.state.workbench.close()


def test_review_never_overwrites_existing_words_or_invents_speaker() -> None:
    original=result((0,10,'S01','existing'))
    gap={'start':10.1,'end':10.6}
    assert _patch_gap(original,result((9.5,10.5,'S99','changed')),gap,[],0,0) is None
    patched=_patch_gap(original,result((10.1,10.5,'S99','new')),gap,[],0,0)
    assert patched is not None and patched.segments[-1].speaker is None


def test_partial_rpc_payload_never_enters_public_error_envelope() -> None:
    error=AsrError(422,'incomplete_transcript','review required',{'candidate_count':1})
    error.partial_transcription=result((0,1,'S01','private transcript'))
    wire=error_to_payload(error)
    assert result_from_payload(wire['partial_transcription'])==error.partial_transcription
    assert 'private transcript' not in str(error_payload(error.code,error.message,error.details))


def test_checkpoint_corruption_and_interrupted_review_budget(tmp_path: Path) -> None:
    store=MossCheckpoints(tmp_path,{'audio':'synthetic'})
    store.save('chunk-00000','window',{'result':'synthetic'})
    path=store.root/'chunk-00000.json';path.write_text(path.read_text().replace('synthetic','modified'))
    with pytest.raises(AsrError) as failure:store.load('chunk-00000','window')
    assert failure.value.code=='checkpoint_corrupt'
    assert store.reserve('review-0',0,10,30)
    restarted=MossCheckpoints(tmp_path,{'audio':'synthetic'})
    assert not restarted.reserve('review-0',0,10,30)
    assert not restarted.reserve('review-1',0,21,30)
