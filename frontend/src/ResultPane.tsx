import VideoProgress from './VideoProgress';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, bytes, clock, date, errorNames, jobDetail, message, pendingSpeaker, phaseName, terminal, canRetry, retryFromStart, retryLabel, progressText, completeness, jobStatusLabel, showingVideoProgress, videoPending, type Issue, type Job, type ModelOption, type Segment } from './api';
import { Icon } from './icons';
import SpeakerTools from './SpeakerTools';
import ReplaceTools from './ReplaceTools';
import Transcript, { VirtualText } from './Transcript';
import { useEntrance } from './motion';

function issueText(issue: Issue | string) {
  if (typeof issue === 'string') return issue;
  if (issue.kind === 'language_fallback') return '依据此前片段语言恢复了文字，请回听确认';
  if (issue.kind === 'near_silent_tail') return '尾段疑似接近静音，暂无文字，可回听确认';
  if (issue.kind === 'empty_without_detected_speech') return '此处未识别出文字，语音检测也未检出语音，请回听确认';
  const names: Record<string, string> = { repetition: '检测到疑似重复，请回听确认', suspected_repetition: '检测到疑似重复，请回听确认', empty_speech: '该区间暂无文字，可能是语音、音乐或音效，请回听确认', empty_transcription: '该区间暂无文字，可能是语音、音乐或音效，请回听确认', term_candidate: '术语写法需要确认', suspected_term: '术语写法需要确认', boundary_conflict: '分段交界处需要确认', review_budget_exhausted: '此处仍需要人工回听', alignment_failed: '字幕对齐未完成', suspected_uncovered_audio: '该区间暂无对应文字，可能是语音、音乐或音效，请回听确认', alignment_unavailable_mock: '此测试结果没有真实时间对齐', alignment_text_mapping: '文字与字幕对齐关系需要确认', generation_truncated: '此处识别输出可能不完整' };
  return names[issue.kind || issue.code || issue.type || ''] || issue.message || '此片段需要回听确认';
}

export default function ResultPane({ jobId, models, onBack, onChanged }: { jobId: string; models: ModelOption[]; onBack: () => void; onChanged: () => void }) {
  const root = useRef<HTMLDivElement>(null);
  const modal = useRef<HTMLDivElement>(null);
  const tools = useRef<HTMLElement>(null);
  const toolsButton = useRef<HTMLButtonElement>(null);
  const toolsClose = useRef<HTMLButtonElement>(null);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [issuePage, setIssuePage] = useState(0);
  const [changePage, setChangePage] = useState(0);
  const [gapPage, setGapPage] = useState(0);
  const [recovering, setRecovering] = useState(false);
  const [jump, setJump] = useState<{ seconds: number; serial: number } | null>(null);
  const [job, setJob] = useState<Job | null>(null); const [error, setError] = useState(''); const [raw, setRaw] = useState(false);
  const [loading, setLoading] = useState(true);
  const [edit, setEdit] = useState<Segment | null>(null); const [editBaseVersion, setEditBaseVersion] = useState(1); const [draft, setDraft] = useState(''); const [saving, setSaving] = useState(false);
  const [speakerDraft, setSpeakerDraft] = useState('');
  const [mediaMode, setMediaMode] = useState<'video' | 'audio'>('video');
  const video = useRef<HTMLVideoElement>(null);
  const [position, setPosition] = useState(0); const [exportVersion, setExportVersion] = useState<number | null>(null);
  const [format, setFormat] = useState('md'); const [exporting, setExporting] = useState(false); const [notice, setNotice] = useState('');
  const [partialFormat, setPartialFormat] = useState('txt');
  const [audioError, setAudioError] = useState(false); const audio = useRef<HTMLAudioElement>(null);
  const changed = useRef(onChanged); changed.current = onChanged;
  const observed = useRef('');
  useEntrance(root, !!job?.result, '.audio-panel, .review-body, .waiting-result');
  useEntrance(modal, edit?.id, '.edit-modal');
  useEntrance(root, toolsOpen, '.result-aside');
  useEffect(() => {
    if (!toolsOpen) return;
    toolsClose.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setToolsOpen(false); }
      if (event.key !== 'Tab') return;
      const nodes = Array.from(tools.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex="0"]') || []);
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); toolsButton.current?.focus(); };
  }, [toolsOpen]);
  useEffect(() => {
    if (!edit) return;
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !saving) setEdit(null);
      if (event.key !== 'Tab') return;
      const nodes = Array.from(modal.current?.querySelectorAll<HTMLElement>('button:not(:disabled), textarea, select, input, [tabindex="0"]') || []);
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); previous?.focus(); };
  }, [edit, saving]);
  useEffect(() => { if (!notice) return; const timer = window.setTimeout(() => setNotice(''), 5000); return () => window.clearTimeout(timer); }, [notice]);
  const etag = useRef<{ id: string; value?: string }>({ id: jobId });
  const selectedId = useRef(jobId); selectedId.current = jobId;
  const capabilities = models.find(model => model.id === job?.model)?.capabilities;
  const editable = job?.status === 'completed' && job.result?.alignment_status !== 'pending';
  const canEdit = !!(capabilities?.forced_alignment || capabilities?.segment_timestamps) && editable;
  const canSubtitle = !!(capabilities?.forced_alignment || capabilities?.segment_timestamps);
  useEffect(() => { if (!canSubtitle) setFormat('md'); }, [canSubtitle]);

  const load = useCallback(async () => {
    try {
      const next = await jobDetail(jobId, etag.current.id === jobId ? etag.current.value : undefined);
      if (selectedId.current !== jobId) return;
      etag.current = { id: jobId, value: next.etag };
      const loaded = next.job;
      if (loaded) {
        const signature = `${loaded.id}:${loaded.status}:${loaded.result?.alignment_status}:${loaded.audio_asset?.state}`;
        if (observed.current && observed.current !== signature) changed.current();
        observed.current = signature;
        setJob(current => current?.id === loaded.id && ((current.result?.version ?? 0) > (loaded.result?.version ?? 0)
          || (current.revision ?? 0) > (loaded.revision ?? 0)) ? current : loaded);
      }
      setError('');
    } catch (e) { if (selectedId.current === jobId) setError(message(e)); }
    finally { if (selectedId.current === jobId) setLoading(false); }
  }, [jobId]);
  useEffect(() => {
    etag.current = { id: jobId }; observed.current = ''; setJob(null); setMediaMode('video'); setPosition(0); setJump(null); setAudioError(false); setLoading(true); void load();
  }, [jobId, load]);
  const pending = !!job && (!terminal(job.status) || ['preparing', 'queued'].includes(job.video_asset?.state || '') || job.result?.alignment_status === 'pending');
  useEffect(() => {
    if (!pending) return;
    let alive = true; let timer: number;
    const poll = async () => {
      if (document.visibilityState === 'visible') await load();
      if (alive) timer = window.setTimeout(() => void poll(), document.visibilityState === 'hidden' ? 30000 : 3500);
    };
    const visible = () => { if (document.visibilityState === 'visible') void load(); };
    timer = window.setTimeout(() => void poll(), 3500);
    document.addEventListener('visibilitychange', visible);
    return () => { alive = false; window.clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [pending, load]);
  useEffect(() => { if (job?.result?.version !== undefined) setExportVersion(job.result.version); }, [job?.result?.version]);
  const asset = job?.audio_asset;
  const audioReady = asset?.state === 'ready';
  const audioSource = audioReady ? `/api/jobs/${encodeURIComponent(jobId)}/audio?v=${encodeURIComponent(asset.sha256 || String(asset.size_bytes || 'ready'))}` : undefined;
  useEffect(() => { setAudioError(false); }, [audioSource]);

  const videoReady = job?.video_asset?.state === 'ready';
  const useVideo = videoReady && mediaMode === 'video';
  const offset = job?.video_asset?.audio_offset_seconds || 0;
  const videoSource = videoReady ? `/api/jobs/${encodeURIComponent(jobId)}/video?v=${job?.video_asset?.sha256 || ''}` : undefined;
  const media = () => useVideo ? video.current : audio.current;
  function switchMedia(mode: 'video' | 'audio') {
    if (mode === mediaMode) return;
    const player = media();
    if (player) { setPosition(Math.max(0, player.currentTime - (useVideo ? offset : 0))); player.pause(); }
    setMediaMode(mode); setAudioError(false);
  }
  async function cancelVideo() {
    try { const next = await api<Job>(`/jobs/${jobId}/cancel`, {method: 'POST'}); setJob(next); etag.current = {id: jobId}; }
    catch (e) { setError(message(e)); }
  }
  async function retryVideo() {
    try { const next = await api<Job>(`/jobs/${jobId}/video/retry`, { method: 'POST' }); setJob(next); etag.current = {id: jobId}; }
    catch (e) { setError(message(e)); }
  }
  function speakersSaved(next: Job) { setJob(next); etag.current = {id: jobId}; onChanged(); }
  async function bulkEdit(path: string, fields: Record<string, unknown>, success: string) {
    const version = job?.result?.version;
    const next = await api<Job>(`/jobs/${jobId}/${path}`, {
      method: 'POST', body: JSON.stringify({ ...fields, base_version: version }),
    });
    if (selectedId.current !== jobId) return;
    speakersSaved(next);
    setNotice(next.result?.alignment_status === 'pending' ? `${success}，正在更新字幕对齐。` : `${success}。`);
  }
  function seek(seconds: number) { setJump(previous => ({ seconds, serial: (previous?.serial ?? 0) + 1 })); setPosition(seconds); if (!audioReady && !useVideo) { setNotice('媒体正在准备。'); return; } const player = media(); if (player) { player.currentTime = Math.max(0, seconds + (useVideo ? offset : 0)); void player.play().catch(() => setNotice('请点击播放器的播放按钮。')); } }
  function beginEdit(segment: Segment) { setEdit(segment); setEditBaseVersion(job?.result?.version ?? 1); setDraft(segment.text); setSpeakerDraft(segment.speaker_id || ''); setError(''); }
  async function save() {
    if (!edit || !job?.result || !draft.trim()) return;
    setSaving(true); setError('');
    try {
      const next = await api<Job>(`/jobs/${jobId}/segments/${encodeURIComponent(edit.id)}`, { method: 'PATCH', body: JSON.stringify({ text: draft.trim(), ...(capabilities?.diarization ? {speaker_id: speakerDraft || null} : {}), base_version: editBaseVersion }) });
      etag.current = { id: jobId }; setJob(next); setEdit(null); setNotice(next.result?.alignment_status === 'pending' ? '已保存，正在更新字幕对齐。' : '校对已保存为新版本。'); onChanged();
    } catch (e) { setError(message(e)); } finally { setSaving(false); }
  }
  async function resume() {
    if (!job || recovering || !canRetry(job)) return;
    if (retryFromStart(job) && !window.confirm('从头处理将保留已有文字和版本，并重新识别音频。继续？')) return;
    setRecovering(true); setError('');
    try {
      await api(`/jobs/${jobId}/retry`, { method: 'POST', body: JSON.stringify(retryFromStart(job) ? { from_start: true } : {}) });
      etag.current = { id: jobId }; await load(); setNotice('已加入队列，已有可用文字会继续保留。'); onChanged();
    } catch (e) { setError(message(e)); } finally { setRecovering(false); }
  }
  async function download() {
    if (!job || (!job.result && !job.progress?.text)) return;
    setExporting(true); setError('');
    try {
      const selectedFormat = job.result ? format : partialFormat;
      const query = new URLSearchParams({ format: selectedFormat });
      if (job.result && exportVersion !== null) query.set('version', String(exportVersion));
      else query.set('partial', 'true');
      const response = await fetch(`/api/jobs/${jobId}/export?${query}`, { credentials: 'same-origin' });
      if (!response.ok) { if (response.status === 401) window.dispatchEvent(new Event('asr:unauthorized')); const data = await response.json().catch(() => null); throw new Error(data?.error?.message || `导出失败 (${response.status})`); }
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `${job.filename.replace(/\.[^.]+$/, '')}${exportVersion !== null ? `-v${exportVersion}` : ''}${completeness(job).complete === false || !job.result ? '-incomplete' : ''}.${selectedFormat}`; anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(message(e)); } finally { setExporting(false); }
  }
  async function copy() {
    const text = raw ? job?.result?.raw_text : job?.result?.text;
    if (!text) return;
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else { const field = document.createElement('textarea'); field.value = text; field.style.position = 'fixed'; field.style.opacity = '0'; document.body.append(field); field.select(); const copied = document.execCommand('copy'); field.remove(); if (!copied) throw new Error('浏览器未允许复制，请直接选择文字复制'); }
      setNotice('文字已复制');
    } catch (e) { setError(message(e)); }
  }

  const result = job?.result; const segments = result?.segments || []; const issues = result?.quality?.issues || [];
  const coverage = job ? completeness(job) : {}; const gaps = job?.gaps || result?.gaps || [];
  const incomplete = coverage.complete === false || (coverage.gap_count ?? 0) > 0;
  const versionStatus = result?.versions?.find(v => v.version === exportVersion)?.alignment_status || result?.alignment_status;
  const segmentTimed = !!capabilities?.segment_timestamps && ['segment_timed', 'unavailable'].includes(versionStatus || '');
  const subtitleBlocked = ['srt', 'vtt'].includes(format) && (incomplete || (versionStatus !== 'aligned' && !segmentTimed));
  const taskTone = incomplete && terminal(job?.status || '') ? 'warning' : job?.status === 'completed' ? 'success' : job?.status === 'failed' ? 'danger' : ['cancelled', 'expired'].includes(job?.status || '') ? 'neutral' : ['retry_wait', 'paused_for_release'].includes(job?.status || '') ? 'warning' : 'processing';
  return <div ref={root} className="result-workspace" aria-busy={loading && !job}>
    <div className="review-main" inert={toolsOpen}>
    <div className="result-topbar"><button type="button" className="text-link back-link" onClick={onBack}><Icon name="arrow" size={16} />我的任务</button>
      <h2 className="result-filename" title={job?.filename}>{job?.filename || '回看与校对'}</h2>
      {job && <span className={`badge ${taskTone}`}>{jobStatusLabel(job)}</span>}
      {result && <button ref={toolsButton} type="button" className="button secondary small tools-toggle" aria-expanded={toolsOpen} aria-controls="review-tools" aria-haspopup="dialog" onClick={() => setToolsOpen(true)}><Icon name="panel" size={16} />工具</button>}
    </div>
    {error && <div className="banner error" role="alert"><Icon name="alert" size={18} /><span>{error}</span></div>}
    {notice && <div className="banner neutral" role="status"><span>{notice}</span><button type="button" className="icon-button" aria-label="关闭提示" onClick={() => setNotice('')}><Icon name="close" size={16} /></button></div>}
    {!job ? loading ? <div className="panel loading-panel result-load-state" role="status"><div className="loading-skeleton" aria-hidden="true"><span /><span /><span /></div><p>正在读取任务…</p></div> : <div className="panel small-empty result-error-state"><Icon name="alert" size={24} /><h3>暂时无法读取任务</h3><p>请返回任务列表，确认任务状态后重试。</p></div> : <>
      <div className="review-body"><div className="review-media">
      <section className="audio-panel" aria-label="媒体回看">
        <div className="media-toolbar">
          {videoReady && <div className="segmented media-switch" role="group" aria-label="播放方式"><button type="button" className={useVideo ? 'selected' : ''} aria-pressed={useVideo} onClick={() => switchMedia('video')}>视频</button><button type="button" className={!useVideo ? 'selected' : ''} aria-pressed={!useVideo} onClick={() => switchMedia('audio')}>音频</button></div>}
          <details className="file-details"><summary>文件信息</summary><div className="file-details-content">
            <div className="metadata-list"><span>{date(job.created_at)} 创建</span>{job.expires_at && <span>{date(job.expires_at)} 到期</span>}</div>
            {useVideo ? <div className="audio-format"><span>{job.video_asset?.width} × {job.video_asset?.height}</span><span>{bytes(job.video_asset?.size_bytes || 0)}</span></div> : <div className="audio-format"><strong>高质量回听</strong><span>MP3 192 kbps</span>{audioReady && <div className="audio-specs metadata-list">{!!asset.sample_rate && <span>{asset.sample_rate / 1000} kHz</span>}{asset.channels === 1 ? <span>单声道</span> : asset.channels === 2 ? <span>双声道</span> : null}{asset.size_bytes !== undefined && <span>{bytes(asset.size_bytes)}</span>}{asset.duration_seconds !== undefined && <span>{clock(asset.duration_seconds)}</span>}</div>}</div>}
          </div></details>
        </div>
        {!useVideo && !audioReady && <p className="footnote" role="status">{asset?.state === 'failed' ? '音频准备失败，重试任务可重新处理' : '正在准备回听音频，准备完成后自动可播放'}</p>}
        <div className="media-stage">
        {useVideo ? <video key={videoSource} ref={video} src={videoSource} controls playsInline preload="metadata" className="retained-video" onTimeUpdate={() => setPosition(Math.max(0, (video.current?.currentTime || 0) - offset))} onLoadedMetadata={() => { if (video.current) video.current.currentTime = Math.max(0, position + offset); }} onError={() => setAudioError(true)} />
          : <audio key={audioSource} ref={audio} src={audioSource} controls preload="metadata" onTimeUpdate={() => setPosition(audio.current?.currentTime ?? 0)} onLoadedMetadata={() => { if (audio.current) audio.current.currentTime = Math.min(position, audio.current.duration || position); }} onError={() => { if (audioReady) setAudioError(true); }} />}
        </div>
        {videoPending(job) && <VideoProgress job={job} />}
        {job.video_asset?.state === 'failed' && <p className="footnote" role="alert">视频准备失败，音频和转录仍可使用。{job.video_asset.error_code === 'video_format_unsupported' && '当前视频为 HDR，请提供 SDR 版本。'}{terminal(job.status) && <button type="button" className="text-link" onClick={() => void retryVideo()}>仅重试视频</button>}</p>}
        {['preparing', 'queued'].includes(job.video_asset?.state || '') && terminal(job.status) && <p role="status">转录稿可继续使用。<button type="button" className="text-link" onClick={() => void cancelVideo()}>取消视频处理</button></p>}
        {(asset?.source_audio_streams ?? 0) > 1 && <p className="footnote">素材包含 {asset?.source_audio_streams} 条音轨，已使用首条音轨（音轨索引 {asset?.source_stream_index ?? 0}）。</p>}
        {audioError && <p className="footnote" role="alert">媒体暂时无法加载。<button type="button" className="text-link" onClick={() => { setAudioError(false); media()?.load(); }}>重新加载</button></p>}
        {!terminal(job.status) && !showingVideoProgress(job) && <div className="processing-note" role="status"><span className="spinner" aria-hidden="true" /><div className="processing-copy"><strong>{phaseName(job.progress?.phase || job.status)}</strong>{progressText(job) ? <span>{progressText(job)}</span> : job.queue_position ? <span>前方 {job.queue_position} 个任务</span> : null}</div><small>{job.next_attempt_at ? `已保存进度，下次尝试 ${date(job.next_attempt_at)}` : '可查看已保存文字，处理进度将自动更新'}</small></div>}
        {job.error && <p className="inline-error" role="alert">{errorNames[job.error.code] || job.error.message}</p>}
        {progressText(job) && terminal(job.status) && <p className="footnote">{progressText(job)}</p>}
        {canRetry(job) && <button type="button" className="button secondary small" disabled={recovering} aria-busy={recovering} onClick={() => void resume()}><Icon name="refresh" size={16} />{recovering ? '正在加入队列…' : retryLabel(job)}</button>}
        {(job.next_action === 'listen_or_change_parameters' || job.retryability === 'exhausted' || (typeof job.retryability === 'object' && job.retryability.reason === 'review_budget_exhausted')) && <p className="footnote">当前参数的自动复核额度已用完。请回听标记区间，确认内容或提供新的处理条件。</p>}
      </section>
      {incomplete && <section className="panel gap-panel" aria-label="待回听确认区间"><div className="section-heading"><div><h3>已处理，部分声音待回听确认</h3><p className="muted">{progressText(job)}。可用文字已保存，以下区间可直接回听。自动检测标记是待确认的疑点，是否存在漏词需回听判断。</p></div><span className="count">{coverage.gap_count ?? gaps.length}</span></div>{gaps.length > 0 && <div className="issue-list">{gaps.slice(gapPage * 20, (gapPage + 1) * 20).map(gap => <button type="button" className="issue-item" key={`${gap.chunk_index}:${gap.start}:${gap.end}`} onClick={() => seek(gap.start)}><Icon name="alert" size={17} /><span><strong>{clock(gap.start)} 至 {clock(gap.end)}</strong><small>第 {gap.chunk_index + 1} 段，{issueText({ kind: gap.reason })}</small></span><Icon name="play" size={12} /></button>)}</div>}{gaps.length > 20 && <Pager page={gapPage} count={gaps.length} setPage={setGapPage} />}</section>}
      </div>
      <div className="result-document">{result ? <section className="panel transcript-panel" aria-label="转录正文"><div className="document-toolbar"><div className="segmented" role="group" aria-label="文本视图"><button type="button" className={!raw ? 'selected' : ''} aria-pressed={!raw} onClick={() => setRaw(false)}>交付稿</button><button type="button" className={raw ? 'selected' : ''} aria-pressed={raw} onClick={() => setRaw(true)}>模型原稿</button></div><div className="document-tools"><span className="subtle-label">当前 v{result.version}</span><button type="button" className="button ghost small" onClick={() => void copy()}>复制全文</button></div></div>
        {incomplete && <p className="document-warning" role="status">部分声音待回听确认，当前为不完整结果。</p>}
        {result.speaker_count_mismatch && <p className="footnote">检测到的人物数量与预计 {result.expected_speakers} 人不同，请回看确认。</p>}
        {result.diarization?.status === 'partial' && segments.some(s => pendingSpeaker(s, result.speakers || {}, raw)) && <p className="footnote">部分片段归属待确认{editable && !raw ? '，可筛选「待确认」后批量指派' : ''}。</p>}
        {segments.length ? <Transcript speakers={result.speakers} segments={segments} raw={raw} position={position} seek={seek}
          onEdit={canEdit ? beginEdit : undefined} issues={issues} jump={jump}
          onAssign={editable && capabilities?.diarization && !edit
            ? (ids, speakerId) => bulkEdit('speakers/assign-pending', { segment_ids: ids, speaker_id: speakerId }, `已指派 ${ids.length} 段`) : undefined}
        /> : <VirtualText text={(raw ? result.raw_text ?? result.text : result.text) || ''} />}
      </section> : <section className="panel waiting-result" role="status"><Icon name="wave" size={34} /><h3>{job.status === 'failed' ? '任务未完成' : job.status === 'cancelled' ? '任务已取消' : job.status === 'expired' ? '任务已过期' : job.status === 'completed' ? '暂无可用文字' : '等待转录结果'}</h3><p>{job.status === 'failed' ? '已有处理进度会保留。返回任务列表查看错误或重试。' : terminal(job.status) ? '此任务暂无可用结果，请返回任务列表查看任务状态。' : '任务完成后，回听、校对和导出工具会显示在这里。'}</p>{job.progress?.text && <div className="partial-text"><strong>已保存片段，仍有声音待确认</strong><VirtualText text={job.progress.text} /><div className="partial-export"><select aria-label="部分结果格式" value={partialFormat} onChange={e => setPartialFormat(e.target.value)}><option value="txt">TXT 含待确认区间</option><option value="json">JSON 含完整性信息</option></select><button type="button" className="button secondary small" disabled={exporting} aria-busy={exporting} onClick={() => void download()}>{exporting ? '正在导出…' : '导出部分结果'}</button></div></div>}</section>}</div></div>
    </>}
    </div>
    {result && <div className="tools-layer" hidden={!toolsOpen} onClick={() => setToolsOpen(false)}><div className="tools-surface" onClick={e => e.stopPropagation()}>
<aside ref={tools} id="review-tools" className="result-aside" role="dialog" aria-modal="true" aria-labelledby="review-tools-title">
        <div className="tools-heading"><h2 id="review-tools-title">工具</h2><button ref={toolsClose} type="button" className="icon-button" aria-label="关闭工具" onClick={() => setToolsOpen(false)}><Icon name="close" size={20} /></button></div>
        {error && <div className="banner error" role="alert"><Icon name="alert" size={18} /><span>{error}</span></div>}
        <ReplaceTools key={jobId} result={result} disabled={!editable || !!edit}
          onReplace={(find, replacement) => bulkEdit('replace-text', { find, replacement }, '已替换并保存新版本')} />
        {capabilities?.diarization && <SpeakerTools job={job} onSaved={speakersSaved} disabled={!canEdit || !!edit} />}<section className="panel export-panel"><div className="section-heading"><h3>导出结果</h3><Icon name="download" size={19} /></div><label className="field-label">文本版本<select value={exportVersion ?? result.version} onChange={e => setExportVersion(Number(e.target.value))}>{(result.versions?.length ? result.versions : [{ version: result.version, created_at: '', alignment_status: result.alignment_status }]).map(v => <option key={v.version} value={v.version}>v{v.version}{v.version === result.version ? '（当前版本）' : ''}{v.alignment_status === 'pending' ? ' 字幕更新中' : ''}</option>)}</select></label><label className="field-label">文件格式<select value={format} onChange={e => setFormat(e.target.value)}><option value="md">Markdown 对话稿（供 Agent 分析）</option><option value="txt">TXT 纯文本</option>{canSubtitle && <><option value="srt">SRT 字幕</option><option value="vtt">VTT 网页字幕</option></>}<option value="json">JSON 完整结果</option></select></label><button type="button" className="button primary wide" disabled={exporting || subtitleBlocked} aria-busy={exporting} aria-describedby={subtitleBlocked ? 'subtitle-blocked-reason' : undefined} onClick={() => void download()}><Icon name="download" size={17} />{exporting ? '正在导出…' : '下载所选版本'}</button>{subtitleBlocked && <p className="footnote" id="subtitle-blocked-reason">{incomplete ? '此结果仍有待确认区间，字幕暂不可导出，可先导出含区间说明的 TXT / JSON。' : versionStatus === 'failed' ? '此版本字幕对齐失败，可先导出文字或选择此前已对齐的版本。' : '字幕正在更新，可先导出文字或选择此前已对齐的版本。'}</p>}<p className="footnote">{incomplete ? '部分导出会保留完整性标记及待确认区间的时间。' : '导出使用所选版本的交付稿。'}模型原稿可在正文区查看、复制，也保留在完整 JSON 中。</p></section>
        {result.alignment_status !== 'aligned' && !segmentTimed && <div className="banner neutral" role="status"><Icon name="refresh" size={18} /><span>{result.alignment_status === 'pending' ? '字幕更新中' : '字幕暂未完成对齐'}</span></div>}
        <section className="panel review-panel"><div className="section-heading"><h3>回听清单</h3><span className="count">{issues.length}</span></div>{issues.length === 0 ? <p className="muted">系统未标记待复核问题。你仍可以回听音频进行校对。</p> : <div className="issue-list">{issues.slice(issuePage * 20, (issuePage + 1) * 20).map((issue, i) => <button type="button" className="issue-item" key={i} onClick={() => { seek(typeof issue !== 'string' ? issue.start ?? 0 : 0); setToolsOpen(false); }}><Icon name="alert" size={17} /><span><strong>{typeof issue !== 'string' && issue.start !== undefined ? clock(issue.start) : '需要确认'}</strong><small>{issueText(issue)}</small></span><Icon name="play" size={12} /></button>)}</div>}{issues.length > 20 && <Pager page={issuePage} count={issues.length} setPage={setIssuePage} />}</section>
        {!!result.changes?.length && <section className="panel"><h3>词汇与修改记录</h3><p className="muted">已记录 {result.changes.length} 项调整，原始识别稿保持可对照。</p><div className="changes-list">{result.changes.slice(changePage * 20, (changePage + 1) * 20).map((change, i) => { const value = change as { before?: string; after?: string; original?: string; replacement?: string; from?: string; to?: string; source?: string }; const before = value.before ?? value.original ?? value.from; const after = value.after ?? value.replacement ?? value.to; return <div key={i}>{before && after ? <><span>{before}</span><Icon name="arrow" size={12} /><strong>{after}</strong></> : <span>第 {i + 1} 项修改已保存</span>}</div>; })}</div>{result.changes.length > 20 && <Pager page={changePage} count={result.changes.length} setPage={setChangePage} />}</section>}
      </aside>
    </div></div>}
    {edit && <div ref={modal} className="modal-backdrop" role="presentation" onClick={() => { if (!saving) setEdit(null); }}><section className="edit-modal" role="dialog" aria-modal="true" aria-labelledby="edit-title" aria-busy={saving} onClick={e => e.stopPropagation()}><div className="section-heading"><h2 id="edit-title">校对片段</h2><button type="button" className="icon-button" disabled={saving} aria-label="关闭编辑" onClick={() => setEdit(null)}><Icon name="close" size={20} /></button></div><button type="button" className="text-link" onClick={() => seek(edit.start)}><Icon name="play" size={14} />回听 {clock(edit.start)} 至 {clock(edit.end)}</button>{capabilities?.diarization && <label className="field-label">说话人<select value={speakerDraft} onChange={e => setSpeakerDraft(e.target.value)}><option value="">待确认</option>{Object.entries(job?.result?.speakers || {}).map(([id, person]) => <option key={id} value={id}>{id} {person.name}</option>)}</select></label>}<label className="field-label">片段内容<textarea value={draft} onChange={e => setDraft(e.target.value)} rows={7} autoFocus /></label><p className="footnote">{capabilities?.forced_alignment ? '修改文字会重新对齐；只改说话人不需要推理。' : '文字修改沿用该段起止时间；不代表逐字对齐。'}原稿及先前版本会保留。</p>{error && <p className="inline-error" role="alert">{error}</p>}<div className="modal-actions"><button type="button" className="button secondary" disabled={saving} onClick={() => setEdit(null)}>取消</button><button type="button" className="button primary" disabled={saving || !draft.trim() || (draft.trim() === edit.text.trim() && !capabilities?.diarization)} onClick={() => void save()}>{saving ? '保存中…' : '保存校对'}</button></div></section></div>}
  </div>;
}

function Pager({ page, count, setPage }: { page: number; count: number; setPage: (page: number) => void }) {
  return <div className="aside-pager"><button type="button" className="button ghost small" disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</button><span>{page + 1} / {Math.ceil(count / 20)}</span><button type="button" className="button ghost small" disabled={(page + 1) * 20 >= count} onClick={() => setPage(page + 1)}>下一页</button></div>;
}
