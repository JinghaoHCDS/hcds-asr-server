import { useEffect, useRef, useState, type DragEvent } from 'react';
import { sha256 } from '@noble/hashes/sha2.js';
import { ApiError, api, bytes, message, type Capabilities, type Job, type Upload, type User } from './api';
import { Icon } from './icons';
import { StorageMeter } from './App';

interface SavedUpload { id: string; filename: string; size: number; lastModified: number; hash: string }
interface UploadRow { key: string; file: File; state: string; percent: number; error?: string; job?: Job }
const ALLOWED = /\.(wav|mp3|m4a|aac|flac|ogg|opus|aiff|aif|wma|mp4|mov|mkv|webm|avi|m4v)$/i;
const rowKey = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
function readSaved(key: string): SavedUpload[] { try { const values: unknown = JSON.parse(localStorage.getItem(key) || '[]'); return Array.isArray(values) ? values.filter((v): v is SavedUpload => !!v && typeof v.id === 'string' && typeof v.hash === 'string' && typeof v.filename === 'string' && Number.isFinite(v.size)) : []; } catch { return []; } }

export default function UploadPane({ user, caps, onSubmitted, onOpenJobs }: { user: User; caps: Capabilities | null; onSubmitted: () => void; onOpenJobs: () => void }) {
  const storageKey = `asr.uploads.${user.id}`;
  const [saved, setSaved] = useState(() => readSaved(storageKey));
  const [rows, setRows] = useState<UploadRow[]>([]);
  const [domain, setDomain] = useState('ai');
  const [hotwords, setHotwords] = useState('');
  const [language, setLanguage] = useState('auto');
  const [model, setModel] = useState('');
  const [profile, setProfile] = useState('');
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState('');
  const picker = useRef<HTMLInputElement>(null);
  const savedRef = useRef(saved);
  const abort = useRef(false);
  useEffect(() => () => { abort.current = true; }, []);

  const availableModels = caps?.models.filter(candidate => candidate.pipelines?.length > 0) || [];
  const selectedModel = availableModels.find(candidate => candidate.id === model)
    || availableModels.find(candidate => candidate.id === caps?.default_model)
    || availableModels.find(candidate => candidate.default) || availableModels[0];
  const availableProfiles = (caps?.quality_profiles || []).filter(candidate => selectedModel?.pipelines.includes(candidate.id)
    && (candidate.id !== 'quality_v1' || selectedModel.capabilities.forced_alignment));
  const selectedProfile = availableProfiles.find(candidate => candidate.id === profile)?.id
    || availableProfiles.find(candidate => candidate.id === caps?.default_quality_profile)?.id || availableProfiles[0]?.id;
  const availableLanguages = selectedModel?.capabilities.languages || ['auto'];
  const selectedLanguage = availableLanguages.includes(language) ? language : availableLanguages[0] || 'auto';
  const canAlign = selectedModel?.capabilities.forced_alignment || selectedModel?.capabilities.segment_timestamps;

  function persist(next: SavedUpload[]) { savedRef.current = next; setSaved(next); try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* Server offset remains authoritative if browser storage is unavailable. */ } }
  function update(key: string, values: Partial<UploadRow>) { setRows(old => old.map(r => r.key === key ? { ...r, ...values } : r)); }
  function addFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter(f => ALLOWED.test(f.name) || f.type.startsWith('audio/') || f.type.startsWith('video/'));
    if (accepted.length !== files.length) setError('部分文件不是支持的音视频格式，请检查扩展名。'); else setError('');
    setRows(old => [...old, ...accepted.map(file => ({ key: rowKey(), file, state: '等待上传', percent: 0 }))]);
  }
  function drop(e: DragEvent) { e.preventDefault(); setDrag(false); if (!busy) addFiles(e.dataTransfer.files); }
  async function removeUpload(record: SavedUpload) {
    setError('');
    try { await api(`/uploads/${record.id}`, { method: 'DELETE' }); persist(savedRef.current.filter(v => v.id !== record.id)); }
    catch (e) { if (e instanceof ApiError && e.status === 404) persist(savedRef.current.filter(v => v.id !== record.id)); else setError(message(e)); }
  }

  async function transfer(row: UploadRow) {
    if (!selectedModel || !selectedProfile) throw new Error('当前模型没有可用的转录流程，请检查服务配置');
    if (caps && row.file.size > caps.max_upload_bytes) throw new Error(`文件超过 ${bytes(caps.max_upload_bytes)} 的上传限制`);
    update(row.key, { state: '校验文件', percent: 0, error: undefined });
    const hasher = sha256.create();
    const block = 8 * 1024 * 1024;
    for (let offset = 0; offset < row.file.size; offset += block) {
      if (abort.current) throw new Error('上传已暂停，重新选择相同文件可继续');
      hasher.update(new Uint8Array(await row.file.slice(offset, offset + block).arrayBuffer()));
      update(row.key, { percent: row.file.size ? Math.min(100, (offset + block) / row.file.size * 100) : 0 });
    }
    const hash = Array.from(hasher.digest()).map(v => v.toString(16).padStart(2, '0')).join('');
    if (abort.current) throw new Error('上传已暂停，重新选择相同文件可继续');
    let previous = savedRef.current.find(v => v.filename === row.file.name && v.size === row.file.size && v.lastModified === row.file.lastModified && v.hash === hash);
    let upload: Upload;
    let record: SavedUpload;
    if (previous) {
      try { upload = await api<Upload>(`/uploads/${previous.id}`); }
      catch (e) {
        if (!(e instanceof ApiError) || e.status !== 404) throw e;
        const expiredId = previous.id; persist(savedRef.current.filter(v => v.id !== expiredId)); previous = undefined;
      }
    }
    if (previous) record = previous;
    else {
      upload = await api<Upload>('/uploads', { method: 'POST', body: JSON.stringify({ filename: row.file.name, size_bytes: row.file.size }) });
      record = { id: upload.id, filename: row.file.name, size: row.file.size, lastModified: row.file.lastModified, hash };
      persist([...savedRef.current, record]);
    }
    if (upload!.job_id) {
      const job = await api<Job>(`/jobs/${upload!.job_id}`);
      persist(savedRef.current.filter(v => v.id !== record.id));
      update(row.key, { state: '任务已提交', percent: 100, job }); onSubmitted(); return;
    }
    while (upload!.offset < row.file.size) {
      if (abort.current) throw new Error('上传已暂停，重新选择相同文件可继续');
      update(row.key, { state: '上传中', percent: row.file.size ? upload!.offset / row.file.size * 100 : 0 });
      const currentOffset = upload!.offset;
      const body = await row.file.slice(currentOffset, Math.min(currentOffset + upload!.chunk_size, row.file.size)).arrayBuffer();
      upload = await api<Upload>(`/uploads/${upload!.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/octet-stream', 'Upload-Offset': String(currentOffset) }, body });
      if (upload.offset <= currentOffset || upload.offset > row.file.size) throw new Error('服务器返回了异常的上传位置，请重新选择文件恢复上传');
    }
    if (abort.current) throw new Error('上传已暂停，重新选择相同文件可继续');
    update(row.key, { state: '提交任务', percent: 100 });
    const job = await api<Job>(`/uploads/${upload!.id}/complete`, { method: 'POST', body: JSON.stringify({ domain, hotwords, language: selectedLanguage,
      model: selectedModel.id, quality_profile: selectedProfile, idempotency_key: `upload:${record.id}` }) });
    persist(savedRef.current.filter(v => v.id !== record.id));
    update(row.key, { state: '已加入队列', percent: 100, job });
    onSubmitted();
  }

  async function start() {
    if (!caps) { setError('服务配置尚未加载，请稍后再试。'); return; }
    setBusy(true); abort.current = false; setError('');
    try {
      for (const row of rows.filter(v => !v.job)) {
        if (abort.current) break;
        try { await transfer(row); } catch (e) { update(row.key, { state: '待继续', error: message(e) }); }
      }
    } finally { setBusy(false); }
  }

  const count = rows.filter(v => !v.job).length;
  return <div className="upload-layout"><div className="upload-primary">
    <section className="panel upload-panel"><div className="panel-heading"><div className="step-number">01</div><div><h2>添加你的素材</h2><p>音频或视频，都从这里开始。</p></div><span className="badge">多文件排队</span></div>
      <input ref={picker} type="file" multiple accept="audio/*,video/*,.mkv,.flac,.m4a,.opus" className="visually-hidden" disabled={busy} onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = ''; }} />
      <button type="button" className={`drop-zone ${drag ? 'dragging' : ''}`} disabled={busy} onClick={() => picker.current?.click()} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={drop}><span className="upload-symbol"><Icon name="upload" size={28} /></span><strong>把音频或视频拖到这里</strong><span>或 <u>选择本地文件</u></span><small>MP3、WAV、M4A、MP4、MOV 等常见格式<br />最长 {caps ? Math.floor(caps.max_duration_seconds / 3600) : 6} 小时 · 单文件最高 {bytes(caps?.max_upload_bytes ?? 8 * 1024 ** 3)}</small></button>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {rows.length > 0 && <div className="upload-files">{rows.map(row => <div className="upload-file" key={row.key}><Icon name={row.job ? 'check' : 'file'} size={20} /><div><strong title={row.file.name}>{row.file.name}</strong><span>{bytes(row.file.size)} · {row.state}{['校验文件', '上传中'].includes(row.state) ? ` ${Math.floor(row.percent)}%` : ''}</span>{['校验文件', '上传中'].includes(row.state) && <div className="progress-track"><i style={{ width: `${row.percent}%` }} /></div>}{row.error && <span className="inline-error">{row.error}</span>}</div>{!busy && <button className="icon-button" aria-label={`移除 ${row.file.name}`} onClick={() => setRows(old => old.filter(v => v.key !== row.key))}><Icon name="close" size={16} /></button>}</div>)}</div>}
      {saved.length > 0 && <div className="resume-note"><Icon name="refresh" size={18} /><div><strong>{saved.length} 个上传可以继续</strong><p>重新选择下列原文件，校验后从服务器保存的位置继续。继续前请确认本页的领域和热词。</p>{saved.map(s => <div className="resume-file" key={s.id}><span>{s.filename} · {bytes(s.size)}</span><button className="text-link" disabled={busy} onClick={() => void removeUpload(s)}>取消上传并删除</button></div>)}</div></div>}
    </section>
    <section className="panel settings-panel"><div className="panel-heading"><div className="step-number">02</div><div><h2>让转录更懂你的内容</h2><p>选一个领域，需要时再补充本次的专有名词。</p></div></div>
      <div className="domain-options" role="group" aria-label="内容领域">{(caps?.domains || [{ id: 'general', label: '通用内容' }, { id: 'ai', label: 'AI / 技术内容' }]).map(d => <button key={d.id} disabled={busy} className={`domain-option ${domain === d.id ? 'selected' : ''}`} onClick={() => setDomain(d.id)}><span className="radio-dot" /><strong>{d.label}</strong>{domain === d.id && <Icon name="check" size={16} />}</button>)}</div>
      <label className="field-label">本次热词 <span className="optional">选填</span><textarea value={hotwords} disabled={busy} onChange={e => setHotwords(e.target.value)} rows={3} placeholder="例如：Qwen3-ASR, 混吃等鼠, Agent 工作流" /><small>填写人名、产品名或专业术语，用英文逗号分隔。词汇只作为识别参考。</small></label>
      <details className="advanced"><summary>更多设置</summary>
        <label className="field-label">转录模型<select value={selectedModel?.id || ''} disabled={busy} onChange={e => { setModel(e.target.value); setProfile(''); }}>{availableModels.map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.id}</option>)}</select></label>
        <label className="field-label">转录流程<select value={selectedProfile || ''} disabled={busy} onChange={e => setProfile(e.target.value)}>{availableProfiles.map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.label}</option>)}</select></label>
        <label className="field-label">音频语言<select value={selectedLanguage} disabled={busy} onChange={e => setLanguage(e.target.value)}>{availableLanguages.map(code => <option key={code} value={code}>{({ auto: '自动识别', zh: '中文', en: '英文' } as Record<string, string>)[code] || code}</option>)}</select></label>
      </details>
      <div className="submit-row"><span><Icon name="check" size={16} />{canAlign ? '自动转录与字幕对齐' : '自动转录'}</span>{busy ? <button className="button secondary" onClick={() => { abort.current = true; }}>暂停上传</button> : <button className="button primary" disabled={!count || !caps || !selectedModel || !selectedProfile} onClick={() => void start()}>开始转录{count > 0 && ` · ${count} 个文件`}<Icon name="arrow" size={18} /></button>}</div>
      {selectedProfile === 'quality_v1' && caps?.quality_profiles.some(p => p.id === 'quality_v1' && p.validated === false) && <p className="footnote">当前转录流程处于试运行阶段，正在进行真实长音频验收。可通过结果页回听、校对后交付。</p>}
      {rows.some(v => v.job) && <button className="text-link uploaded-link" onClick={onOpenJobs}>前往我的任务，查看处理进度 <Icon name="arrow" size={15} /></button>}
    </section>
  </div><aside className="upload-aside"><div className="guide-card"><div className="eyebrow">A QUIETER WORKFLOW</div><h3>从声音到成稿，<br />少一点来回。</h3><div className="guide-step"><b>1</b><div><strong>上传后，即可离开</strong><p>文件上传完成后，任务会在服务器继续处理。</p></div></div><div className="guide-step"><b>2</b><div><strong>完整转录，局部复核</strong><p>系统自动处理长音频，并检查需要复核的片段。</p></div></div><div className="guide-step"><b>3</b><div><strong>回听校对，导出交付</strong><p>点击文字回到原音，保留修改记录和字幕版本。</p></div></div><div className="retention-note"><Icon name="headphones" size={18} /><span>任务完成后保留 {caps?.retention_days ?? 7} 天<br /><small>可随时手动删除</small></span></div></div>{caps?.storage && <StorageMeter used={caps.storage.used_bytes} quota={caps.storage.quota_bytes} />}</aside></div>;
}
