import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api, bytes, date, errorNames, message, phaseName, terminal, type Capabilities, type Job, type JobPage, type User } from './api';
import { Icon, type IconName } from './icons';
import UploadPane from './UploadPane';
import ResultPane from './ResultPane';
import AdminPane from './AdminPane';

type Page = 'upload' | 'jobs' | 'result' | 'admin';

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [page, setPage] = useState<Page>('upload');
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busyJob, setBusyJob] = useState('');
  const polling = useRef(false);
  const currentUser = useRef<User | null>(user);
  const currentJobs = useRef<Job[]>(jobs);
  currentUser.current = user;
  currentJobs.current = jobs;

  useEffect(() => {
    api<{ user: User }>('/auth/me', {}, true).then(v => setUser(v.user)).catch(() => {}).finally(() => setReady(true));
    const unauth = () => { setUser(null); setCaps(null); setJobs([]); setActiveId(null); setError('登录已失效，请重新登录。'); };
    window.addEventListener('asr:unauthorized', unauth);
    return () => window.removeEventListener('asr:unauthorized', unauth);
  }, []);

  const refresh = useCallback(async () => {
    if (polling.current) return;
    polling.current = true;
    const expectedUser = currentUser.current?.id;
    try {
      const data = await api<JobPage>('/jobs?view=summary&limit=50');
      if (currentUser.current?.id !== expectedUser) return;
      setJobs(data.jobs);
      setNextCursor(data.next_cursor);
      setError('');
    } catch (e) { setError(message(e)); }
    finally { polling.current = false; }
  }, []);

  useEffect(() => {
    if (!user) return;
    let alive = true;
    api<Capabilities>('/capabilities').then(value => { if (alive) setCaps(value); }).catch(e => { if (alive) setError(message(e)); });
    void refresh();
    let timer: number;
    const poll = async () => {
      const active = currentJobs.current.filter(task => !terminal(task.status) || task.alignment_status === 'pending').slice(0, 100);
      if (document.visibilityState === 'visible' && active.length && !polling.current) {
        polling.current = true;
        try {
          const data = await api<JobPage>(`/jobs?view=summary&limit=100&ids=${active.map(task => encodeURIComponent(task.id)).join(',')}`);
          if (alive) {
            const updated = new Map(data.jobs.map(task => [task.id, task]));
            const watched = new Set(active.map(task => task.id));
            setJobs(previous => previous.filter(task => !watched.has(task.id) || updated.has(task.id)).map(task => updated.get(task.id) ?? task));
          }
        } catch (e) { if (alive) setError(message(e)); }
        finally { polling.current = false; }
      }
      if (alive) timer = window.setTimeout(() => void poll(), document.visibilityState === 'hidden' ? 30000 : 3500);
    };
    const visible = () => { if (document.visibilityState === 'visible') void refresh(); };
    timer = window.setTimeout(() => void poll(), 3500);
    document.addEventListener('visibilitychange', visible);
    return () => { alive = false; window.clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [user, refresh]);

  async function loadMore() {
    if (nextCursor === null || loadingMore) return;
    setLoadingMore(true);
    const expectedUser = currentUser.current?.id;
    try {
      const data = await api<JobPage>(`/jobs?view=summary&limit=50&before=${nextCursor}`);
      if (currentUser.current?.id !== expectedUser) return;
      setJobs(previous => [...previous, ...data.jobs.filter(task => !previous.some(old => old.id === task.id))]);
      setNextCursor(data.next_cursor);
    } catch (e) { setError(message(e)); }
    finally { setLoadingMore(false); }
  }

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(''), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function jobAction(job: Job, action: 'cancel' | 'retry' | 'delete') {
    if (action === 'delete' && !window.confirm(`删除「${job.filename}」及其音频、结果和编辑记录？`)) return;
    setBusyJob(job.id);
    try {
      await api(action === 'delete' ? `/jobs/${job.id}` : `/jobs/${job.id}/${action}`, { method: action === 'delete' ? 'DELETE' : 'POST' });
      if (action === 'delete' && activeId === job.id) { setActiveId(null); setPage('jobs'); }
      await refresh();
      setNotice(action === 'delete' ? '任务已删除' : action === 'cancel' ? '已请求取消，当前片段结束后生效' : '已重新加入队列');
    } catch (e) { setError(message(e)); } finally { setBusyJob(''); }
  }

  async function logout() {
    try { await api('/auth/logout', { method: 'POST' }); setUser(null); setJobs([]); setCaps(null); setPage('upload'); }
    catch (e) { setError(message(e)); }
  }

  if (!ready) return <div className="boot"><div className="brand-mark"><Icon name="wave" size={28} /></div><p>正在连接工作台…</p></div>;
  if (!user) return <Login initialError={error} onLogin={v => { setUser(v); setError(''); }} />;

  const nav: { id: Page; icon: IconName; title: string }[] = [{ id: 'upload', icon: 'upload', title: '新建转录' }, { id: 'jobs', icon: 'list', title: '我的任务' }, ...(user.is_admin ? [{ id: 'admin' as Page, icon: 'settings' as IconName, title: '管理工作台' }] : [])];
  const running = jobs.filter(j => !terminal(j.status)).length;
  const selected = jobs.find(j => j.id === activeId);
  const titles: Record<Page, [string, string]> = {
    upload: ['让声音，成为可用的文字。', '上传素材，剩下的交给工作台。'], jobs: ['每一段声音，都有迹可循。', '查看处理进度，回到你的转录结果。'],
    result: ['回听、校对，然后交付。', selected?.filename || '转录结果'], admin: ['管理工作台', '维护账户、词库和共享资源。'],
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <a href="#upload" className="brand" onClick={e => { e.preventDefault(); setPage('upload'); }}><span className="brand-mark"><Icon name="wave" size={25} /></span><span>声笺<small>ASR WORKSPACE</small></span></a>
      <div className="workspace-label">我的工作空间</div>
      <nav aria-label="主导航">{nav.map(n => <button key={n.id} className={`nav-item ${page === n.id ? 'active' : ''}`} onClick={() => setPage(n.id)}><Icon name={n.icon} /><span>{n.title}</span>{n.id === 'jobs' && running > 0 && <b className="nav-count">{running}</b>}</button>)}</nav>
      <div className="sidebar-foot"><div className="local-note"><span className="status-dot" />局域网工作台<small>音频与结果在本地服务器处理</small></div><div className="user-row"><span className="avatar">{user.username.slice(0, 1).toUpperCase()}</span><span className="user-name">{user.username}<small>{user.is_admin ? '管理员' : '个人空间'}</small></span><button className="icon-button" aria-label="退出登录" title="退出登录" onClick={() => void logout()}><Icon name="logout" size={18} /></button></div></div>
    </aside>
    <main className="main-content">
      <div className="topline"><span>工作空间 <span className="slash">/</span> {page === 'result' ? '转录结果' : nav.find(n => n.id === page)?.title}</span><span className="topline-right"><span className="status-dot" /> 本地转录<button className="icon-button mobile-logout" aria-label="退出登录" onClick={() => void logout()}><Icon name="logout" size={16} /></button></span></div>
      <header className="page-header"><div><div className="eyebrow">{page === 'upload' ? 'FROM AUDIO TO WORDS' : 'YOUR AUDIO WORKSPACE'}</div><h1>{titles[page][0]}</h1><p>{titles[page][1]}</p></div>{page === 'jobs' && <button className="button primary" onClick={() => setPage('upload')}><Icon name="plus" size={18} />新建转录</button>}</header>
      {error && <div className="banner error" role="alert"><Icon name="alert" size={18} /><span>{error}</span><button className="icon-button" aria-label="关闭提示" onClick={() => setError('')}><Icon name="close" size={16} /></button></div>}
      {notice && <div className="toast" role="status"><Icon name="check" size={18} />{notice}</div>}
      <section hidden={page !== 'upload'}><UploadPane user={user} caps={caps} onSubmitted={() => { void refresh(); api<Capabilities>('/capabilities').then(setCaps).catch(() => {}); setNotice('素材已进入转录队列'); }} onOpenJobs={() => setPage('jobs')} /></section>
      {page === 'jobs' && <><div className="section-heading"><h2>我的任务 <span className="count">{jobs.length}</span></h2><button className="button ghost small" onClick={() => void refresh()}><Icon name="refresh" size={16} />刷新</button></div>{jobs.length === 0 ? <Empty title="还没有转录任务" detail="上传一段音频或视频，你的结果会保存在这里。" action={() => setPage('upload')} /> : <div className="job-list">{jobs.map(job => <JobCard key={job.id} job={job} busy={busyJob === job.id} onOpen={() => { setActiveId(job.id); setPage('result'); }} onAction={a => void jobAction(job, a)} />)}</div>}{nextCursor !== null && <button className="button secondary" disabled={loadingMore} onClick={() => void loadMore()}>{loadingMore ? '正在读取…' : '加载更多任务'}</button>}<p className="footnote">任务结束后保留 {caps?.retention_days ?? 7} 天。删除后将同时移除音频、文字和编辑记录。</p></>}
      {page === 'result' && activeId && <ResultPane jobId={activeId} models={caps?.models ?? []} onBack={() => setPage('jobs')} onChanged={() => void refresh()} />}
      {page === 'admin' && user.is_admin && <AdminPane />}
      <footer className="page-footer"><span>声笺 <span className="muted">·</span> 把精力留给内容</span><span>本地处理 · 忠实转录 · 可回听校对</span></footer>
    </main>
  </div>;
}

function Login({ initialError, onLogin }: { initialError: string; onLogin: (user: User) => void }) {
  const [username, setUsername] = useState(''); const [password, setPassword] = useState(''); const [error, setError] = useState(initialError); const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent) { e.preventDefault(); setBusy(true); setError(''); try { const r = await api<{ user: User }>('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }, true); setPassword(''); onLogin(r.user); } catch (e) { setError(message(e)); } finally { setBusy(false); } }
  return <main className="login-shell"><section className="login-story"><div className="brand light"><span className="brand-mark"><Icon name="wave" size={27} /></span><span>声笺<small>ASR WORKSPACE</small></span></div><div><div className="eyebrow">LISTEN. TRANSCRIBE. CREATE.</div><h1>声音里的想法，<br />值得被好好记录。</h1><p>从一段口播，到一整期播客。<br />在本地完成转录、回听和字幕交付。</p><div className="sound-decoration" aria-hidden="true">{[16, 28, 45, 31, 65, 84, 56, 97, 72, 44, 68, 88, 54, 34, 46, 23, 14].map((h, i) => <i key={i} style={{ height: `${h}px` }} />)}</div></div><span className="login-caption">YOUR WORDS, CLOSE TO HOME.</span></section><section className="login-form-wrap"><form className="login-form" onSubmit={e => void submit(e)}><div className="eyebrow">WELCOME BACK</div><h2>进入你的工作空间</h2><p>使用管理员为你创建的账户登录。</p><label>用户名<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required autoFocus placeholder="输入用户名" /></label><label>密码<input value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password" type="password" required placeholder="输入密码" /></label>{error && <div className="inline-error" role="alert">{error}</div>}<button type="submit" className="button primary wide" disabled={busy}>{busy ? '正在登录…' : '进入工作台'}<Icon name="arrow" size={18} /></button><p className="login-help">需要账户或重置密码？请联系工作台管理员。</p></form></section></main>;
}

export function Empty({ title, detail, action }: { title: string; detail: string; action?: () => void }) { return <div className="empty-state"><div className="empty-icon"><Icon name="headphones" size={28} /></div><h3>{title}</h3><p>{detail}</p>{action && <button className="button secondary" onClick={action}>上传第一份素材<Icon name="arrow" size={16} /></button>}</div>; }

function JobCard({ job, busy, onOpen, onAction }: { job: Job; busy: boolean; onOpen: () => void; onAction: (a: 'cancel' | 'retry' | 'delete') => void }) {
  const progress = job.progress; const active = !terminal(job.status); const percent = Math.max(0, Math.min(100, progress?.percent ?? 0));
  return <article className="job-card"><div className="file-icon"><Icon name="file" size={24} /></div><div className="job-main"><div className="job-title-row"><button className="text-link job-title" onClick={onOpen}>{job.filename}</button><span className={`badge ${job.status === 'completed' ? 'success' : job.status === 'failed' ? 'danger' : active ? 'processing' : ''}`}>{phaseName(job.status)}</span></div><div className="job-meta"><span>{date(job.created_at)}</span>{job.queue_position ? <span>前方 {job.queue_position} 个任务</span> : progress?.total_chunks ? <span>已完成 {progress.completed_chunks ?? 0} / {progress.total_chunks} 段</span> : null}{job.expires_at && <span>{date(job.expires_at)} 到期</span>}</div>{active && <div className="job-progress"><div className="progress-track"><span style={{ width: `${percent}%` }} /></div><small>{phaseName(progress?.phase || job.status)}{progress?.total_chunks ? ` · ${progress.completed_chunks ?? 0}/${progress.total_chunks}` : ''}</small></div>}{job.error && <p className="inline-error">{errorNames[job.error.code] || job.error.message}</p>}</div><div className="job-actions"><button className="button secondary small" onClick={onOpen}>{job.result || job.has_result ? '查看结果' : '查看任务'}<Icon name="arrow" size={15} /></button>{active ? <button className="button ghost small" disabled={busy || job.status === 'cancel_requested'} onClick={() => onAction('cancel')}>取消</button> : <>{['failed', 'cancelled'].includes(job.status) && <button className="button ghost small" disabled={busy} onClick={() => onAction('retry')}>重试</button>}<button className="button ghost small danger-text" disabled={busy} onClick={() => onAction('delete')}>删除</button></>}</div></article>;
}

export function StorageMeter({ used, quota }: { used: number; quota: number }) { return <div className="storage-meter"><div><span>已用空间</span><strong>{bytes(used)} <small>/ {bytes(quota)}</small></strong></div><div className="progress-track"><span style={{ width: `${Math.min(100, quota ? used / quota * 100 : 0)}%` }} /></div></div>; }
