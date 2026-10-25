export interface User { id: string; username: string; is_admin: boolean; disabled?: boolean }
export interface ModelOption {
  id: string;
  provider: string;
  default: boolean;
  engine_id: string;
  pipelines: string[];
  capabilities: { forced_alignment: boolean; timestamps: string[]; languages: string[]; segment_timestamps: boolean; diarization: boolean };
}
export interface Capabilities {
  domains: { id: string; label: string }[];
  quality_profiles: { id: string; label: string; validated?: boolean }[];
  models: ModelOption[];
  default_model?: string;
  default_quality_profile?: string;
  max_upload_bytes: number;
  chunk_size: number;
  max_duration_seconds: number;
  retention_days: number;
  storage: { used_bytes: number; quota_bytes: number };
}
export interface Issue { code?: string; kind?: string; type?: string; message?: string; start?: number; end?: number; chunk_index?: number }
export interface Segment { id: string; index: number; start: number; end: number; text: string; raw_text?: string; timestamps?: unknown[] }
export interface Result {
  text: string;
  raw_text?: string;
  segments: Segment[];
  version: number;
  versions?: { version: number; created_at: string; alignment_status: string }[];
  alignment_status: 'aligned' | 'pending' | 'failed' | 'unavailable' | 'incomplete';
  quality?: { issues?: Issue[]; profile_version?: string; validation_status?: string };
  changes?: unknown[];
}
export interface Job {
  id: string;
  filename: string;
  status: string;
  created_at: string;
  completed_at?: string;
  expires_at?: string;
  queue_position?: number;
  revision?: number;
  has_result?: boolean;
  alignment_status?: string;
  model?: string;
  progress?: { phase?: string; percent?: number; completed_chunks?: number; total_chunks?: number; current_chunk?: number; text?: string };
  result?: Result | null;
  error?: { code: string; message: string } | null;
}
export interface JobPage { jobs: Job[]; next_cursor: number | null }
export interface Upload { id: string; offset: number; size_bytes: number; chunk_size: number; job_id?: string | null }
export interface Term { canonical: string; aliases: string[]; domain: string; auto_normalize: boolean }
export class ApiError extends Error { constructor(message: string, public status: number, public code?: string) { super(message); } }
export const errorNames: Record<string, string> = {
  unauthorized: '登录已失效，请重新登录。', forbidden: '你没有执行此操作的权限。', username_exists: '这个用户名已存在。',
  login_rate_limited: '登录尝试过于频繁，请一分钟后重试。', upload_completed: '素材已提交，请在我的任务中管理或删除。',
  workbench_busy: '工作台正在处理较多请求，请稍后重试。',
  upload_not_found: '上传记录已过期或已删除，请重新选择文件。', job_not_found: '任务已过期或已删除。',
  audio_not_ready: '原音还在准备中，请稍后再试。', result_not_ready: '转录结果尚未生成。',
  alignment_pending: '字幕正在更新，请稍后再导出。', alignment_incomplete: '字幕尚未完成对齐，可先导出文字。',
  version_conflict: '内容已在其他页面更新，请刷新后再修改。', storage_quota_exceeded: '存储空间不足，请删除不再需要的任务后重试。',
  storage_unavailable: '服务器可用磁盘空间不足，请联系管理员。', audio_too_large: '文件超过服务允许的上传大小。',
  duration_limit_exceeded: '音频时长超过服务限制。', audio_decode_failed: '无法读取素材的音轨，请检查文件是否完整。',
  source_unavailable: '原音已不可用，无法重新处理此任务。', resume_version_unavailable: '原流程版本不可用，请重新提交素材。',
  upload_offset_mismatch: '上传位置已变化，请重试以从服务器记录的位置继续。', upload_corrupt: '上传文件不完整，请重新上传。',
  job_not_retryable: '此任务当前无法重试。', cross_origin_request: '请从工作台自己的地址打开页面后重试。',
  quality_incomplete: '部分音频未能完整转录，已有处理进度已保留，可回听检查后重试。',
  gpu_unavailable: '转录设备暂不可用，请联系管理员或稍后重试。', capability_not_supported: '服务尚未启用所需的转录或字幕对齐能力，请联系管理员。',
};

export async function api<T>(path: string, options: RequestInit = {}, login = false): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && typeof options.body === 'string') headers.set('Content-Type', 'application/json');
  const response = await fetch(`/api${path}`, { ...options, headers, credentials: 'same-origin' });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    if (response.status === 401 && !login) window.dispatchEvent(new Event('asr:unauthorized'));
    const code = data?.error?.code;
    const detail = typeof data?.detail === 'string' ? data.detail : Array.isArray(data?.detail) ? data.detail.map((item: { msg?: string }) => item.msg).join('；') : '';
    throw new ApiError(login && response.status === 401 ? '用户名或密码不正确。' : errorNames[code] || data?.error?.message || detail || `请求失败 (${response.status})`, response.status, code);
  }
  if (response.status === 204) return undefined as T;
  const body = await response.text();
  return body.trim() ? JSON.parse(body) as T : undefined as T;
}

export async function jobDetail(jobId: string, etag?: string): Promise<{ job?: Job; etag?: string }> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
    credentials: 'same-origin', headers: etag ? { 'If-None-Match': etag } : {},
  });
  if (response.status === 304) return { etag };
  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new Event('asr:unauthorized'));
    const data = await response.json().catch(() => null);
    throw new ApiError(errorNames[data?.error?.code] || data?.error?.message || `请求失败 (${response.status})`, response.status, data?.error?.code);
  }
  return { job: await response.json() as Job, etag: response.headers.get('ETag') || undefined };
}

export function message(error: unknown): string { return error instanceof Error ? error.message : '操作失败，请重试'; }
export const terminal = (status: string) => ['completed', 'failed', 'cancelled', 'expired'].includes(status);
export const phaseNames: Record<string, string> = {
  queued: '排队中', preprocessing: '准备音频', splitting: '分析语音边界', loading_model: '准备模型',
  transcribing: '转录中', reviewing: '复核片段', aligning: '对齐字幕', merging: '整理结果',
  completed: '已完成', failed: '处理失败', cancel_requested: '等待取消', cancelled: '已取消',
  expired: '已过期', uploading: '上传中', pending: '字幕更新中', alignment_pending: '字幕更新中',
};
export const phaseName = (value: string) => phaseNames[value] || value;
export function bytes(value: number) {
  if (!Number.isFinite(value)) return '—';
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}
export function clock(seconds: number) {
  const n = Math.max(0, Math.floor(seconds));
  return [Math.floor(n / 3600), Math.floor(n % 3600 / 60), n % 60].map(v => String(v).padStart(2, '0')).join(':');
}
export function date(value?: string) { return value ? new Date(value).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'; }
