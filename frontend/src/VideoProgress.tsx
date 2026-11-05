import { phaseName, videoProgressText, type Job } from './api';

export default function VideoProgress({ job }: { job: Job }) {
  const progress = job.video_asset?.progress || { phase: job.video_asset?.state === 'queued' ? 'queued_video' : 'preparing_video', percent: undefined };
  const percent = progress.percent == null ? undefined : Math.max(0, Math.min(100, progress.percent));
  return <div className="video-progress" role="status">
    <div className="video-progress-label"><strong>{phaseName(progress.phase || 'preparing_video')}</strong><span>{videoProgressText(job)}</span></div>
    <div className={`progress-track${percent === undefined ? ' indeterminate' : ''}`} role="progressbar" aria-label={phaseName(progress.phase || 'preparing_video')} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
      <span style={percent !== undefined ? { transform: `scaleX(${percent / 100})` } : undefined} />
    </div>
  </div>;
}
