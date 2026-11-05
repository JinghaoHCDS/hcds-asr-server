import { useDeferredValue, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { useVirtualizer, type Virtualizer } from '@tanstack/react-virtual';
import { clock, speakerColor, type Speaker, type Issue, type Segment } from './api';
import { Icon } from './icons';

// Invalidate offscreen sizes, then restore mounted sizes: clearing the cache
// alone leaves unchanged DOM rows at their estimates until their next resize.
function useRemeasureOnWidth(root: RefObject<HTMLDivElement | null>, list: Virtualizer<HTMLDivElement, Element>) {
  useEffect(() => {
    if (!root.current) return;
    let width = 0;
    let frame = 0;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width === width) return;
      width = entry.contentRect.width;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        list.measure();
        list.getVirtualItems();
        root.current?.querySelectorAll<HTMLElement>('[data-index]').forEach(row => {
          list.resizeItem(Number(row.dataset.index), Math.round(row.getBoundingClientRect().height));
        });
      });
    });
    observer.observe(root.current);
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [root, list]);
}

export function VirtualText({ text }: { text: string }) {
  const root = useRef<HTMLDivElement>(null);
  const blocks = useMemo(() => text.match(/[\s\S]{1,2000}/gu) || ['此素材未识别出可转录的文字。'], [text]);
  const list = useVirtualizer({ count: blocks.length, getScrollElement: () => root.current, estimateSize: () => 500, overscan: 2 });
  useRemeasureOnWidth(root, list);
  return <div ref={root} className="transcript-scroll" role="region" tabIndex={0} aria-label="转录全文"><div style={{ height: list.getTotalSize(), position: 'relative' }}>
    {list.getVirtualItems().map(row => <div key={row.key} data-index={row.index} ref={list.measureElement} className="plain-transcript virtual-row" style={{ transform: `translateY(${row.start}px)` }}>{blocks[row.index]}</div>)}
  </div></div>;
}

export default function Transcript({ segments, raw, position, seek, onEdit, issues, jump, speakers = {} }: {
  segments: Segment[]; speakers?: Record<string, Speaker>; raw: boolean; position: number; seek: (seconds: number) => void;
  onEdit?: (segment: Segment) => void; issues: (Issue | string)[]; jump: { seconds: number; serial: number } | null;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState(''); const deferredQuery = useDeferredValue(query.trim().toLocaleLowerCase());
  const [follow, setFollow] = useState(false); const [match, setMatch] = useState(0);
  const [person, setPerson] = useState('all');
  const personId = (segment: Segment) => raw ? segment.model_speaker_id : segment.speaker_id;
  const [time, setTime] = useState(''); const [timeError, setTimeError] = useState('');
  const rows = useMemo(() => segments.map((segment, index) => ({ segment, index }))
    .filter(({ segment }) => (person === 'all' || (personId(segment) || 'unknown') === person) && (!deferredQuery || (raw ? segment.raw_text || segment.text : segment.text).toLocaleLowerCase().includes(deferredQuery))), [segments, raw, deferredQuery, person]);
  const list = useVirtualizer({ count: rows.length, getScrollElement: () => root.current, estimateSize: () => 112, overscan: 4,
    getItemKey: index => rows[index].segment.id || rows[index].index });
  useRemeasureOnWidth(root, list);
  const current = rows.findIndex(({ segment }) => position >= segment.start && position < segment.end);
  const pendingJump = useRef<number | null>(null);

  useEffect(() => { setMatch(0); root.current?.scrollTo({ top: 0 }); }, [deferredQuery, raw]);
  useEffect(() => {
    if (jump) {
      pendingJump.current = jump.seconds;
      // Keep the current search/person when replaying a visible result.
      if (!rows.some(({ segment }) => segment.start <= jump.seconds && segment.end > jump.seconds)) {
        setQuery(''); setPerson('all');
      }
      setFollow(false);
    }
  }, [jump]);
  useEffect(() => {
    if (pendingJump.current === null) return;
    const seconds = pendingJump.current;
    if ((person !== 'all' || deferredQuery) && !rows.some(({ segment }) => segment.start <= seconds && segment.end > seconds)) return;
    pendingJump.current = null;
    let index = rows.findIndex(({ segment }) => segment.end > seconds);
    if (index < 0) index = rows.length - 1;
    if (index >= 0) list.scrollToIndex(index, { align: 'center' });
  }, [jump, deferredQuery, person, rows, list]);
  useEffect(() => {
    if (follow && current >= 0 && !deferredQuery) list.scrollToIndex(current, { align: 'center' });
  }, [current, follow, deferredQuery, list]);

  function nextMatch(delta: number) {
    if (!rows.length) return;
    const next = (match + delta + rows.length) % rows.length; setMatch(next);
    list.scrollToIndex(next, { align: 'center' });
  }
  function jumpTime() {
    const parts = time.trim().split(':');
    const valid = /^\d+(?::[0-5]?\d){0,2}$/.test(time.trim());
    if (!valid) { setTimeError('请输入秒数、分:秒或时:分:秒'); return; }
    const seconds = parts.reduce((total, part) => total * 60 + Number(part), 0);
    if (seconds > segments.reduce((end, segment) => Math.max(end, segment.end), 0)) { setTimeError('时间超出转录范围'); return; }
    setTimeError(''); seek(seconds);
  }

  return <>
    <div className="transcript-navigation">
      <div className="transcript-filters">
        <div className="search-field"><Icon name="search" size={17} /><input type="search" autoComplete="off" aria-label="搜索全文" placeholder="搜索全文…" value={query} onChange={e => { setQuery(e.target.value); setFollow(false); }} onKeyDown={e => { if (e.key === 'Enter') nextMatch(e.shiftKey ? -1 : 1); }} />
          {query && <button type="button" className="icon-button" aria-label="清除搜索" onClick={() => setQuery('')}><Icon name="close" size={14} /></button>}
        </div>
        {Object.keys(speakers).length > 0 && <label className="speaker-filter"><span className="visually-hidden">说话人</span><select aria-label="按说话人筛选" value={person} onChange={e => { setPerson(e.target.value); setFollow(false); }}><option value="all">全部说话人</option>{Object.entries(speakers).map(([id, p]) => <option key={id} value={id}>{id} {p.name}{p.scope === 'chunk' ? '（分块内）' : ''}</option>)}<option value="unknown">待确认</option></select></label>}
      </div>
      <div className="transcript-playback">
        <button type="button" className={`button ghost small follow-button ${follow ? 'following' : ''}`} aria-pressed={follow} onClick={() => setFollow(!follow)}><Icon name="locate" size={15} />{follow ? '正在跟随' : '跟随播放'}</button>
        {deferredQuery && <div className="search-count"><span role="status">{rows.length ? `${Math.min(match + 1, rows.length)} / ${rows.length} 段` : '没有匹配'}</span><button type="button" className="icon-button" disabled={!rows.length} aria-label="上一匹配" onClick={() => nextMatch(-1)}><span className="previous-match"><Icon name="arrow" size={14} /></span></button><button type="button" className="icon-button" disabled={!rows.length} aria-label="下一匹配" onClick={() => nextMatch(1)}><span className="next-match"><Icon name="arrow" size={14} /></span></button></div>}
        <form className="time-jump" onSubmit={e => { e.preventDefault(); jumpTime(); }}><input aria-label="跳转时间" aria-invalid={!!timeError} aria-describedby={timeError ? 'transcript-time-error' : undefined} placeholder="00:00:00" value={time} onChange={e => setTime(e.target.value)} /><button type="submit" className="icon-button" aria-label="跳转到时间" title="跳转到时间"><Icon name="arrow" size={16} /></button></form>
      </div>
    </div>
    {timeError && <p className="inline-error time-error" id="transcript-time-error" role="alert">{timeError}</p>}
    <div ref={root} className="transcript-scroll" role="region" tabIndex={0} aria-label="转录片段" onWheel={() => setFollow(false)} onTouchMove={() => setFollow(false)} onKeyDown={e => { if (['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End'].includes(e.key)) setFollow(false); }}>
      {!rows.length && <div className="search-empty"><Icon name="search" size={25} /><p>没有匹配的片段</p><button type="button" className="text-link" onClick={() => { setQuery(''); setPerson('all'); }}>清除筛选，查看全文</button></div>}
      <div style={{ height: list.getTotalSize(), position: 'relative' }}>
        {list.getVirtualItems().map(row => {
          const { segment, index } = rows[row.index];
          const near = issues.some(issue => typeof issue !== 'string' && (issue.start !== undefined ? issue.start <= segment.end && (issue.end ?? issue.start) >= segment.start : issue.chunk_index === segment.index));
          const speakerId = personId(segment);
          const speaker = speakerId ? speakers[speakerId] : undefined;
          const uncertain = !speakerId || ((raw || segment.speaker_assignment !== 'user') && (speaker?.scope !== 'global' || ['new_candidate', 'unresolved'].includes(segment.speaker_resolution || '')));
          const text = raw ? segment.raw_text || segment.text : segment.text;
          const at = deferredQuery ? text.toLocaleLowerCase().indexOf(deferredQuery) : -1;
          return <div key={row.key} data-index={row.index} ref={list.measureElement} className="virtual-row" style={{ transform: `translateY(${row.start}px)` }}>
            <div className={`transcript-segment ${position >= segment.start && position < segment.end ? 'playing' : ''} ${near ? 'needs-review' : ''}`}>
              <button type="button" className="segment-time" onClick={() => seek(segment.start)} aria-label={`回听 ${clock(segment.start)} 的片段`}><span className="segment-number">{String(index + 1).padStart(2, '0')}</span>{clock(segment.start)}<Icon name="play" size={12} /></button>
              <div className="segment-body">{(Object.keys(speakers).length > 0 || segment.speaker_resolution) && <div className="speaker-label" style={{ color: speakerColor(speakerId) }}>{speakerId && <span className="speaker-dot" style={{ background: speakerColor(speakerId) }} />}{speaker ? `${speakerId} ${speaker.name}` : '说话人待确认'}{speaker && uncertain && <small> · 归属待确认{speaker.scope === 'chunk' ? '（分块内）' : ''}</small>}</div>}<button type="button" className="segment-text" onClick={() => seek(segment.start)}>{at < 0 ? text : <>{text.slice(0, at)}<mark>{text.slice(at, at + deferredQuery.length)}</mark>{text.slice(at + deferredQuery.length)}</>}{near && <span className="review-label" title="此处有待复核问题">待回听</span>}</button></div>
              {!raw && onEdit && <button type="button" className="icon-button edit-segment" aria-label={`修改第 ${index + 1} 段文字`} onClick={() => onEdit(segment)}><Icon name="edit" size={16} /></button>}
            </div>
          </div>;
        })}
      </div>
    </div>
    <div className="document-status"><span>{deferredQuery ? `${rows.length} 个匹配片段` : `${segments.length.toLocaleString()} 个片段`}</span><span>{follow ? '滚动正文可暂停跟随' : onEdit ? '点击文字回听，修改自动保留版本' : '点击文字回听原音'}</span></div>
  </>;
}
