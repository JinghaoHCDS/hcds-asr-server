import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { clock, type Issue, type Segment } from './api';
import { Icon } from './icons';

export function VirtualText({ text }: { text: string }) {
  const root = useRef<HTMLDivElement>(null);
  const blocks = useMemo(() => text.match(/[\s\S]{1,2000}/gu) || ['此素材未识别出可转录的文字。'], [text]);
  const list = useVirtualizer({ count: blocks.length, getScrollElement: () => root.current, estimateSize: () => 500, overscan: 2 });
  return <div ref={root} className="transcript-scroll" tabIndex={0} aria-label="转录全文"><div style={{ height: list.getTotalSize(), position: 'relative' }}>
    {list.getVirtualItems().map(row => <div key={row.key} data-index={row.index} ref={list.measureElement} className="plain-transcript virtual-row" style={{ transform: `translateY(${row.start}px)` }}>{blocks[row.index]}</div>)}
  </div></div>;
}

export default function Transcript({ segments, raw, position, seek, onEdit, issues, jump }: {
  segments: Segment[]; raw: boolean; position: number; seek: (seconds: number) => void;
  onEdit?: (segment: Segment) => void; issues: (Issue | string)[]; jump: { seconds: number; serial: number } | null;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState(''); const deferredQuery = useDeferredValue(query.trim().toLocaleLowerCase());
  const [follow, setFollow] = useState(false); const [match, setMatch] = useState(0);
  const [time, setTime] = useState(''); const [timeError, setTimeError] = useState('');
  const rows = useMemo(() => segments.map((segment, index) => ({ segment, index }))
    .filter(({ segment }) => !deferredQuery || (raw ? segment.raw_text || segment.text : segment.text).toLocaleLowerCase().includes(deferredQuery)), [segments, raw, deferredQuery]);
  const list = useVirtualizer({ count: rows.length, getScrollElement: () => root.current, estimateSize: () => 112, overscan: 4,
    getItemKey: index => rows[index].segment.id || rows[index].index });
  const current = rows.findIndex(({ segment }) => position >= segment.start && position < segment.end);
  const pendingJump = useRef<number | null>(null);

  useEffect(() => { setMatch(0); root.current?.scrollTo({ top: 0 }); }, [deferredQuery, raw]);
  useEffect(() => {
    if (jump) { pendingJump.current = jump.seconds; setQuery(''); setFollow(false); }
  }, [jump]);
  useEffect(() => {
    if (pendingJump.current === null || deferredQuery) return;
    const seconds = pendingJump.current; pendingJump.current = null;
    let index = rows.findIndex(({ segment }) => segment.end > seconds);
    if (index < 0) index = rows.length - 1;
    if (index >= 0) list.scrollToIndex(index, { align: 'center' });
  }, [jump, deferredQuery, rows, list]);
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
      <div className="search-field"><Icon name="search" size={17} /><input aria-label="搜索全文" placeholder="搜索全文…" value={query} onChange={e => { setQuery(e.target.value); setFollow(false); }} onKeyDown={e => { if (e.key === 'Enter') nextMatch(e.shiftKey ? -1 : 1); }} />
        {query && <button className="icon-button" aria-label="清除搜索" onClick={() => setQuery('')}><Icon name="close" size={14} /></button>}
      </div>
      {deferredQuery ? <div className="search-count"><span role="status">{rows.length ? `${Math.min(match + 1, rows.length)} / ${rows.length} 段` : '没有匹配'}</span><button className="icon-button" disabled={!rows.length} aria-label="上一匹配" onClick={() => nextMatch(-1)}>↑</button><button className="icon-button" disabled={!rows.length} aria-label="下一匹配" onClick={() => nextMatch(1)}>↓</button></div>
        : <button className={`button ghost small follow-button ${follow ? 'following' : ''}`} aria-pressed={follow} onClick={() => setFollow(!follow)}><Icon name="locate" size={15} />{follow ? '正在跟随' : '跟随播放'}</button>}
      <form className="time-jump" onSubmit={e => { e.preventDefault(); jumpTime(); }}><input aria-label="跳转时间" placeholder="00:00:00" value={time} onChange={e => setTime(e.target.value)} /><button className="icon-button" aria-label="跳转到时间" title="跳转到时间"><Icon name="arrow" size={16} /></button></form>
    </div>
    {timeError && <p className="inline-error time-error" role="alert">{timeError}</p>}
    <div ref={root} className="transcript-scroll" tabIndex={0} aria-label="转录片段" onWheel={() => setFollow(false)} onTouchMove={() => setFollow(false)} onKeyDown={e => { if (['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End'].includes(e.key)) setFollow(false); }}>
      {!rows.length && <div className="search-empty"><Icon name="search" size={25} /><p>没有找到这段文字</p><button className="text-link" onClick={() => setQuery('')}>清除搜索，查看全文</button></div>}
      <div style={{ height: list.getTotalSize(), position: 'relative' }}>
        {list.getVirtualItems().map(row => {
          const { segment, index } = rows[row.index];
          const near = issues.some(issue => typeof issue !== 'string' && (issue.start !== undefined ? issue.start <= segment.end && (issue.end ?? issue.start) >= segment.start : issue.chunk_index === segment.index));
          const text = raw ? segment.raw_text || segment.text : segment.text;
          const at = deferredQuery ? text.toLocaleLowerCase().indexOf(deferredQuery) : -1;
          return <div key={row.key} data-index={row.index} ref={list.measureElement} className="virtual-row" style={{ transform: `translateY(${row.start}px)` }}>
            <div className={`transcript-segment ${current === row.index ? 'playing' : ''} ${near ? 'needs-review' : ''}`}>
              <button className="segment-time" onClick={() => seek(segment.start)} aria-label={`回听 ${clock(segment.start)} 的片段`}><span className="segment-number">{String(index + 1).padStart(2, '0')}</span>{clock(segment.start)}<Icon name="play" size={12} /></button>
              <button className="segment-text" onClick={() => seek(segment.start)}>{at < 0 ? text : <>{text.slice(0, at)}<mark>{text.slice(at, at + deferredQuery.length)}</mark>{text.slice(at + deferredQuery.length)}</>}{near && <span className="review-dot" title="此处有待复核问题" />}</button>
              {!raw && onEdit && <button className="icon-button edit-segment" aria-label={`修改第 ${index + 1} 段文字`} onClick={() => onEdit(segment)}><Icon name="edit" size={16} /></button>}
            </div>
          </div>;
        })}
      </div>
    </div>
    <div className="document-status"><span>{deferredQuery ? `${rows.length} 个匹配片段` : `${segments.length.toLocaleString()} 个片段`}</span><span>{follow ? '滚动正文可暂停跟随' : onEdit ? '点击文字回听 · 修改自动保留版本' : '点击文字回听原音'}</span></div>
  </>;
}
