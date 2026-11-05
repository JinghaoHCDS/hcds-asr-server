import { useMemo, useState } from 'react';
import { message, type Result } from './api';

export default function ReplaceTools({ result, disabled, onReplace }: {
  result: Result; disabled: boolean; onReplace: (find: string, replacement: string) => Promise<void>;
}) {
  const [find, setFind] = useState('');
  const [replacement, setReplacement] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const counts = useMemo(() => {
    let occurrences = 0, blocks = 0;
    if (find.trim()) {
      for (const { text } of result.segments?.length ? result.segments : [result]) {
        let index = text.indexOf(find);
        if (index !== -1) blocks++;
        while (index !== -1) {
          occurrences++;
          index = text.indexOf(find, index + find.length);
        }
      }
    }
    return { occurrences, blocks };
  }, [find, result]);
  const blocked = disabled || saving || !counts.occurrences || find === replacement;

  async function replace() {
    if (blocked) return;
    setSaving(true); setError('');
    try {
      await onReplace(find, replacement);
      setFind('');
      setReplacement('');
    } catch (e) { setError(message(e)); }
    finally { setSaving(false); }
  }

  return <section className="panel replace-tools">
    <h3>查找与替换</h3>
    <form onSubmit={e => { e.preventDefault(); void replace(); }}>
      <label className="field-label">查找交付稿<input value={find} maxLength={20000} disabled={disabled || saving} onChange={e => { setFind(e.target.value); setError(''); }} /></label>
      <label className="field-label">替换为<input value={replacement} maxLength={20000} placeholder="留空则删除匹配文字" disabled={disabled || saving} onChange={e => { setReplacement(e.target.value); setError(''); }} /></label>
      <div className="replace-actions">
        <span className="muted" role="status">{find.trim() && (counts.occurrences
          ? `匹配 ${counts.occurrences} 处${result.segments?.length ? `，共 ${counts.blocks} 段` : ''}`
          : '没有匹配文字')}</span>
        <button type="submit" className="button secondary small" disabled={blocked}>{saving ? '替换中…' : '全部替换'}</button>
      </div>
    </form>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </section>;
}
