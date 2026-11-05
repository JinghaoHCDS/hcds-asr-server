import { useState } from 'react';
import { api, message, speakerColor, type Job, type Speaker } from './api';

export default function SpeakerTools({ job, onSaved, disabled }: { job: Job; onSaved: (job: Job) => void; disabled: boolean }) {
  const [creating, setCreating] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function save(name: string, id?: string) {
    setSaving(true); setError('');
    try {
      const next = await api<Job>(`/jobs/${job.id}/speakers${id ? '/' + encodeURIComponent(id) : ''}`, { method: id ? 'PATCH' : 'POST', body: JSON.stringify({ name: name.trim(), base_version: job.result?.version }) });
      onSaved(next); if (!id) setCreating('');
    } catch (e) { setError(message(e)); } finally { setSaving(false); }
  }
  return <section className="panel speaker-tools"><h3>说话人</h3>{Object.entries(job.result?.speakers || {}).map(([id, person]) => <Person key={`${id}:${person.name}`} id={id} person={person} disabled={disabled || saving} save={save} />)}
    <form className="speaker-name-row" onSubmit={e => { e.preventDefault(); void save(creating); }}><input aria-label="新说话人名称" maxLength={120} value={creating} placeholder="新说话人名称" onChange={e => setCreating(e.target.value)} disabled={disabled || saving} /><button className="button secondary small" disabled={disabled || saving || !creating.trim()}>添加</button></form>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </section>;
}
function Person({ id, person, disabled, save }: { id: string; person: Speaker; disabled: boolean; save: (name: string, id: string) => Promise<void> }) {
  const [name, setName] = useState(person.name);
  return <form className="speaker-name-row" onSubmit={e => { e.preventDefault(); void save(name, id); }}><span style={{ color: speakerColor(id) }}>{id}</span><input aria-label={`${id} 的名称，修改应用于全部相关片段`} maxLength={120} value={name} onChange={e => setName(e.target.value)} disabled={disabled} /><button className="button ghost small" disabled={disabled || !name.trim() || name.trim() === person.name}>改名</button>{person.scope === 'chunk' && <small>分块内编号：{person.source_speaker}</small>}</form>;
}
