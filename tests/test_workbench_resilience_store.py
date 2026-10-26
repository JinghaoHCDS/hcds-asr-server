from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from asr_server.errors import AsrError
from asr_server.workbench.store import SCHEMA_VERSION, WorkbenchStore, now


def task(store: WorkbenchStore, name: str = 'task') -> dict[str, Any]:
    job: dict[str, Any] = {'id': name, 'owner_id': 'alice', 'status': 'queued', 'created_at': now(),
                           'request': {'model': 'test'}, 'progress': {}, 'result': None}
    with store.connection() as db:
        db.execute('INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES(?,?,?,?,?)',
                   (name, 'alice', 'queued', 100, json.dumps(job)))
    return job


def test_schema4_upgrade_preserves_tasks_versions_chunks_accounts_and_expiry(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    user = store.create_user('owner', 'test-password')
    job = task(store)
    job.update({'owner_id': user['id'], 'status': 'completed', 'expires_at': '2030-01-01T00:00:00+00:00',
                'result': {'text': 'unchanged', 'version': 1, 'alignment_status': 'aligned'}})
    store.save_version(job)
    with store.connection() as db:
        db.execute("INSERT INTO job_chunks VALUES('task',7,?)", (json.dumps({'index': 7, 'text': 'sparse'}),))
        db.execute('PRAGMA user_version=4')
        before = {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table}')] for table in ('users', 'versions', 'job_chunks')}
        old_payload = db.execute("SELECT payload FROM jobs WHERE id='task'").fetchone()[0]
    store.close()
    reopened = WorkbenchStore(tmp_path)
    reopened.initialize()
    with reopened.connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION == 5
        assert db.execute("SELECT payload FROM jobs WHERE id='task'").fetchone()[0] == old_payload
        assert all([tuple(row) for row in db.execute(f'SELECT * FROM {table}')] == rows for table, rows in before.items())
    assert reopened.job('task')['expires_at'] == '2030-01-01T00:00:00+00:00'
    reopened.close()
    # Opening an already migrated store must not mutate the preserved records.
    second = WorkbenchStore(tmp_path)
    second.initialize()
    assert second.job('task')['result']['text'] == 'unchanged'
    second.close()


def test_attempt_fencing_rejects_superseded_cancelled_deleted_and_edited_commits(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    job = task(store)
    first = store.begin_attempt('task')
    stale = store.job('task')
    current = store.begin_attempt('task')
    with pytest.raises(AsrError, match='no longer current'):
        store.save_job(stale, attempt_id=first['attempt_id'], generation=first['generation'])
    store.save_chunk('task', {'index': 2, 'text': 'durable'}, 4,
                     attempt_id=current['attempt_id'], generation=current['generation'])
    latest = store.job('task')
    latest['cancel_requested'] = True
    store.save_job(latest)
    with pytest.raises(AsrError) as cancelled:
        store.save_chunk('task', {'index': 3, 'text': 'late'}, 4,
                         attempt_id=current['attempt_id'], generation=current['generation'])
    assert cancelled.value.code == 'attempt_superseded'
    latest['cancel_requested'] = False
    latest['result'] = {'version': 1, 'text': 'human edit'}
    store.save_version(latest)
    with pytest.raises(AsrError) as edited:
        store.save_job(stale, attempt_id=current['attempt_id'], generation=current['generation'])
    assert edited.value.code == 'version_conflict'
    assert store.job('task')['result']['text'] == 'human edit'
    with store.connection() as db:
        db.execute("DELETE FROM jobs WHERE id='task'")
    assert not store.attempt_is_current('task', current['attempt_id'], current['generation'])
    with pytest.raises(AsrError) as deleted:
        store.save_job(job, attempt_id=current['attempt_id'], generation=current['generation'])
    assert deleted.value.code == 'job_not_found'
    store.close()


def test_infrastructure_budget_survives_restart_and_stops_repeated_operation(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    task(store)
    first = store.begin_attempt('task')
    error = {'code': 'worker_timeout', 'message': 'deadline elapsed'}
    scheduled = store.schedule_infrastructure_retry('task', first['attempt_id'], first['generation'], error, 0)
    assert scheduled['status'] == 'queued' and scheduled['progress']['phase'] == 'retry_wait'
    assert scheduled['recovery']['infrastructure_retries'] == 1
    assert store.due_retry_jobs() == ['task']
    with pytest.raises(AsrError):
        store.schedule_infrastructure_retry('task', first['attempt_id'], first['generation'], error, 0)
    store.close()
    store = WorkbenchStore(tmp_path)
    assert store.job('task')['error'] == error
    store.claim_due_retry('task')
    retry = store.begin_attempt('task', operation_id=first['operation_id'])
    stopped = store.schedule_infrastructure_retry('task', retry['attempt_id'], retry['generation'], error, 0)
    assert stopped['status'] == 'failed' and stopped['next_action'] == 'retry'
    assert stopped['recovery']['infrastructure_retries'] == 1
    for expected in (2, 3):
        job = store.job_metadata('task')
        job['status'] = 'queued'
        store.save_job(job)
        manual = store.begin_attempt('task')
        scheduled = store.schedule_infrastructure_retry('task', manual['attempt_id'], manual['generation'], error, 0)
        assert scheduled['recovery']['infrastructure_retries'] == expected
        store.claim_due_retry('task')
    final = store.begin_attempt('task')
    exhausted = store.schedule_infrastructure_retry('task', final['attempt_id'], final['generation'], error, 0)
    assert exhausted['status'] == 'failed' and not exhausted['retryability']['retryable']
    assert len(exhausted['recovery']['history']) == 3 and store.due_retry_jobs() == []
    store.close()


def test_partial_summary_is_bounded_and_sparse_chunks_count_resolved_only(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    task(store)
    attempt = store.begin_attempt('task')
    fence = {'attempt_id': attempt['attempt_id'], 'generation': attempt['generation']}
    store.save_chunk('task', {'index': 6, 'text': 'later success'}, 8, **fence)
    store.save_chunk('task', {'index': 1, 'text': '', 'resolution': 'unresolved'}, 8, **fence)
    job = store.job('task')
    job['completeness'] = {'complete': False, 'resolved_chunks': 1, 'total_chunks': 8, 'gap_count': 7}
    job['gaps'] = [{'chunk_index': i, 'reason': 'private-evidence'} for i in range(1000)]
    job['retryability'] = {'retryable': False, 'reason': 'review_budget_exhausted'}
    store.save_job(job, **fence)
    summary = store.job_summaries('alice')['jobs'][0]
    assert summary['progress']['completed_chunks'] == 1
    assert summary['completeness']['gap_count'] == 7
    assert summary['retryability']['reason'] == 'review_budget_exhausted'
    assert 'gaps' not in summary and 'private-evidence' not in json.dumps(summary)
    assert 'later success' not in json.dumps(summary)
    assert [chunk['index'] for chunk in store.job('task')['progress']['chunks']] == [1, 6]
    store.close()


def test_migration_failure_rolls_back_schema_and_task_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = WorkbenchStore(tmp_path)
    task(store)
    with store.connection() as db:
        db.execute('PRAGMA user_version=4')
        original = db.execute("SELECT payload FROM jobs WHERE id='task'").fetchone()[0]
    store.close()
    broken = WorkbenchStore(tmp_path)
    migrate = broken._migrate
    def fail_after_migration(db: Any) -> None:
        migrate(db)
        db.execute("UPDATE jobs SET payload='{}' WHERE id='task'")
        raise RuntimeError('migration failed')
    monkeypatch.setattr(broken, '_migrate', fail_after_migration)
    with pytest.raises(RuntimeError, match='migration failed'):
        broken.initialize()
    broken.close()
    import sqlite3
    with sqlite3.connect(tmp_path / 'workbench.sqlite3') as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 4
        assert db.execute("SELECT payload FROM jobs WHERE id='task'").fetchone()[0] == original
    healthy = WorkbenchStore(tmp_path)
    healthy.initialize()
    assert healthy.job('task')['id'] == 'task'
    healthy.close()


def test_resilience_aggregate_uses_summary_without_transcript_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    from contextlib import contextmanager
    from typing import Iterator
    store = WorkbenchStore(tmp_path)
    job = task(store)
    job.update({'status': 'failed', 'error': {'code': 'quality_incomplete', 'message': 'private'},
                'completeness': {'complete': False, 'gap_count': 2},
                'recovery': {'infrastructure_retries': 1}})
    store.save_job(job)
    connection = store.connection
    @contextmanager
    def summaries_only() -> Iterator[sqlite3.Connection]:
        with connection() as db:
            db.set_authorizer(lambda operation, table, column, _database, _source:
                sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_READ and table == 'jobs' and column == 'payload' else sqlite3.SQLITE_OK)
            yield db
    monkeypatch.setattr(store, 'connection', summaries_only)
    metrics = store.resilience_summary()
    assert metrics['tasks'] == metrics['failed_tasks'] == 1
    assert metrics['quality_gap_count'] == 2 and metrics['infrastructure_retry_debits'] == 1
    assert metrics['failure_counts'] == [{'code': 'quality_incomplete', 'tasks': 1}]
    assert 'private' not in json.dumps(metrics)
    store.close()


def test_interrupted_attempt_is_debited_before_restart_can_repeat_it(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    task(store)
    first = store.begin_attempt('task')
    store.close()
    store = WorkbenchStore(tmp_path)
    resumed = store.begin_attempt('task', first['operation_id'])
    assert resumed['infrastructure_retries'] == resumed['operation_retries'] == 1
    assert resumed['history'][0]['error_code'] == 'process_interrupted'
    store.close()
    store = WorkbenchStore(tmp_path)
    with pytest.raises(AsrError) as exhausted:
        store.begin_attempt('task', first['operation_id'])
    assert exhausted.value.code == 'infrastructure_retry_exhausted'
    assert store.job_metadata('task')['recovery']['generation'] == resumed['generation']
    store.close()


def test_due_retry_joins_after_durable_work_submitted_during_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import store as module
    clock = ['2026-10-26T00:00:00+00:00']
    monkeypatch.setattr(module, 'now', lambda: clock[0])
    store = WorkbenchStore(tmp_path)
    task(store, 'retry')
    attempt = store.begin_attempt('retry')
    waiting = store.schedule_infrastructure_retry('retry', attempt['attempt_id'], attempt['generation'],
                                                   {'code': 'worker_timeout', 'message': 'timeout'}, 5)
    old_order = waiting['recovery']['retry_order']
    clock[0] = '2026-10-26T00:00:02+00:00'
    existing = task(store, 'already-persisted')  # Memory queue may be full; durable order still counts.
    existing['created_at'] = clock[0]
    store.save_job(existing)
    clock[0] = '2026-10-26T00:00:06+00:00'
    claimed = store.claim_due_retry('retry', at='9999-01-01T00:00:00+00:00')
    assert claimed['queued_at'] == clock[0]  # The comparison override must never become a stored submission date.
    assert claimed['recovery']['retry_order'] > old_order
    assert store.pending_jobs() == ['already-persisted', 'retry']
    assert store.due_retry_jobs('9999-01-01T00:00:00+00:00') == []
    store.close()
    reopened = WorkbenchStore(tmp_path)
    assert reopened.pending_jobs() == ['already-persisted', 'retry']
    reopened.close()


async def test_storage_failure_callback_preserves_errors_and_ignores_business_conflicts(tmp_path: Path) -> None:
    import errno
    import sqlite3
    store = WorkbenchStore(tmp_path)
    failures: list[BaseException] = []
    store.storage_error_callback = failures.append
    store.create_user('owner', 'private-password')
    with pytest.raises(AsrError) as duplicate:
        store.create_user('owner', 'private-password')
    assert duplicate.value.code == 'username_exists' and failures == []
    task(store)
    def write_to_readonly_database() -> None:
        with store.connection() as db:
            db.execute('PRAGMA query_only=ON')
            db.execute("UPDATE jobs SET status='failed'")
    with pytest.raises(sqlite3.OperationalError) as readonly:
        await store.run(write_to_readonly_database)
    assert failures == [readonly.value]  # The same connection/run failure is reported only once.
    unavailable = OSError(errno.ENOSPC, 'no space')
    def fail_outside_connection() -> None:
        raise unavailable
    with pytest.raises(OSError) as disk:
        await store.run(fail_outside_connection)
    assert disk.value is unavailable and failures == [readonly.value, unavailable]
    store.close()


def test_initialization_storage_failure_notifies_and_propagates(tmp_path: Path) -> None:
    blocked = tmp_path / 'data-root'
    blocked.write_bytes(b'not-a-directory')
    store = WorkbenchStore(blocked)
    failures: list[BaseException] = []
    store.storage_error_callback = failures.append
    with pytest.raises(OSError) as unavailable:
        store.initialize()
    assert failures == [unavailable.value]
    store.close()


@pytest.mark.parametrize('status', ['completed', 'failed', 'cancelled'])
def test_terminal_commit_finishes_attempt_without_later_etag_revision(tmp_path: Path, status: str) -> None:
    store = WorkbenchStore(tmp_path)
    task(store)
    attempt = store.begin_attempt('task')
    job = store.job_metadata('task')
    job['status'] = status
    if status == 'completed':
        job['result'] = {'version': 1, 'text': 'published transcript', 'alignment_status': 'aligned'}
    if status == 'cancelled':
        job['cancel_requested'] = True
        store.save_job(job)  # Cancellation is an independent user action, not a worker commit.
    else:
        store.save_version(job, attempt_id=attempt['attempt_id'], generation=attempt['generation'])
    terminal = store.job_metadata('task')
    assert terminal['recovery']['active'] is False and terminal['recovery']['finished_at']
    revision = terminal['revision']
    store.finish_attempt('task', attempt['attempt_id'], attempt['generation'])
    assert store.job_revision('task', 'alice') == revision
    store.close()


def test_pending_edit_alignment_keeps_attempt_active_until_alignment_result_commits(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    job = task(store)
    job.update({'status': 'completed', 'result': {'version': 2, 'text': 'edited transcript', 'alignment_status': 'pending'}})
    store.save_version(job)
    attempt = store.begin_attempt('task', kind='alignment')
    fence = {'attempt_id': attempt['attempt_id'], 'generation': attempt['generation']}
    progress = store.job_metadata('task')
    progress['retained_file_bytes'] = {'operation/audio.wav': 44}
    store.save_job(progress, **fence)
    assert store.attempt_is_current('task', attempt['attempt_id'], attempt['generation'])
    aligned = store.job('task')
    aligned['result']['alignment_status'] = 'aligned'
    store.save_version(aligned, **fence)
    terminal = store.job_metadata('task')
    assert terminal['recovery']['active'] is False
    revision = terminal['revision']
    store.finish_attempt('task', attempt['attempt_id'], attempt['generation'])
    assert store.job_revision('task', 'alice') == revision
    store.close()
