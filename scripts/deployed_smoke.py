#!/usr/bin/env python3
"""Read-only deployment smoke: expected Git source hashes, health and durable backlog."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Runs inside the selected container. Reads source bytes and SQLite via mode=ro;
# only GETs existing health routes. Never loads application configuration/secrets.
RUNTIME_PROBE = r'''
import hashlib,json,pathlib,sqlite3,sys,urllib.request,urllib.error
files=json.loads(sys.stdin.read())
result={'hashes':{},'missing':[],'probes':{},'queues':{}}
for name in files:
    path=pathlib.Path('/app')/name
    if path.is_file():
        result['hashes'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        result['missing'].append(name)
for route in ('/health','/health/providers'):
    try:
        response=urllib.request.urlopen('http://127.0.0.1:7788'+route,timeout=15)
    except urllib.error.HTTPError as error:
        response=error
    with response:
        body=json.loads(response.read())
        # Preserve status/error presence only: never dump arbitrary health details.
        safe={'status':body.get('status')}
        safe.update({key:True for key in ('health_check_error','provider_health_check_error') if key in body})
        result['probes'][route]={'code':response.code,'body':safe}
for filename,table in (('index-requests.sqlite3','index_requests'),('hook-outbox.sqlite3','hook_deliveries')):
    path=pathlib.Path('/data/index')/filename
    if not path.exists():
        result['queues'][table]=None
        if table=='hook_deliveries':
            result['queues']['hook_redrive_required']=None
        continue
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)
    try:
        result['queues'][table]=connection.execute("SELECT count(*) FROM "+table+" WHERE status='pending'").fetchone()[0]
        if table=='hook_deliveries':
            result['queues']['hook_redrive_required']=connection.execute("SELECT count(*) FROM hook_deliveries WHERE status='redrive_required'").fetchone()[0]
    finally:
        connection.close()
print(json.dumps(result))
'''


def build_manifest(repo: Path, revision: str) -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--verify', f'{revision}^{{commit}}'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', '--name-only', commit]).decode().split('\0')
    manifest = {}
    for name in names:
        if (name.endswith('.py') and not name.startswith('tests/')) or name == 'requirements.txt':
            data = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{name}'])
            manifest[name] = hashlib.sha256(data).hexdigest()
    if not manifest:
        raise ValueError('revision has no runtime files')
    return commit, manifest


def collect_runtime(container: str, manifest: dict[str, str]) -> dict:
    # Filter inside Docker's template. Full inspect contains environment secrets.
    template = ('{"id":{{json .Id}},"image":{{json .Image}},"running":{{json .State.Running}},'
                '"oom_killed":{{json .State.OOMKilled}},"restarts":{{json .RestartCount}},'
                '"started_at":{{json .State.StartedAt}},'
                '"health":{{if .State.Health}}{{json .State.Health.Status}}{{else}}"unknown"{{end}}}')
    identity = json.loads(subprocess.check_output(['docker', 'inspect', '--format', template, container], text=True, timeout=20))
    completed = subprocess.run(['docker', 'exec', '-i', identity['id'], 'python', '-c', RUNTIME_PROBE],
                               input=json.dumps(list(manifest)), text=True, capture_output=True, timeout=90, check=True)
    result = {**identity, **json.loads(completed.stdout)}
    current_id = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}', container], text=True, timeout=20).strip()
    if current_id != identity['id']:
        raise RuntimeError('container changed during smoke')
    return result


def assess_deployment(manifest: dict[str, str], runtime: dict, *, max_pending=0) -> dict:
    if not manifest or max_pending < 0:
        raise ValueError('nonempty source manifest and nonnegative backlog budget required')
    mismatched = sorted(name for name, digest in manifest.items() if runtime.get('hashes', {}).get(name) != digest)
    failures = []
    if mismatched:
        failures.append('revision_mismatch')
    if not runtime.get('running') or runtime.get('health') != 'healthy' or runtime.get('oom_killed'):
        failures.append('container_unhealthy')
    for route in ('/health', '/health/providers'):
        probe = runtime.get('probes', {}).get(route, {})
        body = probe.get('body', {})
        if (probe.get('code') != 200 or body.get('status') != 'ok'
                or 'health_check_error' in body or 'provider_health_check_error' in body):
            failures.append(f'probe_failed:{route}')
    for name in ('index_requests', 'hook_deliveries'):
        count = runtime.get('queues', {}).get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > max_pending:
            failures.append(f'backlog_failed:{name}')
    if runtime.get('queues', {}).get('hook_redrive_required') != 0:
        failures.append('callback_redrive_required')
    return {'passed': not failures, 'failures': failures, 'mismatched': mismatched,
            'files_checked': len(manifest), 'runtime': runtime,
            'revision_evidence': 'container-filesystem-sha256', 'max_pending': max_pending}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--max-pending', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        revision, manifest = build_manifest(ROOT, args.expected_revision)
        report = assess_deployment(manifest, collect_runtime(args.container, manifest), max_pending=args.max_pending)
        report['expected_revision'] = revision
    except Exception as error:
        # subprocess errors may contain remote stdout/stderr; record only class.
        report = {'passed': False, 'error': type(error).__name__}
    report['checked_at'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'runtime'}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
