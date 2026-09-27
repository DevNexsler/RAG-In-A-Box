"""Deployment gate rejects stale code, masked health failures, and backlog."""
import pytest

from scripts.deployed_smoke import assess_deployment


@pytest.fixture
def snapshot():
    return {'running': True, 'health': 'healthy', 'oom_killed': False,
            'hashes': {'server.py': 'expected'}, 'missing': [],
            'probes': {'/health': {'code': 200, 'body': {'status': 'ok'}},
                       '/health/providers': {'code': 200, 'body': {'status': 'ok'}}},
            'queues': {'index_requests': 0, 'hook_deliveries': 0, 'hook_redrive_required': 0}}


def test_healthy_old_revision_fails(snapshot):
    snapshot['hashes']['server.py'] = 'old'
    report = assess_deployment({'server.py': 'expected'}, snapshot)
    assert not report['passed']
    assert report['mismatched'] == ['server.py']


@pytest.mark.parametrize('failure', ['missing', 'provider', 'masked-health', 'backlog', 'oom'])
def test_green_docker_health_cannot_hide_failure(snapshot, failure):
    if failure == 'missing':
        snapshot['hashes'].clear()
        snapshot['missing'] = ['server.py']
    elif failure == 'provider':
        snapshot['probes']['/health/providers']['code'] = 503
    elif failure == 'masked-health':
        snapshot['probes']['/health']['body']['health_check_error'] = 'unavailable'
    elif failure == 'backlog':
        snapshot['queues']['hook_deliveries'] = 1
    else:
        snapshot['oom_killed'] = True
    assert not assess_deployment({'server.py': 'expected'}, snapshot)['passed']


def test_matching_revision_and_healthy_runtime_pass(snapshot):
    assert assess_deployment({'server.py': 'expected'}, snapshot)['passed']


def test_empty_manifest_cannot_certify_deployment(snapshot):
    with pytest.raises(ValueError):
        assess_deployment({}, snapshot)


def test_terminal_callback_backlog_fails_even_when_pending_empty(snapshot):
    snapshot['queues']['hook_redrive_required'] = 2
    assert not assess_deployment({'server.py': 'expected'}, snapshot)['passed']
