import pytest

from cds_exact_events import event_request, fetch_event_page


@pytest.mark.parametrize('ref', ['calendar:', 'calendar:   '])
def test_empty_calendar_uid_never_queries_unassigned_records(ref):
    with pytest.raises(ValueError, match='nonempty UID'):
        fetch_event_page(None, [ref], event_request([ref]))


@pytest.mark.parametrize('rows,field', [([], 'missing_refs'), ([('one',), ('two',)], 'ambiguous_refs')])
def test_calendar_gaps_and_multiple_resource_groups_remain_explicit(rows, field):
    class Cursor:
        def execute(self, sql, params):
            pass

        def fetchall(self):
            return rows
    refs = ['calendar:meeting@example.test']
    page = fetch_event_page(Cursor(), refs, event_request(refs))
    assert page['status'] == 'degraded'
    assert page[field] == refs
    assert page['messages'] == []
