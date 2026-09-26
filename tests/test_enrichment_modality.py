"""Guard speech-act instructions and exercise semantics with the configured LLM.

Unit tests verify the provider-bound contract, not a simulated model's language
understanding. Live cases catch actual question-to-policy transformations.
"""

import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from doc_enrichment import enrich_document

CASES = json.loads(
    (Path(__file__).parent / 'fixtures/enrichment_modality.json').read_text()
)


@pytest.mark.parametrize('context', [False, True], ids=['atomic', 'with_context'])
def test_modality_contract_reaches_generator(context):
    generator = Mock()
    generator.generate.return_value = json.dumps(
        {'summary': 'The sender asks whether the cutoff applies in future.',
         'doc_type': ['question'], 'key_facts': []}
    )
    case = CASES[0]
    enrich_document(
        text=case['text'], title='Tenant message', source_type='pg_message',
        context_text=case['context_text'] if context else '', generator=generator,
    )
    prompt = generator.generate.call_args.args[0]
    # All three fields stored in the reported corrupt rows need the same rule.
    assert 'summary, doc_type, and key_facts' in prompt
    assert 'Questions remain questions, not proposals or confirmations' in prompt
    assert 'A question about whether a rule applies does not request its adoption' in prompt
    assert 'Proposals remain proposals, not adopted rules' in prompt
    assert 'temporary instructions must not become standing policy' in prompt
    assert 'Explicitly adopted standing rules remain statements of policy' in prompt
    assert 'Nearby context must not change the primary item' in prompt
    assert case['text'] in prompt


@pytest.mark.live
@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_live_enrichment_preserves_modality(case):
    from core.config import load_config
    from providers.llm import build_llm_provider

    config = load_config()
    generator = build_llm_provider(config)
    assert generator is not None, 'Configured enrichment provider unavailable'
    enrichment = config['enrichment']
    result = enrich_document(
        text=case['text'], context_text=case['context_text'],
        title='Tenant message', source_type='pg_message', generator=generator,
        max_input_chars=enrichment.get('max_input_chars', 20000),
        max_output_tokens=enrichment.get('max_output_tokens', 5000),
        record_taxonomy_usage=False,
    )
    assert '_enrichment_failed' not in result, result
    summary = result['enr_summary'].lower()
    facts = json.loads(result['enr_key_facts'])
    assert facts, result
    if case['id'] == 'question':
        assert re.search(r'\b(ask\w*|question\w*|inquir\w*|whether)\b', summary), result
        assert 'proposal' not in result['enr_doc_type'].lower(), result
        # A fact about this message must retain the question, not assert its answer.
        assert any(re.search(r'\b(ask\w*|question\w*|inquir\w*|whether)\b', fact.lower())
                   for fact in facts), result
        # Reject the observed intent changes, while allowing context to mention
        # a different person's confirmation (e.g. the fire marshal's safety check).
        assert not re.search(
            r'\b(?:sender|tenant)\s+(?:is\s+)?(?:propos\w*|confirm\w*)\b'
            r'|\b(?:new|standing|permanent) (?:rule|policy) '
            r'(?:is|has been) (?:propos\w*|establish\w*|adopt\w*)\b',
            ' '.join([summary, *facts]).lower(),
        ), result
    elif case['id'] == 'proposal':
        assert re.search(r'\b(propos\w*|suggest\w*)\b', summary), result
        assert re.search(r'\b(propos\w*|suggest\w*)\b', ' '.join(facts).lower()), result
        assert not re.search(r'\b(adopted|established|enacted)\b', summary), result
    else:
        assert re.search(r'\b(adopt\w*|establish\w*|implement\w*|rule|policy)\b', summary), result
        assert re.search(r'\b(permanent|standing|every night|nightly)\b',
                         ' '.join([summary, *facts]).lower()), result
        assert not re.search(r'\b(propos\w*|asks?|whether)\b', summary), result
