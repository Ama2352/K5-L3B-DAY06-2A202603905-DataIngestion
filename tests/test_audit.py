import pytest


def test_unreviewed_rows_cannot_be_reported_as_zero_scores():
    from starter.src.audit import summarize_review
    with pytest.raises(ValueError):
        summarize_review({'coherence': [{'strategy': 'A', 'score': None}], 'grounding': [], 'metadata': []})


def test_filter_compliance_counts_questions_not_individual_hits():
    from starter.src.audit import filter_compliance
    rows = [{'question_id': 'q1', 'filter_compliant': True},
            {'question_id': 'q1', 'filter_compliant': False},
            {'question_id': 'q1', 'filter_compliant': True},
            {'question_id': 'q2', 'filter_compliant': True},
            {'question_id': 'q3', 'filter_compliant': None}]
    assert filter_compliance(rows) == (1, 2)


def test_no_filter_is_na_instead_of_perfect_score():
    from starter.src.audit import filter_compliance
    assert filter_compliance([{'question_id': 'q1', 'filter_compliant': None}]) == (0, 0)
