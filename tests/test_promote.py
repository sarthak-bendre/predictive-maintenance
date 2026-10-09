from src.promote import decide, passes_gate

GOOD = {"recall": 0.85, "pr_auc": 0.90, "cost": 111.0}


def test_gate_passes_good_model():
    assert passes_gate(GOOD)[0]


def test_gate_rejects_low_recall_and_pr_auc():
    ok, reason = passes_gate({**GOOD, "recall": 0.5, "pr_auc": 0.6})
    assert not ok and "recall" in reason and "PR-AUC" in reason


def test_first_model_is_promoted():
    assert decide(GOOD, None)[0]


def test_lower_cost_wins():
    assert decide(GOOD, {**GOOD, "cost": 130.0})[0]
    assert not decide({**GOOD, "cost": 130.0}, GOOD)[0]


def test_pr_auc_breaks_cost_tie_and_equal_keeps_champion():
    assert decide({**GOOD, "pr_auc": 0.95}, GOOD)[0]
    assert not decide(GOOD, GOOD)[0]
