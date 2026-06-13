"""Tests for decide_scheme against the real practice-set calibration."""

from recency_cv import decide_scheme


def _diag(rel, frac, n):
    return {
        "ok": True,
        "rel": rel,
        "frac_improved": frac,
        "n_features_scanned": n,
    }


def test_award_A_expanding_two_brakes_fire():
    # award_A is the closest would-be-sliding case; the primary frac gate AND the
    # feature-breadth floor BOTH fail, so two independent brakes hold expanding.
    out = decide_scheme(_diag(rel=0.623, frac=0.556, n=9))
    assert out["scheme"] == "expanding"
    assert out["gates"] == {"rel_ok": True, "frac_ok": False, "nfeat_ok": False}


def test_retail_sales_expanding():
    out = decide_scheme(_diag(rel=0.022, frac=0.200, n=5))
    assert out["scheme"] == "expanding"
    assert out["gates"] == {"rel_ok": False, "frac_ok": False, "nfeat_ok": False}


def test_energy_load_expanding():
    out = decide_scheme(_diag(rel=-0.514, frac=0.500, n=4))
    assert out["scheme"] == "expanding"
    assert out["gates"] == {"rel_ok": False, "frac_ok": False, "nfeat_ok": False}


def test_synthetic_sliding_case():
    out = decide_scheme(_diag(rel=0.40, frac=0.75, n=20))
    assert out["scheme"] == "sliding"
    assert out["gates"] == {"rel_ok": True, "frac_ok": True, "nfeat_ok": True}


def test_not_ok_diagnostic_falls_back_to_expanding():
    out = decide_scheme({"ok": False, "reason": "no time axis"})
    assert out["scheme"] == "expanding"
    assert out["gates"] == {"rel_ok": False, "frac_ok": False, "nfeat_ok": False}
    assert "no time axis" in out["reason"]
