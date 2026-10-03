#!/usr/bin/env python3
"""Derive review_priority in code from state text + the model's pattern answer.

Typesafe's own guidance ("atomic questions, composed in code"): priority here
is a deterministic function of thresholds the model already struggles to apply
in one step (eval v2: acc 0.335, flat distributions). Ask the model for
`pattern`, compute the priority from the numbers.

Usage:
  python3 eval/compose.py --selfcheck eval/cases.v2.jsonl
  echo '{"state": "...", "pattern": "merchant_risk"}' | python3 eval/compose.py

Selfcheck must print 200/200: the parser recovers the generator's features
faithfully from text (same predicates as eval/gen_cases.py).
"""
import json
import re
import sys

NA = "tidak berlaku"


def num(s):
    return int(s.replace(".", ""))


def rp(s):
    return num(s.replace("Rp ", ""))


def pct(s):
    return float(s.replace("%", "").replace(",", "."))


def opt_int(s):
    return None if s.strip() == NA or s.strip().startswith(NA) else num(s.split()[0])


def parse_state(text):
    g = lambda pat, fn=str: fn(re.search(pat, text, re.M).group(1).strip())
    return {
        "amount": g(r"- Nominal transaksi: Rp ([\d.]+)", rp),
        "limit": g(r"- Limit maksimal transaksi: Rp ([\d.]+)", rp),
        "method": g(r"- Metode pembayaran: (\S+)"),
        "card_accts": g(r"- Jumlah akun yang memakai nomor kartu ini: (.+)", opt_int),
        "card_tx_1h": g(r"- Jumlah transaksi dengan nomor kartu ini dalam 1 jam terakhir: (.+)", opt_int),
        "card_merchants_24h": g(r"- Jumlah merchant berbeda yang dipakai nomor kartu ini dalam 24 jam terakhir: (.+)", opt_int),
        "m_age": g(r"- Umur merchant: (\d+) hari", int),
        "m_cat": g(r"- Kategori risiko merchant: (\S+)"),
        "chargeback": g(r"- Chargeback rate merchant 30 hari: ([\d,]+)%", pct),
        "m_cards_1h": g(r"- Jumlah nomor kartu berbeda yang bertransaksi di merchant ini dalam 1 jam terakhir: (\d+)", int),
    }


def is_card(f):
    return f["method"] == "credit_card"


def holds(f, pat):
    if pat == "transaction_above_limit":
        return f["amount"] > f["limit"]
    if pat == "merchant_random_credit_cards":
        return is_card(f) and f["m_cards_1h"] >= 10
    if pat == "transaction_identical_credit_cards":
        return is_card(f) and (f["card_tx_1h"] >= 5 or f["card_accts"] >= 5)
    if pat == "merchant_identical_credit_cards":
        return is_card(f) and f["card_merchants_24h"] >= 4
    if pat == "merchant_risk":
        return f["chargeback"] >= 2.0 or (f["m_age"] < 30 and f["m_cat"] == "tinggi")
    return False


def strong(f, pat):
    if pat == "transaction_above_limit":
        return f["amount"] >= 2 * f["limit"]
    if pat == "merchant_random_credit_cards":
        return f["m_cards_1h"] >= 20
    if pat == "transaction_identical_credit_cards":
        return f["card_tx_1h"] >= 10 or f["card_accts"] >= 10
    if pat == "merchant_identical_credit_cards":
        return f["card_merchants_24h"] >= 8
    if pat == "merchant_risk":
        return f["chargeback"] >= 4.0
    return False


def priority(f, pattern):
    """1 Rutin (no base threshold holds), 2 Tinggi (holds, not urgent),
    3 Segera (strong signal + amount >= 20% of limit)."""
    if not pattern or pattern == "insufficient_evidence":
        return 1
    if not holds(f, pattern):
        return 1
    if strong(f, pattern) and f["amount"] >= 0.2 * f["limit"]:
        return 3
    return 2

PATTERNS = ["transaction_above_limit", "merchant_random_credit_cards",
            "transaction_identical_credit_cards", "merchant_identical_credit_cards",
            "merchant_risk"]


def true_patterns(f):
    return [p for p in PATTERNS if holds(f, p)]


def decide(f, pattern):
    """Priority plus a conflict flag.

    Conflict means the model's pattern contradicts the deterministic
    thresholds (missed OR-branch, hallucinated pattern): route to an
    analyst regardless of confidence. Priority then comes from the
    thresholds, not the model.
    """
    true = true_patterns(f)
    if not pattern or pattern == "insufficient_evidence":
        if true:
            return {"priority": max(priority(f, p) for p in true),
                    "conflict": True,
                    "detail": f"model said insufficient, thresholds hold for {true}"}
        return {"priority": 1, "conflict": False, "detail": ""}
    if pattern not in true:
        base = max([priority(f, p) for p in true], default=1)
        return {"priority": base, "conflict": True,
                "detail": f"model said {pattern}, thresholds hold for {true or 'none'}"}
    return {"priority": priority(f, pattern), "conflict": False, "detail": ""}


def _test_conflicts():
    # The live miss: card_accts=8 meets the OR-branch, model said insufficient.
    f = {"amount": 15265000, "limit": 20000000, "method": "credit_card",
         "card_accts": 8, "card_tx_1h": 2, "card_merchants_24h": 1,
         "m_age": 2950, "m_cat": "sedang", "chargeback": 1.9, "m_cards_1h": 6}
    d = decide(f, "insufficient_evidence")
    assert d["conflict"] and d["priority"] == 2, d
    d = decide(f, "transaction_identical_credit_cards")
    assert not d["conflict"] and d["priority"] == 2, d
    d = decide(f, "merchant_risk")
    assert d["conflict"] and d["priority"] == 2, d


def selfcheck(path):
    n = bad = 0
    for line in open(path):
        c = json.loads(line)
        n += 1
        try:
            f = parse_state(c["state"])
            d = decide(f, c["expected"]["pattern"])
        except Exception as e:  # noqa: BLE001
            print(f"parse error case {n}: {e}")
            bad += 1
            continue
        want = int(c["expected"]["review_priority"])
        # Labels are generator-consistent: priority must match and the
        # model's (true) pattern must never conflict with the thresholds.
        if d["priority"] != want or d["conflict"]:
            print(f"mismatch case {n}: {d} want priority {want}")
            bad += 1
    _test_conflicts()
    print(f"selfcheck: {n - bad}/{n} priorities match labels, conflict tests pass")
    return bad == 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--selfcheck":
        sys.exit(0 if selfcheck(sys.argv[2]) else 1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        print(json.dumps(decide(parse_state(c["state"]), c.get("pattern"))))


if __name__ == "__main__":
    main()
