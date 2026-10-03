#!/usr/bin/env python3
"""A/B prompt order: state-first (current) vs question-first (cache-friendly).

Same components, only block order differs. Measures per order, over N cases:
pattern/noul accuracy, argmax agreement, and summed cached/prompt tokens.

Usage:
  NETRA_KEY=... python3 eval/bench_order.py [ncases]
"""
import json
import math
import os
import sys
import urllib.request

API = "https://api.openai.com/v1/chat/completions"
KEY = os.environ["NETRA_KEY"]
MODEL = "gpt-6-sol"
SYS = ("You are a precise decision engine. Use only the information in the state. "
       "Reply with the answer label only, nothing else.")
LOGFLOOR = -30.0


def call(user):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": SYS}, {"role": "user", "content": user}],
        "temperature": 0, "max_completion_tokens": 2,
        "logprobs": True, "top_logprobs": 5, "reasoning": {"enabled": False}, "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    o = json.load(urllib.request.urlopen(req, timeout=90))
    ch = o["choices"][0]
    cont = (ch.get("logprobs") or {}).get("content") or []
    top = {t["token"]: t["logprob"] for t in cont[0].get("top_logprobs", [])} if cont else {}
    u = o.get("usage", {})
    return top, (u.get("prompt_tokens") or 0, ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))


def dist(top, labels):
    agg = {}
    for tok, lp in top.items():
        if tok.strip() in labels:
            agg[tok.strip()] = agg.get(tok.strip(), 0.0) + math.exp(lp)
    vals = {l: math.log(agg[l]) if l in agg else LOGFLOOR for l in labels}
    tot = sum(math.exp(v) for v in vals.values())
    return {l: math.exp(v) / tot for l, v in vals.items()}


def options_block(names):
    return "\n".join(f"{chr(65 + i)}. {k}" for i, k in enumerate(names))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cases = [json.loads(l) for l in open("eval/cases.v2.jsonl")][:n]
    stat = {"old": {"pok": 0, "sok": 0, "n": 0, "pt": 0, "ct": 0, "agree": 0},
            "new": {"pok": 0, "sok": 0, "n": 0, "pt": 0, "ct": 0, "agree": 0}}
    agree = 0
    for c in cases:
        q = c["questions"]
        names = sorted(q["pattern"]["criteria"])
        qp = (q["pattern"]["instructions"] + "\n\nCriteria:\n" +
              "".join(f"- {k}: {q['pattern']['criteria'][k]}\n" for k in names)).strip()
        sv = q["signals_support_alert"]
        qs = (sv["instructions"] + "\n\nYes: " + sv["criteria"]["true"] +
              "\nNo: " + sv["criteria"]["false"])
        exp_p, exp_s = c["expected"]["pattern"], c["expected"]["signals_support_alert"]
        outs = {}
        for tag, user_p, user_s in [
                ("old", f"State:\n{c['state']}\n\nQuestion: {qp}\n\nOptions:\n{options_block(names)}\n\nAnswer with only the letter of the correct option.",
                        f"State:\n{c['state']}\n\nQuestion: {qs}\n\nOptions:\nA. Yes\nB. No\n\nAnswer with only the letter of the correct option."),
                ("new", f"Question: {qp}\n\nOptions:\n{options_block(names)}\n\nState:\n{c['state']}\n\nAnswer with only the letter of the correct option.",
                        f"Question: {qs}\n\nOptions:\nA. Yes\nB. No\n\nState:\n{c['state']}\n\nAnswer with only the letter of the correct option.")]:
            top_p, (pt1, ct1) = call(user_p)
            top_s, (pt2, ct2) = call(user_s)
            pp = dist(top_p, [chr(65 + i) for i in range(len(names))])
            ps = dist(top_s, ["A", "B"])
            outs[tag] = (names[max(range(len(names)), key=lambda i: pp[chr(65 + i)])],
                         ps["A"] >= 0.5, pt1 + pt2, ct1 + ct2)
            s = stat[tag]
            s["n"] += 1
            s["pok"] += outs[tag][0] == exp_p
            s["sok"] += (outs[tag][1]) == (exp_s == "true")
            s["pt"] += outs[tag][2]
            s["ct"] += outs[tag][3]
        agree += outs["old"][:2] == outs["new"][:2]
    for tag in ["old", "new"]:
        s = stat[tag]
        print(f"{tag}: pattern {s['pok']}/{s['n']} signals {s['sok']}/{s['n']} "
              f"prompt={s['pt']} cached={s['ct']} ({s['ct']/max(s['pt'],1)*100:.0f}%)")
    print(f"agreement old-vs-new: {agree}/{n}")


if __name__ == "__main__":
    main()
