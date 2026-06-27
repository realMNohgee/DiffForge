#!/usr/bin/env python3
"""
DiffForge — semantic diff for agent outputs across prompt versions.

When you update a prompt and want to know if the output got better or worse,
DiffForge compares two outputs and reports:
  • Lexical overlap (n-gram Jaccard)
  • Length change — did the agent get more/less verbose?
  • Sentiment shift — did the tone change?
  • Fact presence — which key entities/numbers appeared or disappeared?
  • Structural similarity — paragraph count, headings, bullet ratio

Pure Python standard library. Zero dependencies.

Domains: prompt engineering · agent evaluation · QA · A/B testing.
"""
import argparse, json, math, re, sys
from collections import Counter

STOPWORDS = set("the a an and or but if then of to in on at by for with is are was were be been being this that these those it its".split())


def ngrams(text, n=2):
    words = [w.lower() for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def entities(text):
    """Extract capitalized phrases and numbers as 'facts'."""
    caps = set(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text))
    nums = set(re.findall(r"\b\d[\d,.]*\b", text))
    return caps | nums


def sentiment_simple(text):
    pos = {"great", "excellent", "good", "best", "improved", "better", "positive", "success"}
    neg = {"bad", "worst", "poor", "failed", "error", "negative", "worse", "problem"}
    words = set(text.lower().split())
    return {"positive": len(words & pos), "negative": len(words & neg)}


def diff(a, b):
    wa = len(re.findall(r"\S+", a))
    wb = len(re.findall(r"\S+", b))
    na = ngrams(a); nb = ngrams(b)
    overlap = len(na & nb) / max(1, len(na | nb))

    sa = sentiment_simple(a); sb = sentiment_simple(b)
    sent_shift = (sb["positive"] - sb["negative"]) - (sa["positive"] - sa["negative"])

    ea = entities(a); eb = entities(b)
    new_facts = eb - ea
    lost_facts = ea - eb

    # Structure
    pa = len([l for l in a.split("\n") if l.strip() and not l.startswith(("#","-","*","1."))])
    pb = len([l for l in b.split("\n") if l.strip() and not l.startswith(("#","-","*","1."))])

    return {
        "length_a": wa, "length_b": wb, "length_delta": wb - wa,
        "length_change_pct": round((wb - wa) / max(1, wa) * 100, 1),
        "overlap": round(overlap, 3),
        "sentiment_shift": sent_shift,
        "new_facts": sorted(new_facts)[:10],
        "lost_facts": sorted(lost_facts)[:10],
        "paragraphs_a": pa, "paragraphs_b": pb,
    }


def cmd_diff(args):
    a = open(args.a, encoding="utf-8").read()
    b = open(args.b, encoding="utf-8").read()
    result = diff(a, b)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"  {'Metric':<24}{'Before':>10}{'After':>10}{'Delta':>10}")
        print(f"  {'─'*54}")
        print(f"  {'Words':<24}{result['length_a']:>10}{result['length_b']:>10}{result['length_delta']:>+10}")
        print(f"  {'Overlap (2-gram)':<24}{'':>10}{result['overlap']:>10.3f}")
        print(f"  {'Sentiment shift':<24}{'':>10}{result['sentiment_shift']:>+10}")
        print(f"  {'Paragraphs':<24}{result['paragraphs_a']:>10}{result['paragraphs_b']:>10}")
        if result["new_facts"]:
            print(f"\n  🟢 New facts: {', '.join(result['new_facts'][:5])}")
        if result["lost_facts"]:
            print(f"  🔴 Lost facts: {', '.join(result['lost_facts'][:5])}")
    return 0


def main():
    p = argparse.ArgumentParser(prog="diffforge", description=__doc__)
    p.add_argument("--a", required=True, help="first output file (before)")
    p.add_argument("--b", required=True, help="second output file (after)")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_diff)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
