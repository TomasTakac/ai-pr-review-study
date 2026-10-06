"""Quality checks of the classification rules in preprocessing.py (results in results/qa/).

  signature_recall.csv        share of AIDev-labelled agent PRs matched by AGENT_SIGNATURE
  botlike_user_accounts.csv   User accounts reclassified as bots by BOT_NAME, with activity
  steering_manual_check.csv   40 random human messages addressing an agent, for MANUAL labelling
                              (created only if missing, so manual labels are never overwritten)

Usage:
    python src/qa_checks.py
"""
import argparse
import collections
import json
import random
from pathlib import Path

import pandas as pd

import preprocessing as pp

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data/interim/sample.csv"))
    ap.add_argument("--details", default=str(ROOT / "data/raw/details"))
    ap.add_argument("--out", default=str(ROOT / "results/qa"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample)
    group = {(r.full_name, r.number): (r.agent if isinstance(r.agent, str) else "Human") for r in sample.itertuples()}
    matched, total = collections.Counter(), collections.Counter()
    botlike = collections.Counter()
    steering = []
    for f in sorted(Path(args.details).glob("*.jsonl")):
        repo = f.stem.replace("__", "/")
        for line in f.open():
            pr = json.loads(line)
            g = group[(repo, pr["number"])]
            author, _ = pp.login(pr["author"])
            if g != "Human":
                text = (pr["body"] or "") + "\n".join(c["commit"]["message"] for c in pr["commits"]["nodes"])
                total[g] += 1
                matched[g] += bool(pp.AGENT_SIGNATURE.search(text))
            for kind, nodes in (("review", pr["reviews"]["nodes"]), ("comment", pr["comments"]["nodes"])):
                for n in nodes:
                    who, typ = pp.login(n["author"])
                    if who == author:
                        continue
                    if typ == "User" and pp.BOT_NAME.search(who):
                        botlike[who] += 1
                    if not pp.is_bot(who, typ) and pp.STEERING.search(n.get("body") or ""):
                        steering.append({"author_group": g, "kind": kind, "repo": repo, "pr": pr["number"],
                                         "text": (n["body"] or "")[:300]})

    recall = pd.DataFrame({"agent_prs": pd.Series(total), "matched": pd.Series(matched)})
    recall["share_matched"] = recall.matched / recall.agent_prs
    recall.to_csv(out / "signature_recall.csv")
    pd.Series(botlike, name="actions").sort_values(ascending=False).rename_axis("login").to_csv(
        out / "botlike_user_accounts.csv")
    manual = out / "steering_manual_check.csv"
    if not manual.exists():
        random.seed(7)
        pd.DataFrame(random.sample(steering, 40)).assign(manual_label="").to_csv(manual, index=False)
    print(recall.round(3))
    print(f"bot-like User accounts: {len(botlike)} ({sum(botlike.values())} actions); "
          f"human messages addressing an agent: {len(steering)}")


if __name__ == "__main__":
    main()
