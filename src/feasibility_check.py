"""Feasibility check of the GitHub-API collection on a few test repositories.

Classifies collected PRs into AI (labelled in AIDev), bot-authored, unlabelled AI
(agent signature found but not in AIDev) and human, then compares review counts
of AI PRs with the AIDev `pr_reviews` table as a sanity check.

Usage:
    python src/feasibility_check.py --raw data/raw/github
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

AIDEV_REVISION = "37bbe1533e26cc1e1374917dba1186d1c8a4dc81"  # AIDev-7.6M (v5) snapshot used in the study
AIDEV = f"hf://datasets/hao-li/AIDev-7.6M@{AIDEV_REVISION}/"

# Traces that AI coding agents leave in PR bodies or commit messages.
AGENT_SIGNATURE = re.compile(
    r"generated with \[?claude code|claude\.ai/code|co-authored-by: claude"
    r"|cursor\.com|cursoragent|co-authored-by: cursor"
    r"|chatgpt\.com/codex|codex task"
    r"|devin\.ai|devin-ai-integration"
    r"|jules\.google|google-labs-jules"
    r"|copilot-swe-agent|co-authored-by: copilot",
    re.IGNORECASE,
)


def load_raw(raw_dir):
    rows = []
    for f in sorted(Path(raw_dir).glob("*.jsonl")):
        repo = f.stem.replace("__", "/")
        for line in f.open():
            pr = json.loads(line)
            author = pr["author"] or {"login": "ghost", "__typename": "User"}
            text = (pr["body"] or "") + "\n".join(c["commit"]["message"] for c in pr["commits"]["nodes"])
            reviews = [r for r in pr["reviews"]["nodes"] if r["author"]]
            ext = [r for r in reviews if r["author"]["login"] != author["login"]]
            rows.append({
                "repo": repo, "number": pr["number"], "created_at": pr["createdAt"],
                "author": author["login"], "author_type": author["__typename"],
                "additions": pr["additions"], "deletions": pr["deletions"],
                "changed_files": pr["changedFiles"],
                "has_signature": bool(AGENT_SIGNATURE.search(text)),
                "reviews_total": pr["reviews"]["totalCount"],
                "reviews_truncated": pr["reviews"]["totalCount"] > len(pr["reviews"]["nodes"]),
                "reviews_ext": len(ext),
                "reviews_human": sum(r["author"]["__typename"] == "User" for r in ext),
                "reviews_bot": sum(r["author"]["__typename"] == "Bot" for r in ext),
                "review_comments": sum(r["comments"]["totalCount"] for r in ext),
                "reviewers": len({r["author"]["login"] for r in ext}),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/github")
    args = ap.parse_args()

    gh = load_raw(args.raw)
    repos = gh.repo.unique()
    aidev = pd.read_parquet(AIDEV + "pull_request.parquet", columns=["id", "number", "agent", "repo_url"])
    aidev["repo"] = aidev.repo_url.str.removeprefix("https://api.github.com/repos/")
    aidev = aidev[aidev.repo.isin(repos)]
    gh = gh.merge(aidev[["repo", "number", "agent", "id"]], on=["repo", "number"], how="left")

    gh["group"] = "human"
    gh.loc[gh.author_type == "Bot", "group"] = "bot_author"
    gh.loc[gh.has_signature, "group"] = "unlabelled_ai"
    gh.loc[gh.agent.notna(), "group"] = "ai"

    print("=== PRs per repo and group")
    print(pd.crosstab(gh.repo, gh.group, margins=True))
    print("\nAIDev PRs of these repos found on GitHub:",
          f"{aidev.set_index(['repo', 'number']).index.isin(gh.set_index(['repo', 'number']).index).mean():.1%}")
    print("Bot authors (top):", gh[gh.group == "bot_author"].author.value_counts().head(8).to_dict())

    print("\n=== Size and review activity (median / mean)")
    cols = ["additions", "deletions", "changed_files", "reviews_ext", "reviews_human",
            "reviews_bot", "review_comments", "reviewers"]
    sub = gh[gh.group.isin(["ai", "human"])]
    print(sub.groupby("group")[cols].agg(["median", "mean"]).round(1).T)
    print("\nshare with >=1 external review:", sub.groupby("group").reviews_ext.apply(lambda s: (s > 0).mean()).round(3).to_dict())
    print("PRs with >100 reviews (truncated):", int(gh.reviews_truncated.sum()))

    rv = pd.read_parquet(AIDEV + "pr_reviews.parquet", columns=["pr_id"])
    ai = gh[gh.group == "ai"].copy()
    ai["reviews_aidev"] = ai.id.map(rv.pr_id.value_counts()).fillna(0)
    print("\n=== Sanity check: review counts of AI PRs, GitHub API vs AIDev")
    print("exact match:", f"{(ai.reviews_total == ai.reviews_aidev).mean():.1%}",
          "| correlation:", round(ai[["reviews_total", "reviews_aidev"]].corr().iloc[0, 1], 3))

    out = Path(args.raw).parent.parent / "processed"
    out.mkdir(parents=True, exist_ok=True)
    gh.to_csv(out / "feasibility_prs.csv", index=False)
    print("\nsaved", out / "feasibility_prs.csv")


if __name__ == "__main__":
    main()
