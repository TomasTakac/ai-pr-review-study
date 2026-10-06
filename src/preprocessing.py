"""Build the analysis dataset (one row per PR) from the collected GitHub data.

Input:  data/interim/sample.csv and data/raw/details/*.jsonl (see github_collector.py)
Output: data/processed/prs.parquet and data/processed/exclusions.csv

Review activity is counted only for reviewers other than the PR author and the
agent's own working account (e.g. copilot-swe-agent acting on its own Copilot PR).
Reviewers are split into bots (GitHub account type Bot, or a bot-like login such as
openshift-ci-robot) and humans (all other accounts). Human discussion comments that
address an agent ("@copilot fix ...") are counted separately as steering. Formal reviews
addressed to an agent remain reviews (they often carry code feedback) but are also
counted in steering_reviews_human for a robustness check.

Usage:
    python src/preprocessing.py
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code"]  # Google_Jules excluded (too few repos)

# Accounts through which an agent creates PRs, pushes commits or answers instructions. On a PR of
# that agent their activity is the agent itself, not a review (on other PRs they count as bots).
AGENT_WORKING_ACCOUNTS = {
    "Copilot": {"copilot-swe-agent"},
    "Devin": {"devin-ai-integration"},
    "Cursor": {"cursor"},
    "Claude_Code": {"claude"},
    "OpenAI_Codex": {"chatgpt-codex-connector"},
    "Google_Jules": {"google-labs-jules"},
}

# Bots running on ordinary User accounts (e.g. openshift-ci-robot, pytorchmergebot) are recognised
# by name, which raises the low recall of GitHub's account type (Golzadeh et al. 2022).
BOT_NAME = re.compile(r"(bot|robot)s?$|(^|[-_])(bot|robot)([-_]|$)", re.IGNORECASE)

# Traces of a coding agent having WRITTEN code in the PR (PR body or commit messages); human PRs
# with these are excluded. Traces of review bots are deliberately not matched: Cursor Bugbot
# summaries (cursor.com/dashboard), "Open in Devin Review" badges, and commits created by
# applying a bot's review suggestion ("Co-authored-by: Copilot", "... devin-ai-integration[bot]").
AGENT_SIGNATURE = re.compile(
    r"generated with \[?claude code|claude\.ai/code|co-authored-by: claude[^\n]*noreply@anthropic\.com"
    r"|cursoragent|co-authored-by: cursor agent"
    r"|chatgpt\.com/codex|codex task"
    r"|app\.devin\.ai/sessions"
    r"|jules\.google",
    re.IGNORECASE,
)

# A human message that mentions an agent handle is treated as steering the agent.
STEERING = re.compile(r"@(copilot|codex|claude|cursor|devin|jules)\b", re.IGNORECASE)


def login(actor):
    """Normalised login; GraphQL returns bot logins without the "[bot]" suffix."""
    if not actor:
        return "ghost", "User"
    return actor["login"].lower().removesuffix("[bot]"), actor["__typename"]


def is_bot(who, typ):
    return typ == "Bot" or bool(BOT_NAME.search(who))


def pr_row(pr, agent):
    author, author_type = login(pr["author"])
    excluded = {author} | AGENT_WORKING_ACCOUNTS.get(agent, set())
    text = (pr["body"] or "") + "\n".join(c["commit"]["message"] for c in pr["commits"]["nodes"])
    row = {
        "author": author, "author_type": author_type, "author_bot_named": bool(BOT_NAME.search(author)),
        "created_at": pr["createdAt"], "state": pr["state"], "merged": pr["mergedAt"] is not None,
        "is_draft": pr["isDraft"],
        "additions": pr["additions"], "deletions": pr["deletions"],
        "lines_changed": pr["additions"] + pr["deletions"], "files_changed": pr["changedFiles"],
        "has_agent_signature": bool(AGENT_SIGNATURE.search(text)),
        "truncated": pr["reviews"]["totalCount"] > len(pr["reviews"]["nodes"])
                     or pr["comments"]["totalCount"] > len(pr["comments"]["nodes"]),
    }
    for kind in ("human", "bot"):
        for k in ("reviews", "review_comments", "reviewers", "comments"):
            row[f"{k}_{kind}"] = 0
    row["steering_reviews_human"] = row["steering_review_comments_human"] = row["steering_comments_human"] = 0
    reviewers = {"human": set(), "bot": set()}
    for r in pr["reviews"]["nodes"]:
        who, typ = login(r["author"])
        if who in excluded:
            continue
        kind = "bot" if is_bot(who, typ) else "human"
        row[f"reviews_{kind}"] += 1
        row[f"review_comments_{kind}"] += r["comments"]["totalCount"]
        reviewers[kind].add(who)
        if kind == "human" and STEERING.search(r.get("body") or ""):  # still a review; tracked for robustness
            row["steering_reviews_human"] += 1
            row["steering_review_comments_human"] += r["comments"]["totalCount"]
    for c in pr["comments"]["nodes"]:
        who, typ = login(c["author"])
        if who in excluded:
            continue
        kind = "bot" if is_bot(who, typ) else "human"
        if kind == "human" and STEERING.search(c.get("body") or ""):
            row["steering_comments_human"] += 1
            continue
        row[f"comments_{kind}"] += 1
    for kind in ("human", "bot"):
        row[f"reviewers_{kind}"] = len(reviewers[kind])
    for k in ("reviews", "review_comments", "reviewers", "comments"):
        row[f"{k}_total"] = row[f"{k}_human"] + row[f"{k}_bot"]
    return row


def build(sample_csv, details_dir):
    sample = pd.read_csv(sample_csv)
    log = [("sampled PRs", len(sample))]

    rows = []
    for f in sorted(Path(details_dir).glob("*.jsonl")):
        repo = f.stem.replace("__", "/")
        agents = sample[sample.full_name == repo].set_index("number").agent.to_dict()
        for line in f.open():
            pr = json.loads(line)
            agent = agents.get(pr["number"])
            rows.append({"repo": repo, "number": pr["number"], **pr_row(pr, agent if isinstance(agent, str) else None)})
    df = sample[["full_name", "number", "group", "agent", "aidev_id"]].rename(columns={"full_name": "repo"}).merge(
        pd.DataFrame(rows), on=["repo", "number"], how="inner")
    log.append(("details fetched", len(df)))

    steps = [
        ("not Google Jules", lambda d: d.agent.isna() | d.agent.isin(AGENTS)),
        ("human PR without agent signature", lambda d: (d.group == "ai") | ~d.has_agent_signature),
        ("human PR authored by a User account without a bot name",
         lambda d: (d.group == "ai") | ((d.author_type == "User") & ~d.author_bot_named)),
        ("reviews/comments not truncated (<=100)", lambda d: ~d.truncated),
    ]
    for name, keep in steps:
        df = df[keep(df)]
        log.append((name, len(df)))

    df["month"] = df.created_at.str[:7]
    df["author_kind"] = df.group.map({"ai": "AI agent", "human": "Human"})
    exclusions = pd.DataFrame(log, columns=["step", "remaining_prs"])
    return df.reset_index(drop=True), exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data/interim/sample.csv"))
    ap.add_argument("--details", default=str(ROOT / "data/raw/details"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    args = ap.parse_args()
    df, exclusions = build(args.sample, args.details)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "prs.parquet", index=False)
    exclusions.to_csv(out / "exclusions.csv", index=False)
    print(exclusions.to_string(index=False))
    print(f"\nsaved {len(df):,} PRs ({(df.group == 'ai').sum():,} AI / {(df.group == 'human').sum():,} human) "
          f"from {df.repo.nunique()} repositories -> {out / 'prs.parquet'}")


if __name__ == "__main__":
    main()
