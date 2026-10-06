"""Collect PR size and review activity for AI-agent and human PRs from the GitHub GraphQL API.

Pipeline (each stage is resumable; finished repositories are skipped on re-run):

  prepare  Select repositories from AIDev-pop (>= MIN_AI_PRS AI PRs, not forked) and
           store the AIDev AI-PR labels of those repositories.
  list     Light listing of every PR created in the AIDev window (number, date, author).
  sample   Per repository: all AI PRs (at most AI_CAP_PER_AGENT per agent) plus a random
           sample of human PRs from the same months (HUMAN_RATIO x the AI PRs of that month).
  details  Size, reviews, review comments and comments of every sampled PR, measured the
           same way for both groups.
  status   Print progress.

Token: GITHUB_TOKEN in the environment or `GITHUB_TOKEN=...` in the project's `.env`.

Usage:
    python -u src/github_collector.py all        # whole pipeline
    python -u src/github_collector.py status
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.github.com/graphql"
AIDEV_REVISION = "37bbe1533e26cc1e1374917dba1186d1c8a4dc81"  # AIDev-7.6M (v5) snapshot used in the study
AIDEV = f"hf://datasets/hao-li/AIDev-7.6M@{AIDEV_REVISION}/"
WINDOW_START = "2024-12-24T00:00:00Z"  # first PR in AIDev-pop
WINDOW_END = "2026-03-31T23:59:59Z"    # AIDev v5 cutoff

MIN_AI_PRS = 50          # repository selection: depends only on AI-PR volume, not on review outcomes
AI_CAP_PER_AGENT = 200   # keeps a few very active repositories from dominating
HUMAN_RATIO = 2          # human PRs sampled per AI PR, within the same repository and month
SEED = 42
WORKERS = 3
DETAIL_BATCH = 25

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
LISTING = ROOT / "data/raw/listing"
DETAILS = ROOT / "data/raw/details"

LIST_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number createdAt author { login __typename } }
    }
  }
}
"""

PR_FIELDS = """
fragment PR on PullRequest {
  number databaseId createdAt closedAt mergedAt state isDraft
  author { login __typename }
  additions deletions changedFiles
  body
  commits(first: 10) { totalCount nodes { commit { message } } }
  comments(first: 100) { totalCount nodes { author { login __typename } createdAt body } }
  reviews(first: 100) {
    totalCount
    nodes { author { login __typename } state submittedAt body comments { totalCount } }
  }
}
"""
BODY_LIMIT = 1000  # review/comment text is kept only to detect agent steering ("@copilot fix ...")


def trim_bodies(pr):
    for node in pr["comments"]["nodes"] + pr["reviews"]["nodes"]:
        node["body"] = (node.get("body") or "")[:BODY_LIMIT]
    return pr


# ---------------------------------------------------------------- API access

def load_token():
    token = os.environ.get("GITHUB_TOKEN")
    env = ROOT / ".env"
    if not token and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GITHUB_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"')
    if not token:
        raise SystemExit("GITHUB_TOKEN not found (environment variable or .env file).")
    return token


class Client:
    """Thread-safe GraphQL client that waits for primary and secondary rate limits."""

    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.token = load_token()

    def session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers["Authorization"] = f"bearer {self.token}"
        return self.local.s

    def wait(self):
        delay = self.pause_until - time.time()
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds, reason):
        with self.lock:
            until = time.time() + seconds
            if until > self.pause_until:
                self.pause_until = until
                log(f"pausing {seconds:.0f}s ({reason})")

    def query(self, query, variables, retries=8):
        """Returns `data`, or None when GitHub times out on a heavy query (HTTP 502/504)."""
        for attempt in range(retries):
            self.wait()
            try:
                r = self.session().post(API, json={"query": query, "variables": variables}, timeout=120)
            except requests.RequestException as e:
                log(f"network error: {e}")
                time.sleep(min(300, 5 * 2 ** attempt))
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:  # response cut off in transit -> caller retries with a smaller batch
                    log("truncated response, retrying with a smaller batch")
                    return None
                data = body.get("data") or {}
                errors = body.get("errors") or []
                if any(e.get("type") == "RATE_LIMITED" for e in errors):
                    self.pause(60, "rate limited")
                    continue
                if not data.get("repository") and errors:
                    return {"error": errors}
                rl = data.get("rateLimit")
                if rl and rl["remaining"] < 100:
                    reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                    self.pause((reset - datetime.now(timezone.utc)).total_seconds() + 10, "hourly limit used up")
                return data
            if r.status_code in (403, 429):
                wait = int(r.headers.get("retry-after", 60))
                if r.headers.get("x-ratelimit-remaining") == "0":
                    wait = max(wait, int(r.headers["x-ratelimit-reset"]) - int(time.time()) + 10)
                self.pause(wait, f"HTTP {r.status_code}")
            elif r.status_code in (502, 504):
                return None
            else:
                log(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(min(300, 5 * 2 ** attempt))
        raise RuntimeError("too many failed attempts")


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def repo_file(directory, full_name, suffix):
    return directory / f"{full_name.replace('/', '__')}{suffix}"


def run_parallel(fn, items):
    """Run fn over items in parallel; a failure in one repository is logged and does not stop the rest
    (the repository keeps no .done marker, so a re-run retries it)."""
    def safe(item):
        try:
            fn(item)
        except Exception as e:  # noqa: BLE001
            log(f"FAILED {item if isinstance(item, str) else item[0]}: {type(e).__name__}: {str(e)[:200]}")

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(safe, items), 1):
            if i % 25 == 0 or i == len(items):
                log(f"--- {i}/{len(items)} repositories")


# ---------------------------------------------------------------- stages

def prepare():
    INTERIM.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(AIDEV + "pull_request.parquet", columns=["id", "number", "agent", "repo_id", "repo_url"])
    repo = pd.read_parquet(AIDEV + "repository.parquet", columns=["id", "full_name", "stars", "language", "is_forked"])
    counts = pr.groupby("repo_id").size().rename("ai_prs").reset_index()
    repos = counts.merge(repo, left_on="repo_id", right_on="id").drop(columns="id")
    repos = repos[(repos.ai_prs >= MIN_AI_PRS) & ~repos.is_forked].sort_values("ai_prs", ascending=False)
    repos.to_csv(INTERIM / "repositories.csv", index=False)
    ai = pr[pr.repo_id.isin(repos.repo_id)].merge(repos[["repo_id", "full_name"]], on="repo_id")
    ai[["id", "full_name", "number", "agent"]].rename(columns={"id": "aidev_id"}).to_parquet(INTERIM / "aidev_ai_prs.parquet")
    log(f"prepare: {len(repos)} repositories, {len(ai):,} AI PRs")


def list_repo(client, full_name):
    done = repo_file(LISTING, full_name, ".done")
    if done.exists():
        return
    owner, name = full_name.split("/")
    after, rows, t0 = None, [], time.time()
    while True:
        data = client.query(LIST_QUERY, {"owner": owner, "name": name, "after": after})
        if data is None:
            continue
        if "error" in data:
            log(f"{full_name}: not available ({data['error'][0].get('message', '')[:80]})")
            done.write_text(json.dumps({"prs": 0, "error": data["error"][0].get("message", "")}))
            return
        prs = data["repository"]["pullRequests"]
        for n in prs["nodes"]:
            if WINDOW_START <= n["createdAt"] <= WINDOW_END:
                a = n["author"] or {"login": None, "__typename": None}
                rows.append({"number": n["number"], "created_at": n["createdAt"],
                             "author": a["login"], "author_type": a["__typename"]})
        oldest = prs["nodes"][-1]["createdAt"] if prs["nodes"] else WINDOW_START
        if not prs["pageInfo"]["hasNextPage"] or oldest < WINDOW_START:
            break
        after = prs["pageInfo"]["endCursor"]
    pd.DataFrame(rows, columns=["number", "created_at", "author", "author_type"]).to_csv(
        repo_file(LISTING, full_name, ".csv"), index=False)
    done.write_text(json.dumps({"prs": len(rows), "seconds": round(time.time() - t0, 1)}))
    log(f"{full_name}: listed {len(rows)} PRs in {time.time() - t0:.0f}s")


def list_all():
    LISTING.mkdir(parents=True, exist_ok=True)
    client = Client()
    names = pd.read_csv(INTERIM / "repositories.csv").full_name.tolist()
    run_parallel(lambda n: list_repo(client, n), names)


def sample():
    repos = pd.read_csv(INTERIM / "repositories.csv")
    ai = pd.read_parquet(INTERIM / "aidev_ai_prs.parquet")
    frames = []
    for full_name in repos.full_name:
        f = repo_file(LISTING, full_name, ".csv")
        if not f.exists():
            continue
        lst = pd.read_csv(f)
        lst["month"] = lst.created_at.str[:7]
        labels = ai[ai.full_name == full_name][["number", "agent", "aidev_id"]]
        lst = lst.merge(labels, on="number", how="left")
        # random order + rank within group = sampling without replacement per group
        lst = lst.sample(frac=1, random_state=SEED)
        ai_prs = lst[lst.agent.notna()]
        ai_prs = ai_prs[ai_prs.groupby("agent").cumcount() < AI_CAP_PER_AGENT]
        human_pool = lst[lst.agent.isna() & (lst.author_type == "User")
                         & ~lst.author.fillna("").str.endswith("[bot]")]
        want = ai_prs.month.value_counts() * HUMAN_RATIO
        human = human_pool[human_pool.groupby("month").cumcount() < human_pool.month.map(want).fillna(0)]
        part = pd.concat([ai_prs.assign(group="ai"), human.assign(group="human")])
        part["full_name"] = full_name
        part["listed_ai"], part["listed_human_pool"] = (lst.agent.notna().sum(), len(human_pool))
        frames.append(part)
    s = pd.concat(frames, ignore_index=True)
    s.to_csv(INTERIM / "sample.csv", index=False)
    log(f"sample: {len(s):,} PRs ({(s.group == 'ai').sum():,} AI, {(s.group == 'human').sum():,} human) "
        f"from {s.full_name.nunique()} repositories")


def detail_repo(client, full_name, numbers):
    done = repo_file(DETAILS, full_name, ".done")
    if done.exists():
        return
    out = repo_file(DETAILS, full_name, ".jsonl")
    have = set()
    if out.exists():
        have = {json.loads(line)["number"] for line in out.open()}
    todo = [n for n in numbers if n not in have]
    owner, name = full_name.split("/")
    batch, missing, failures, t0 = DETAIL_BATCH, 0, 0, time.time()
    with out.open("a") as fh:
        i = 0
        while i < len(todo):
            chunk = todo[i:i + batch]
            fields = " ".join(f"p{n}: pullRequest(number: {n}) {{ ...PR }}" for n in chunk)
            q = ("query($owner: String!, $name: String!) { rateLimit { cost remaining resetAt } "
                 f"repository(owner: $owner, name: $name) {{ {fields} }} }}" + PR_FIELDS)
            data = client.query(q, {"owner": owner, "name": name})
            if data is None:  # too heavy or cut off, retry the same PRs in a smaller batch
                failures += 1
                if failures > 8:
                    raise RuntimeError(f"no valid response for PRs {chunk[:3]}... after 8 retries")
                batch = max(1, batch // 2)
                continue
            failures = 0
            if "error" in data:
                log(f"{full_name}: not available ({data['error'][0].get('message', '')[:80]})")
                break
            for n in chunk:
                pr = data["repository"].get(f"p{n}")
                if pr:
                    fh.write(json.dumps(trim_bodies(pr)) + "\n")
                else:
                    missing += 1
            fh.flush()
            i += len(chunk)
            batch = min(DETAIL_BATCH, batch * 2)
    done.write_text(json.dumps({"prs": len(numbers), "missing": missing, "seconds": round(time.time() - t0, 1)}))
    log(f"{full_name}: details for {len(numbers) - missing} PRs in {time.time() - t0:.0f}s")


def details_all():
    DETAILS.mkdir(parents=True, exist_ok=True)
    client = Client()
    s = pd.read_csv(INTERIM / "sample.csv")
    jobs = [(name, g.number.astype(int).tolist()) for name, g in s.groupby("full_name")]
    jobs.sort(key=lambda j: -len(j[1]))
    run_parallel(lambda j: detail_repo(client, *j), jobs)


def status():
    repos = pd.read_csv(INTERIM / "repositories.csv") if (INTERIM / "repositories.csv").exists() else None
    total = len(repos) if repos is not None else "?"
    listed = [json.loads(f.read_text()) for f in LISTING.glob("*.done")] if LISTING.exists() else []
    print(f"list:    {len(listed)}/{total} repositories, {sum(d['prs'] for d in listed):,} PRs, "
          f"{sum('error' in d for d in listed)} unavailable")
    if (INTERIM / "sample.csv").exists():
        s = pd.read_csv(INTERIM / "sample.csv", usecols=["full_name", "group"])
        det = [json.loads(f.read_text()) for f in DETAILS.glob("*.done")] if DETAILS.exists() else []
        fetched = sum(1 for f in DETAILS.glob("*.jsonl") for _ in f.open()) if DETAILS.exists() else 0
        print(f"sample:  {len(s):,} PRs ({(s.group == 'ai').sum():,} AI / {(s.group == 'human').sum():,} human)")
        print(f"details: {len(det)}/{s.full_name.nunique()} repositories, {fetched:,}/{len(s):,} PRs fetched")
    else:
        print("sample:  not yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "list", "sample", "details", "status", "all"])
    stage = ap.parse_args().stage
    if stage == "status":
        return status()
    pidfile = ROOT / "data/logs/collector.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
            raise SystemExit(f"Collector is already running (pid {pidfile.read_text()}); not starting a second one.")
        except ProcessLookupError:
            pass  # stale file from a run that ended
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    t0 = time.time()
    for name, fn in [("prepare", prepare), ("list", list_all), ("sample", sample), ("details", details_all)]:
        if stage in (name, "all"):
            if name == "prepare" and stage == "all" and (INTERIM / "repositories.csv").exists():
                continue
            log(f"=== stage {name}")
            fn()
    log(f"finished in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
