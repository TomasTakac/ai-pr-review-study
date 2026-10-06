# Review Activity in AI-Agent and Human Pull Requests: The Role of Pull Request Size and Reviewer Type

Replication package of an empirical Mining Software Repositories study.

**Author:** Tomáš Takáč, Technical University of Košice (Cybersecurity, Master's study programme)

## Research questions

- **RQ1** How is pull request size (changed lines, changed files) associated with the amount of review activity, and does this relationship differ between AI-agent and human pull requests within the same repositories?
- **RQ2** How does this relationship differ across AI coding agents (Codex, Copilot, Devin, Cursor, Claude Code)?
- **RQ3** How do the results change when review activity is separated into human and bot/agent review activity?

## Data

| Source | Used for |
|---|---|
| [AIDev-7.6M](https://huggingface.co/datasets/hao-li/AIDev-7.6M) (AIDev v5, cutoff 2026-03-31, CC BY 4.0), revision `37bbe1533e26cc1e1374917dba1186d1c8a4dc81` | Selecting repositories and labelling AI-agent PRs and their agent |
| GitHub GraphQL API | PR size and review activity of **both** AI-agent and human PRs, measured identically |

AIDev-7.6M contains no human PRs, and its review tables cover only AI-agent PRs. Human PRs are therefore collected from the same repositories and months via the GitHub API (see `notebooks/01_data_exploration.ipynb`).

## Pipeline

```
AIDev-7.6M ──► prepare ──► list ──► sample ──► details ──► preprocessing ──► RQ1 / RQ2 / RQ3
              (select     (all PRs  (AI PRs +   (size,      (one row per PR,  (models, tables,
               repos)      in window) human 2:1) reviews)    exclusions)       figures)
```

1. **prepare**: AIDev-pop repositories (>100 stars) with at least 50 AI-agent PRs, not forked → 1,267 repositories. Selection depends only on AI-PR volume, not on review outcomes.
2. **list**: every PR created between 2024-12-24 and 2026-03-31 (number, date, author).
3. **sample**, per repository:
   - all AI-agent PRs labelled in AIDev, at most 200 per agent;
   - human PRs (author is a `User` account, not in AIDev) sampled 2:1 within the same month;
   - random seed 42.
4. **details**: additions, deletions, changed files, reviews, inline review comments and discussion comments, each with the author's account type.
5. **preprocessing** (`src/preprocessing.py`):
   - Not counted as review activity: the PR author and, on an agent's PR, that agent's own working account (`copilot-swe-agent`, `devin-ai-integration`, `cursor`, `claude`, `chatgpt-codex-connector`).
   - Bots: GitHub account type `Bot` or a bot-like login (e.g. `openshift-ci-robot`).
   - Steering: human discussion comments addressing an agent (`@copilot …`) are counted separately. Formal reviews addressed to an agent stay reviews, with a robustness check without them.
   - Excluded PRs: Google Jules PRs, human PRs with a trace of an agent writing code, human PRs with a bot-named author, and PRs with more than 100 reviews or comments.
   - Manual quality checks: `results/qa/`.
   - Every exclusion step is logged in `results/tables/exclusions.csv`.
6. **analysis**: Poisson pseudo-maximum-likelihood models with repository and month fixed effects and repository-clustered standard errors (`src/analysis.py`). Robustness: a two-part (hurdle) model and a Wald test of equal slopes across agents.

## Repository structure

```
├── README.md
├── requirements.txt
├── run_analysis.sh               # runs notebooks 02–05 after data collection
├── notebooks/
│   ├── 01_data_exploration.ipynb # AIDev structure and feasibility checks
│   ├── 02_data_preparation.ipynb # analysis dataset + exclusion table
│   ├── 03_rq1_analysis.ipynb     # descriptives, RQ1 models, hurdle robustness
│   ├── 04_rq2_analysis.ipynb     # per-agent models, equal-slopes test, concentration
│   └── 05_rq3_analysis.ipynb     # reviewer mix, steering, human vs bot models
├── src/
│   ├── github_collector.py       # data collection (prepare, list, sample, details)
│   ├── preprocessing.py          # one row per PR
│   ├── analysis.py               # models and tables
│   ├── visualization.py          # figures
│   ├── qa_checks.py              # quality checks of the classification rules (results/qa/)
│   └── feasibility_check.py      # pilot check on test repositories
├── results/
│   ├── tables/                   # CSV tables used in the paper
│   └── figures/                  # PNG + PDF figures used in the paper
└── data/                         # created by the pipeline, not versioned
```

## How to reproduce

**Requirements:** Python 3.14, about 2 GB of disk space, and a GitHub account.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**GitHub token:**
- Create a fine-grained personal access token with read-only access to public repositories (no other permissions).
- Store it in a `.env` file in the project root. The file is git-ignored; never commit it.

```
GITHUB_TOKEN=github_pat_...
```

**Collect the data.**
- This takes about 8–10 hours, because of the GitHub rate limit of 5,000 points per hour.
- It is resumable: re-running the same command continues where it stopped, and a second instance refuses to start.

```bash
nohup caffeinate -is .venv/bin/python -u src/github_collector.py all >> data/logs/collect.log 2>&1 &
.venv/bin/python src/github_collector.py status     # progress
```

(`caffeinate` keeps macOS awake; on Linux, drop it.)

**Run the analysis** (notebooks 02 → 05, executed in place; results in `results/`):

```bash
./run_analysis.sh
```

**Notes on reproducibility:**
- The AIDev revision and the sampling seed are fixed.
- GitHub data changes over time: PRs can be edited, deleted or receive new reviews. A re-collection therefore reproduces the procedure, not necessarily identical counts.
- The collected raw data (`data/raw/`) and the analysis dataset (`data/processed/prs.parquet`) are available from the authors on request.

## Ethics and licensing

- **Data and terms:** AIDev is released under CC BY 4.0. GitHub data are collected through the official API within its terms of service.
- **Scope of collection:** only public repositories are used, and only metadata needed for the analysis is stored. Review and comment texts are truncated to 1,000 characters and used only to detect agent steering.
- **Anonymity:** results are reported in aggregate; no individual developer is identified.
