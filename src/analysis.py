"""Statistical analysis shared by the RQ notebooks.

Main model (RQ1): Poisson pseudo-maximum-likelihood with repository and month fixed
effects and standard errors clustered by repository,

    E[y] = exp(b1 * log_size + b2 * ai + b3 * log_size:ai + repo_FE + month_FE),

so every comparison is made within the same repository. b1 is the size elasticity
of review activity for human PRs, b1 + b3 for AI-agent PRs; b3 tests whether the
relationship differs. Poisson PML is consistent for overdispersed counts and,
unlike negative binomial, has no incidental-parameter problem with many fixed effects.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

SIZE_MEASURES = {"log_lines": "Changed lines (log)", "log_files": "Changed files (log)"}
OUTCOMES = {
    "reviews_total": "Reviews",
    "review_comments_total": "Inline review comments",
    "reviewers_total": "Distinct reviewers",
}
# RQ3: the same measures, split by reviewer account type
OUTCOMES_BY_REVIEWER = {
    f"{k}_{kind}": f"{v} ({kind} reviewers)"
    for k, v in [("reviews", "Reviews"), ("review_comments", "Inline review comments"), ("reviewers", "Distinct reviewers")]
    for kind in ("human", "bot")
}
AGENT_LABELS = {"OpenAI_Codex": "Codex", "Copilot": "Copilot", "Claude_Code": "Claude Code",
                "Cursor": "Cursor", "Devin": "Devin"}


def load(path=ROOT / "data/processed/prs.parquet"):
    df = pd.read_parquet(path)
    df["ai"] = (df.group == "ai").astype(int)
    df["log_lines"] = np.log1p(df.lines_changed)
    df["log_files"] = np.log1p(df.files_changed)
    df["any_review"] = (df.reviews_total > 0).astype(int)
    df["author_group"] = df.agent.map(AGENT_LABELS).fillna("Human")
    return df


def concentration(df, top=5):
    """Share of each agent's PRs that come from its `top` largest repositories (dependence on few projects)."""
    rows = []
    for g, d in df.groupby("author_group"):
        counts = d.repo.value_counts()
        rows.append({"author_group": g, "PRs": len(d), "repositories": len(counts),
                     f"share_top{top}_repos": counts.head(top).sum() / len(d), "largest_repo": counts.index[0]})
    return pd.DataFrame(rows).set_index("author_group")


def describe(df, by="author_kind"):
    """Median, mean and share of zeros for size and review-activity variables."""
    cols = ["lines_changed", "files_changed", *OUTCOMES]
    rows = []
    for name, g in df.groupby(by):
        for c in cols:
            rows.append({by: name, "variable": c, "PRs": len(g), "median": g[c].median(),
                         "mean": g[c].mean(), "share_zero": (g[c] == 0).mean()})
    return pd.DataFrame(rows).pivot(index="variable", columns=by, values=["median", "mean", "share_zero"]).round(3)


def binned_means(df, size, outcome, by="author_kind", bins=10):
    """Mean outcome (with 95% CI) per size decile, deciles computed on the pooled data."""
    d = df.assign(bin=pd.qcut(df[size].rank(method="first"), bins, labels=False))
    g = d.groupby([by, "bin"]).agg(x=(size, "median"), mean=(outcome, "mean"),
                                   sd=(outcome, "std"), n=(outcome, "size")).reset_index()
    g["ci"] = 1.96 * g.sd / np.sqrt(g.n)
    return g


def fit_interaction(df, outcome, size, fe="repo + month", interact="ai"):
    """Poisson FE model with a size x author-type interaction; returns the fitted model."""
    fml = f"{outcome} ~ {size} * {interact} | {fe}"
    return pf.fepois(fml, data=df, vcov={"CRV1": "repo"})


def coef_and_vcov(fit):
    names = list(fit._coefnames)
    b = pd.Series(np.asarray(fit.coef()), index=names)
    V = pd.DataFrame(np.asarray(fit._vcov), index=names, columns=names)
    return b, V


def lincom(fit, weights, label):
    """Estimate, SE, 95% CI and p-value of a linear combination {coef_name: weight}."""
    b, V = coef_and_vcov(fit)
    v = pd.Series(0.0, index=b.index)
    for k, w in weights.items():
        v[k] = w
    est, se = float(v @ b), float(np.sqrt(v @ V @ v))
    return {"group": label, "elasticity": est, "se": se, "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se, "p": 2 * stats.norm.sf(abs(est / se))}


def slopes(fit, size, interact="ai"):
    """Size elasticity for each author type (human: b1, AI: b1 + b3) and the difference."""
    inter = f"{size}:{interact}"
    return pd.DataFrame([lincom(fit, {size: 1}, "Human"),
                         lincom(fit, {size: 1, inter: 1}, "AI agent"),
                         lincom(fit, {inter: 1}, "Difference (AI - Human)")])


def rq1_table(df, outcomes=OUTCOMES, sizes=SIZE_MEASURES):
    """All outcome x size-measure models: elasticities per author type plus model info."""
    rows = []
    for y in outcomes:
        for s in sizes:
            fit = fit_interaction(df, y, s)
            t = slopes(fit, s)
            t.insert(0, "size", sizes[s])
            t.insert(0, "outcome", outcomes[y])
            t["n_used"] = fit._N
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def hurdle(df, outcome, size, fe="repo + month"):
    """Two-part robustness check: (1) logit for any activity, (2) Poisson for the amount when > 0."""
    d = df.assign(any_y=(df[outcome] > 0).astype(int))
    part1 = pf.feglm(f"any_y ~ {size} * ai | {fe}", data=d, family="logit", vcov={"CRV1": "repo"})
    part2 = pf.fepois(f"{outcome} ~ {size} * ai | {fe}", data=d[d[outcome] > 0], vcov={"CRV1": "repo"})
    return part1, part2


# ---------------------------------------------------------------- RQ2: agents

def agents_in(df, min_prs=100):
    """Agents with enough PRs for a separate slope (Google Jules is already excluded in preprocessing)."""
    n = df[df.group == "ai"].agent.value_counts()
    return [a for a in AGENT_LABELS if n.get(a, 0) >= min_prs]


def add_agent_terms(df, size, agents):
    """Dummy and size-interaction column per agent; human PRs are the reference group."""
    d = df[df.agent.isna() | df.agent.isin(agents)].copy()
    for a in agents:
        d[f"is_{a}"] = (d.agent == a).astype(int)
        d[f"{size}_x_{a}"] = d[size] * d[f"is_{a}"]
    return d


def fit_agents(df, outcome, size, agents, fe="repo + month"):
    d = add_agent_terms(df, size, agents)
    rhs = " + ".join([size] + [f"is_{a}" for a in agents] + [f"{size}_x_{a}" for a in agents])
    return pf.fepois(f"{outcome} ~ {rhs} | {fe}", data=d, vcov={"CRV1": "repo"})


def equal_slopes_test(fit, size, agents):
    """Wald test H0: all agents have the same size slope (chi-square, df = #agents - 1)."""
    b, V = coef_and_vcov(fit)
    names = [f"{size}_x_{a}" for a in agents]
    R = np.zeros((len(agents) - 1, len(b)))
    for i in range(1, len(agents)):
        R[i - 1, b.index.get_loc(names[i])] = 1
        R[i - 1, b.index.get_loc(names[0])] = -1
    r = R @ b.values
    chi2 = float(r @ np.linalg.solve(R @ V.values @ R.T, r))
    return chi2, len(agents) - 1, float(stats.chi2.sf(chi2, len(agents) - 1))


def rq2_table(df, agents, outcomes=OUTCOMES, sizes=SIZE_MEASURES):
    """Per-agent size elasticities (and difference to human PRs) + equal-slopes test per model."""
    rows, tests = [], []
    for y in outcomes:
        for s in sizes:
            fit = fit_agents(df, y, s, agents)
            res = [lincom(fit, {s: 1}, "Human") | {"vs_human": np.nan, "p_vs_human": np.nan}]
            for a in agents:
                r = lincom(fit, {s: 1, f"{s}_x_{a}": 1}, AGENT_LABELS[a])
                diff = lincom(fit, {f"{s}_x_{a}": 1}, "")
                res.append(r | {"vs_human": diff["elasticity"], "p_vs_human": diff["p"]})
            t = pd.DataFrame(res)
            t.insert(0, "size", sizes[s])
            t.insert(0, "outcome", outcomes[y])
            t["n_used"] = fit._N
            rows.append(t)
            if len(agents) > 1:
                chi2, dof, p = equal_slopes_test(fit, s, agents)
                tests.append({"outcome": outcomes[y], "size": sizes[s], "chi2": chi2, "df": dof, "p": p})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(tests)


# ---------------------------------------------------------------- RQ3: reviewer type

def reviewer_mix(df, by):
    """Share of PRs reviewed by nobody / humans only / bots only / both (reviews, author and agent excluded)."""
    h, b = df.reviews_human > 0, df.reviews_bot > 0
    mix = np.select([~h & ~b, h & ~b, ~h & b], ["No review", "Human only", "Bot only"], "Human and bot")
    return (pd.crosstab(df[by], mix, normalize="index")
            .reindex(columns=["No review", "Human only", "Bot only", "Human and bot"], fill_value=0))


def rate_ratios(df, outcomes=OUTCOMES, size="log_lines", points=(10, 100, 1000)):
    """Within-repository rate ratio AI / human at given PR sizes: exp(b2 + b3 * log(1 + size)).
    Below 1 = agent PRs receive less review activity than equally large human PRs in the same repository."""
    rows = []
    for y, label in outcomes.items():
        fit = fit_interaction(df, y, size)
        for s in points:
            r = lincom(fit, {"ai": 1, f"{size}:ai": float(np.log1p(s))}, label)
            rows.append({"outcome": label, "size": s, "rate_ratio": np.exp(r["elasticity"]),
                         "ci_low": np.exp(r["ci_low"]), "ci_high": np.exp(r["ci_high"]), "p": r["p"]})
    return pd.DataFrame(rows)
