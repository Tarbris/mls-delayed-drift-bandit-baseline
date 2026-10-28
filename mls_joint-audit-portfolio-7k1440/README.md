# Joint measurement portfolio audit

Implement `plan(observations)` and `finish(observations, audit_result)` in
`solver.py`. Extra Python modules in the candidate directory are allowed.
NumPy, SciPy and scikit-learn are available. Network access is disabled. Every
case uses a fresh isolated process; the two calls share that process.

You are producing a comparative audit report for a bank of policies. Later
decision makers may choose different policies because their costs or capacity
requirements differ. Give a useful simultaneous lower guarantee for **every**
policy; selecting just one winner is not the main deliverable.

## Measurement problem

X has four continuous coordinates and is always observed. Z is a binary
pre-treatment covariate that is costly to recover. Z can affect both historical
action assignment and outcomes. The source and target distributions of X and
of Z conditional on X can differ. Policies used at deployment depend only on X.

`pilot_source` is an independent, unbiased gold cohort of 300 records, each with
`x`, `z`, `action`, `reward`, and `reported`. All pilot Z and binary Y are known.
`pilot_target` has 100 independent target records with `x` and `z`, but no Y.
Their `inclusion_probability` is 1 within these fully measured reference cohorts.

`source` has 8,000 other records. It contains X, action A in {0,1,2}, a Boolean
`reported`, `reward` when naturally reported and `None` otherwise, and a public
audit `cell`. Natural reporting may depend on Y and Z. Z is absent from these
records until you buy a source gold measurement, which reveals both Z and Y.
No true logging-propensity vector is provided.

`target` has 5,000 records with X and a public audit cell. A target measurement
reveals only Z. Target outcomes are never available to the candidate.

`policy_bank` has shape [13,3,5]. For policy j and target X, use softmax over
actions of `policy_bank[j] @ [1,X[0],X[1],X[2],X[3]]`. Index 0 is the baseline.
The policy coefficients are independent of the outcome-law parameters. They are
not reward-model coefficients, and no policy requires Z at deployment.

## Identifying assumptions and structural support

Potential outcomes are exchangeable with historical action given (X,Z). The
conditional expected reward `mu(X,Z,a)` is shared by source and target. Source
and target have common joint (X,Z) support, with target/source density ratio at
most 45. This ratio and the logging propensity must be inferred, if your method
needs them. There is no assumption that population shift is a group count ratio.

Historical action 2 has zero support when Z=1 and X[0]>0.8. All other actions
are supported. Each supported action has logging propensity at least 0.08/3.
This is a historical support gap, not an assertion that the action is prohibited.
The public rule is denoted h(X,Z,a); unaudited target Z is not secretly supplied
as a support mask. Gold measurement is randomized with positive coverage, making
the target support distribution estimable.

## One hard budget shared by two kinds of measurement

The total measurement budget is 6,000 cost units. Initial source gold has already
cost 4*300=1,200 and initial target gold has cost 100. Those costs count toward
the total. You have 4,700 units left: a new source gold record costs 4 and a new
target Z record costs 1.

`audit_design` provides 28 `cell_size` values, `cell_cost`, `minimum`, remaining
`budget`, initial costs, and `total_measurement_budget`. Source cells are 0–23;
target cells are 24–27. They are defined from observable X, source A and reporting
status. Return `{"allocation": [m_0,...,m_27]}` from `plan`.

Every m must be an integer between its published minimum and size. Nontrivial
cells have minimum 2. The exact sum of m*cost must not exceed 4,700. After you
commit, exactly m distinct records are sampled uniformly without replacement
from each cell of size N. The first-order inclusion probability is exactly m/N.
The allocation cannot be changed after observing new labels.

`audit_result` contains the committed `allocation`, `source_indices`, their
`source_z` and `source_reward`, `target_indices`, their `target_z`, and exact
confirmation `cost`. These indices refer to the original source or target arrays.

## Required report and evaluation

Return an object with `lower_bounds`, a finite numeric list of length 13 in
[-1,1]. The baseline bound must be 0. You may also return an integer `selection`
as a diagnostic; it is not the primary score or an acceptance gate.

Write d_j(X,a)=pi_j(a|X)-pi_0(a|X). The report's estimand is the sharp robust
improvement on the finite target cohort:

```
L_j = target average of [
  sum_a h(X,Z,a) * d_j(X,a) * mu(X,Z,a)
  + sum_a (1-h(X,Z,a)) * min(d_j(X,a),0)
].
```

The unsupported-action term minimizes over all missing binary-outcome means.
Thus L_j already accounts for genuine nonidentification. The baseline has L_0=0.
The oracle uses this same estimand; it does not reward recovery of unknowable
true outcome benefit outside support.

Report simultaneous one-sided 95% lower confidence bounds for all L_j. Coverage
is assessed empirically on repeated held-out cases. Approximate statistical
methods are permitted; no particular nuisance model or algorithm is required.

For each nonbaseline policy, information loss is
`max(0, L_j - reported_lower_bound_j)`. Its unit is expected binary reward
probability: 0.10 means a ten-percentage-point shortfall in the guaranteed
improvement. The primary score is its mean across policies and cases. A valid
but very broad certificate loses information; an overconfident certificate may
fail joint coverage. Screening and certifying only one policy is allowed, but
all other broad bounds still contribute to the report's information loss.

Functional gates on the complete hidden collection are:

- Joint coverage at least 0.90 for the nominal 95% report.
- Mean information loss at most 0.10.
- Each hidden family's mean information loss at most 0.15.

Hidden cases vary parameter draws and mechanism combinations while respecting
the assumptions above. No hidden labels, generator, reference implementation,
or scoring process is accessible in the candidate sandbox. Inputs are immutable;
copy arrays when needed. Runtime is one CPU thread, 4 GiB address space, and
60 seconds per interface call. The full verifier has a 900-second wall-clock
deadline, with each load/call bounded by its remaining time. An error, timeout,
or failed functional gate returns a nonzero verifier exit status. The starter
gives exact data-free worst-case
bounds; it needs measurements to produce a sufficiently informative report.

## Public development

From the workspace root on Linux with bubblewrap and the supplied scientific
runtime directory:

```
python3 eval_dev/run_dev.py --candidate environment --runtime-root /var/tmp/obm-research-runtime --output dev-result.json
```

Sixty static public examples and their debugging answers are in `eval_dev`.
The original six were retained; 54 independent examples were added after the
independent blind trial. This later expansion does not revise its historical result.
The runner is independent of the hidden generator and references. A development
pass is diagnostic; it is not a complete hidden evaluation or difficulty rating.

The overall deadline is enforced through cooperative checks and RPC timeouts.
It does not claim preemptive termination of arbitrary blocking trusted-parent
generation, process construction, cleanup or filesystem I/O.
