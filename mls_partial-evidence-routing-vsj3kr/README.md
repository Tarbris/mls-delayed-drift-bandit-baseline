# Continuous evidence routing under changing calibration

Implement `Agent.start(config)` and `Agent.act(observation)` in `agent.py`. You may
add Python modules. NumPy, SciPy, and scikit-learn are available. A fresh agent is
started for each independently generated episode.

An incoming case has three continuous context coordinates and an unknown binary
outcome. Eight services offer real-valued evidence scores. You may purchase
reports sequentially, then submit a probability and a binary decision, or defer
the case for specialist review. A score is not itself a calibrated probability.
Its relationship to the outcome depends on context and an unobserved response
mechanism, and scores can share noise. Live service calibration also changes.

Minimize decision/referral loss plus evidence and audit costs while maintaining
useful decision coverage and calibrated probabilities. Learn from partially
observed historical audits and the reports purchased during the episode.

## Available information and statistical assumptions

Each episode provides 3,000 historical cases with context and gold outcome. Only
2 to 4 of the eight scores are visible per historical case. The observed sources
are sampled uniformly, independently of context, outcome, and score values.
Every source and every source pair therefore has positive observation support.
Missing reports are represented by `null`, not zero.

The live stream contains 800 cases. Context coordinates lie in `[-1,1]^3`. The
context/outcome relationship and the pre-drift response law stay stable within
an episode. Response means vary smoothly with context, including interactions
and simple periodic effects. Three unobserved response mechanisms have
context-dependent mixing probabilities. No source has a single outcome
direction that remains correct across all mechanisms. Sources share continuous
noise, and dependence strength also changes with context. Data do not form a
finite table of binary report patterns.

Live source offsets have an affine dependence on context and evolve over time.
There is gradual Gaussian diffusion and occasional abrupt, correlated changes
in those coefficients. These are stochastic changes, not a known periodic
schedule. Their distribution scales are bounded; individual realized Gaussian
increments are not. A typical diffusion standard deviation is 0.004–0.010 per
coefficient per case; abrupt-change probability lies between 0.004 and 0.010 per
case, with coefficient change scale 0.4–0.9. The actual live offset, change times,
response functions, noise loadings, and hidden outcomes are unavailable.

The initial historical archive is collected before live offset drift. Different
episodes have independent response functions, source characteristics, historical
archives, contexts, and drift trajectories. The historical archive, acquired
scores, and optional gold feedback are sufficient inputs for learning; no
external dataset or network access is needed. Any statistically valid method is
allowed. Paying for live audits is optional.

## Protocol

`start(config)` receives `source_count=8`, `case_count=800`, `audit_limit=80`,
`audit_cost=0.04`, `assumptions`, and `history`. Each history row has `x`, `label`,
and eight `reports`, of which 2–4 are non-null. Other historical fields are
ordinary published metadata and do not contain hidden live answers. Return `{}`.

`act(observation)` receives:

- `case_id`: zero-based sequence number;
- `x`: three finite context values;
- `reports`: eight entries, initially all `null`; a purchased report is a finite real score;
- `costs`: all eight query prices, disclosed before action;
- `fp_loss=1`, `fn_loss=1`, `defer_loss=0.18`;
- `audit_remaining` and `feedback`.

Choose one action:

```json
{"type":"query","source":3}
{"type":"decide","decision":1,"probability":0.87,"audit":false}
{"type":"defer","audit":true}
```

`query` purchases one previously unpurchased source and causes another call for
the same case. Source indices must be integers 0–7. A report can be purchased
only once per case. Booleans and floating-point indices are invalid.

`decide` and `defer` irrevocably commit the current case. A decision must be
integer 0 or 1; probability is a finite number in `[0,1]` representing the
probability of outcome 1. Audit flags must be Boolean and respect the budget.

An audit returns the correct outcome only on the first call for the next case.
It includes the previous context and only the scores purchased in that case.
It does not reveal unpurchased scores, repair an earlier mistake, or reduce an
earlier cost. An audit of the final case has no future benefit.

## Objective and acceptance criteria

A wrong committed decision costs 1; a correct decision costs 0. A deferral costs
0.18. Every case also pays all disclosed query prices and 0.04 if audited.
Always deferring therefore has mean loss 0.18 and zero coverage.

Over the full set of 12 independently generated evaluation episodes (9,600
live cases), all four population criteria must hold:

- mean total loss at most **0.17**;
- committed-decision coverage at least **0.60**;
- error rate among committed decisions at most **0.10**;
- mean Brier score among committed decisions at most **0.10**.

Brier score is `(probability - outcome)^2`. Error rate and Brier score are
weighted by the number of committed decisions, not averaged as episode ratios.
Per-episode results and upper-tail risk are reported for analysis; there is no
requirement to pass all four criteria separately on every random episode or to
foresee an unobservable change before evidence arrives.

Same-input replay must produce the same action sequence. Use one process and
one thread; network and subprocess creation are unavailable. Formal evaluation
isolates the candidate from hidden data, generator internals, and author
references. Candidate files must not hardcode hidden outcomes or modify the
evaluator. Initialization has 60 seconds, each response has a parent-clock
2-second deadline including transport, memory is 4 GiB, and each episode has
180 CPU seconds. The solver's total work budget is two hours.

## Public development replay

From a workspace containing only `environment/` and `eval_dev/`:

```bash
python eval_dev/run_dev.py --agent environment/agent.py
```

Public fixtures include answers for analysis. Formal episodes have independent
archives and trajectories and are not present in this workspace. Development
scores are diagnostic, not formal acceptance or a measure of LLM-call difficulty.

The current development release contains twelve complete public episodes. The
original two-episode release remains archived with the blind-challenge evidence.
The additional ten episodes provide more development data; no scoring gate or
hidden episode was changed.
