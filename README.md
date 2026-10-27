# Delayed-feedback bandit workspace

Edit policy.py only. Keep class Policy and its three methods:

1. observe(feedback) receives a list of (played_t, arm, reward) tuples.
   Each event is delivered once, at the start of an observation round.
2. select_arm(t) returns one Python integer from 0 through 5.
3. record_pull(t, arm) is called immediately after the selected arm is
   accepted; it permits a policy to track actions whose rewards have not
   arrived.

The evaluator calls observe, select_arm, then record_pull on each of
900 rounds. A reward generated at round played_t is delivered at
played_t + delay + 1. Delays are between 0 and 18 rounds. In this synthetic
setting successful outcomes use delays 10 through 18, while unsuccessful
outcomes use delays 0 through 2. The six Bernoulli arm means are unknown and
change at four boundaries, after every 180 rounds. One arm has the highest
mean in each segment, and the leading arm changes across adjacent segments.

The policy receives no arm means, future rewards, future delays, segment
leaders, episode seed, or evaluator files. The score is mean dynamic regret:
on each round, the gap between the current best arm mean and the chosen arm
mean, averaged over 900 rounds. Lower is better. Observed rewards and the
public delay bounds are the only feedback about the latent means.

The eval_dev directory contains developer-side evaluation cases and answers.
The eval and reference directories are evaluator-only resources. A deployment
must mount only this environment directory (and intentionally released
development material) in the solver workspace. Running the local scorer from
a single shared directory does not provide that isolation.
