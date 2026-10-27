"""Editable starting policy for delayed-feedback bandit research.

Only this file is supplied as the editable source. The evaluator calls the
three public methods below once per decision round. This starter implements
a deliberately simple, global UCB estimate from feedback received so far.
It does not account for delayed observations or changes in the arm means.
"""

import math


class Policy:
    def __init__(self):
        self.counts = [0] * 6
        self.reward_sums = [0.0] * 6

    def observe(self, feedback):
        """Receive zero or more (played_t, arm, reward) events."""
        for _played_t, arm, reward in feedback:
            self.counts[arm] += 1
            self.reward_sums[arm] += reward

    def select_arm(self, t):
        """Return an integer arm in [0, 5] for decision round t."""
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm
        scores = [
            self.reward_sums[arm] / self.counts[arm]
            + 0.55 * math.sqrt(math.log(t + 2) / self.counts[arm])
            for arm in range(6)
        ]
        return max(range(6), key=lambda arm: (scores[arm], -arm))

    def record_pull(self, t, arm):
        """Record the arm pulled at t; the initial policy needs no action log."""
        return None
