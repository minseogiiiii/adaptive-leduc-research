# A2 follow-up hypothesis — Observation bottleneck after A1 rejection

This hypothesis was formulated AFTER the original A1 failed, and will be
evaluated ON NEW seed 20261022, not on the previous confirm seed 20261020.

**Question:** How distinguishable are Calling and Aggressive opponent types
using a single LEGAL terminal hand, and does changing our own probe policy
improve discrimination at an economic cost?

Null control: a stationary CFR probing agent. TWO predeclared alternatives:
(1) Uniform legal-action policy; (2) aggressive-biased ActionBiasPolicy(
CFR, CHECK:.7, CALL:.7, BET:2.5, RAISE:2.5, smoothing:.05).

Equal prior over Calling and Aggressive, same deck seeds paired by probe,
type and seat. 24 independent replicate blocks x both seats x 48 hands per
opponent type per probe = 13,824 terminal hands across three policies.

Measures:
- Mean per-hand log likelihood ratio (aggressive vs calling) under true
  aggressive, true calling; empirical Jeffreys divergence proxy.
- AUC of LR classifier, and 50/50 prior Bayes-rule accuracy.
- Optimistic change detector: assuming previous type KNOWN Calling and next
  type is either Calling (90%) or Aggressive (10%), posterior switch >0.5
  requires LR > log(9). Estimate sensitivity and false-positive rate.
  This is a diagnostic upper-information experiment, NOT a deployable
  changepoint filter; it unrealistically presumes known types.
- Exact expected chips per hand of each probe policy, averaged 50/50 over
  Calling and Aggressive opponents. This is model-specific economic cost.
- Paired cluster bootstrap 95% CIs over 24 independent replicate blocks for
  AUC and accuracy differences vs CFR.

Success = at least one alternative has AUC improvement >=0.03 with a lower
95% paired CI >0, AND observed exact payoff loss <=0.05 chips/hand on BOTH
seats. Otherwise A2 provides evidence of a cost-of-information bottleneck.
Results must be reported even if failed. No post-hoc test-set tuning.
