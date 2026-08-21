# Lab 19d analysis: full-list rank drift

## Main conclusion

Lab 19 caused a policy-wide redistribution of score, not a narrow failure to
include a few hard negatives.

Every trained Turn 2 winner differed from its incumbent winner. Of the 57 hard
winners, 56 came from below the incumbent top 32. Of the 57 value winners, 53
came from below the top 32. This confirms state-local negative churn: the
actions that won after training were almost never among the competitors mined
from that state before training.

The stronger result is that useful actions lost rank while a small set of
generic words took over. The hard arms chose only seven distinct Turn 2 winners;
`JOLLY` and `ROYAL` accounted for 47 of 57 states. The value arms also chose
seven; `FALSE`, `JOLLY`, and `ROYAL` accounted for 43 of 57. Candidate mass,
candidate-teacher rank, and singleton-answer rank deteriorated on every seed.

The failure is therefore:

```text
fixed local comparisons
-> broad score redistribution
-> a few new state-insensitive attractors
-> loss of candidate ranking and closure
```

Refreshing negatives may be necessary, but it is not sufficient by itself.
The next training experiment also needs an explicit incumbent-preservation
control and full-list drift checks during training.

## Findings

### Nearly every winner came from outside the mined state boundary

All 114 paired Turn 2 winners changed after training.

| Arm | Winners below incumbent top 32 | Median incumbent rank |
| --- | ---: | ---: |
| Hard | 56/57 | 528 |
| Value | 53/57 | 172 |

The per-seed rates were 94.7% to 100% for hard and 89.5% to 94.7% for value.
Median incumbent ranks ranged from 351 to 657 for hard and 130 to 208 for
value.

Observed winners originating far below the state-specific mined top 32 means
the frozen comparisons stopped describing the trained decision boundary. A
one-time hard-negative set cannot constrain competitors that rise only after
the update.

### The new winners were not globally unseen

Every trained winner appeared in that seed's training supports at least once.
Median exposure was two support presentations for both arms, with a range of
one to ten.

This kills the simplest support-coverage hypothesis. The problem was not that
words such as `JOLLY`, `ROYAL`, or `FALSE` never appeared in training. They
usually appeared under another history, often only once or twice, and therefore
did not receive the comparison needed on the held-out Turn 2 state.

Incumbent-negative exposure was less consistent. Depending on seed, 5.3% to
63.2% of hard winners and 0% to 78.9% of value winners had never occupied an
incumbent-negative slot. Global word exposure and state-specific negative
coverage are different quantities.

### Hard training raised new attractors and suppressed old winners

For hard arms, the eventual winner gained 4.93 to 5.99 summed-log-probability
points on average. The incumbent winner simultaneously lost 10.49 to 11.11.
Words outside the incumbent top 32 gained 6.04 to 6.70 on average, while the
top 32 lost 7.03 to 7.19.

Observed low-ranked words rising while incumbent winners fall means hard
training changed both sides of the margin. The expected consequence is exactly
what occurred: `JOLLY` and `ROYAL` became winners across many unrelated states.
A refreshed negative set would catch those words after they rise, but a
preservation term is still needed to stop useful incumbent rankings from
collapsing during the refresh cycle.

### Value training mostly destroyed the old ordering

The selected value-arm winner changed little in absolute score, losing 0.15 to
2.03 points on average. The incumbent winner lost much more, 12.56 to 13.59
points. Candidates lost 5.16 to 6.37 points on average, the incumbent top 32
lost 9.08 to 10.02, and words outside that boundary gained 4.13 to 5.63.

The value arm therefore did not need to raise its final winner. It removed the
old winner and candidate ordering while shifting score toward the long tail.
That explains why the value objective fit its twelve-action distribution
better than hard but produced worse gameplay.

### Candidate ranking deteriorated on every seed

At Turn 2, the best candidate's mean rank worsened by:

| Seed | Hard | Value |
| ---: | ---: | ---: |
| 42 | 35.7 | 117.9 |
| 45 | 30.8 | 109.2 |
| 47 | 37.2 | 152.7 |

Candidate probability mass fell by 0.198 to 0.211 for hard and 0.233 to 0.248
for value. Candidate-teacher rank worsened by 88 to 195 places for hard and 338
to 396 for value.

Observed candidate loss across all seeds means Lab 19 damaged the
state-consistency capability established in Labs 17 and 18. The expected
consequence is weaker information-seeking and more history-inconsistent play,
which matches the lower solve rates.

Open-teacher scores show why absolute score change is not enough. Hard arms
raised open-teacher scores by 2.47 to 3.10 points, yet open-teacher rank still
worsened on seeds 45 and 47 because competing words rose farther. Full-list
rank, not isolated score movement, is the deployment quantity.

### Singleton closure failed because the answer fell deep into the list

At the first singleton state:

| Arm | Median sole-candidate rank | Mean candidate mass |
| --- | ---: | ---: |
| Incumbent | 1.5 to 3 | 0.278 to 0.363 |
| Hard | 41 to 168.5 | 0.0035 to 0.0332 |
| Value | 381 to 996 | 0.0005 to 0.0012 |

Incumbents closed 38.5% to 50% of these games. Hard fell to 6.3% to 18.8%;
value closed none.

The singleton target inside Lab 19 was one-hot on the sole candidate, yet that
candidate became the 41st to 996th action under full-list ranking. This
contradicts the hypothesis that fixing entropy degeneracy inside the shortlist
would fix closure. The correct answer must beat the entire action list, not
only eleven selected distractors.

## Surprises

### Negative churn was real but too narrow as the main explanation

The top-32 boundary became stale on almost every state, which supports negative
churn. However, all eventual winners had appeared somewhere in training, and
the same small vocabulary dominated many unrelated states.

The better account is state-insensitive score collapse combined with stale
state-level negatives. Iterative mining addresses the second mechanism. It
does not by itself preserve the incumbent policy or prevent a new word from
becoming a cross-state attractor.

### Value did not fail because its final winners were aggressively promoted

Hard actively raised its final winners. Value generally did not. Value failed
more severely because it suppressed incumbent winners, candidates, and the
incumbent top boundary while raising the outside tail.

This distinction matters for the next control. Merely penalizing the current
top wrong word may help hard-style collapse but will not restore candidate mass
lost through broad redistribution.

### One-hot singleton supervision did not protect singleton behavior

Lab 19 made hard and value targets identical on singleton states. Their
full-list singleton rankings still diverged sharply from the incumbents, and
value was worse than hard. Correct target shape inside a restricted support did
not provide global closure pressure.

## Limitations

- Turn 2 provides 57 paired states across three seeds and 19 answers. Later
  trajectory states are policy-dependent and cannot be treated as paired.
- Singleton summaries describe each policy's visited states. Differences
  combine policy quality with trajectory selection.
- Support exposure is word-level across all training states. Exact held-out
  gameplay states were excluded, so it cannot measure the missing
  state-specific comparison directly.
- Summed log-probability changes can reflect sequence calibration and token
  length. The rank, mass, and fixed-word margin results are the stronger
  evidence.
- The notebook diagnoses the result of one update budget, optimizer, and
  twelve-action construction. It does not isolate which optimization choice
  caused the redistribution.
- The analysis uses persisted gameplay score vectors rather than scoring every
  trained model on all 466 dev states. Turn 2 and singleton behavior are the
  deployed slices implicated by Labs 18d and 19.

## Implications

Lab 20 should still test policy-created states against static controls, but all
arms need the same policy-preservation mechanism. Otherwise the experiment may
measure another collapse rather than the value of the data source.

The Lab 20 design should add:

1. **A frozen anchor suite.** Include paired Turn 2 states, broad states,
   sharp states, and singleton states. Score all 2,315 answers at
   preregistered checkpoints.
2. **An incumbent-preservation control.** Apply the same replay or KL anchor to
   rollout-correction, static-random, and static-matched arms so the data-source
   comparison remains clean.
3. **Policy-drift stop rules.** Stop an arm when candidate mass, candidate
   rank, singleton rank, or cross-state winner concentration deteriorates
   according to rules fixed before training.
4. **Winner-concentration reporting.** Count unique full-list winners and the
   share captured by the most frequent words. Loss and shortlist accuracy did
   not expose the `JOLLY`/`ROYAL`/`FALSE` collapse.
5. **A separate dynamic-negative experiment.** If Lab 20 later tests refreshed
   hard negatives, compare it against the preserved incumbent under the same
   update budget. Do not fold this intervention only into the rollout arm and
   confound data source with objective.

The next experiment should test this named hypothesis:

> Policy-created expert corrections improve deployment only when continued
> training preserves the incumbent's useful full-list ordering.

That keeps Lab 20 focused on its intended variable, where the corrective states
come from, while preventing Lab 19's failure from silently recurring.
