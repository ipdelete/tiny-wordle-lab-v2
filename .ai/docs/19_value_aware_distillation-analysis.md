# Lab 19 analysis: value-aware distillation

## Main conclusion

Lab 19 did not improve either diagnosed gameplay failure. Both continuation
arms learned their twelve-action training objective, but both became worse than
their own incumbents when ranked over all 2,315 answers. The hard arm retained
some gameplay ability and consistently beat the value arm, while the value arm
solved none of the 57 reserved games.

This is evidence against the current frozen-shortlist continuation method, not
against distillation in general. The central transfer failure is now:

```text
learn the intended ordering inside twelve named actions
                         !=
preserve or improve ranking over all 2,315 actions
```

The full run completed in 22,440 seconds, about 6 hours 14 minutes. The memory
watchdog reported a lowest system availability of 358 GiB, and the training
arms peaked at 13.92 GiB of MPS driver memory.

## Findings

### The shortlist objectives were learned

On the 466-state dev shortlist, both trained arms moved strongly toward their
targets:

- Incumbent KL to the value target was 6.46 to 6.51 nats.
- Hard-arm KL fell to 1.25 to 1.49 nats.
- Value-arm KL fell further to 1.08 to 1.12 nats.
- Hard top-1 rose from 9.2% to 9.9% for the incumbents to 14.6% to 18.0% for
  hard and 11.4% to 13.5% for value.
- Broad dev regret fell from 0.84 to 0.86 bits for the incumbents to 0.53 to
  0.59 for hard and 0.44 to 0.52 for value.

Observed lower shortlist KL and regret means the optimizer changed the policy
in the requested direction on the supported actions. The failure therefore
cannot be explained as "training did nothing."

### Full-list gameplay collapsed

The incumbents solved 10 of 19 answer-constrained games for every seed. The
hard arms solved 4, 4, and 1. The value arms solved 0, 0, and 0.

Across seeds, the mean solve-rate change relative to the incumbent was:

- hard: -36.8 percentage points;
- value: -52.6 percentage points.

Every free-generation arm solved 0 of 19 games and terminated invalidly in all
19 games. The trained policies therefore lost the incumbents' limited
free-generation ability as well as their constrained-ranking ability.

### Broad-state strategy became worse in deployment

Turn 2 open-action regret for the incumbents was 1.05 to 1.45 bits. It rose to
1.45 to 1.61 bits for hard and 1.76 to 2.39 bits for value.

The hard arm added a mean 0.31 bits of regret, and the value arm added 0.84
bits. Candidate mass on Turn 2 also fell from 0.27 to 0.29 for incumbents to
0.07 to 0.08 for hard and 0.04 to 0.05 for value.

The dev shortlist and deployed Turn 2 results point in opposite directions.
The models learned to rank the twelve supported actions better, but that
ordering did not survive competition from the other 2,303 answers.

### Late-game closure became worse

First-singleton close rates for the incumbents were 38.5%, 45.5%, and 50.0%.
They fell to 18.8%, 6.7%, and 6.3% for hard. Every value arm had a 0% close
rate.

The failure is not lack of access to a sharp state. The hard arms reached a
singleton in 47 games and closed only 4; the value arms reached one in 39 games
and closed none. Eighteen of the 27 incumbent constrained failures had ended at
one candidate in Lab 18d. Lab 19 increased that count to 43 of 48 hard failures
and 47 of 57 value failures.

The models still reduce uncertainty, but continued training makes them less
likely to choose the answer once it is known.

### Hard consistently beat value

Hard beat value on all three preregistered metrics for every seed:

- solve rate was higher by 21.1, 21.1, and 5.3 percentage points;
- first-singleton closure was higher by 18.8, 6.7, and 6.3 points;
- Turn 2 regret was lower by 0.86, 0.31, and 0.43 bits.

The soft target was better on its own dev cross-entropy and broad shortlist
regret, yet worse in full-list gameplay. Better imitation of the soft
twelve-action distribution is therefore not evidence of a better deployed
policy.

## Surprises

### Hard-negative mining did not close the support gap

Every training support included the frozen incumbent's top action and four
actions drawn from its top-32 ranking. That was enough to move those supported
actions, but not enough to control the new ranking after 1,029 updates. A fixed
set of pre-training hard negatives can become stale as the policy changes.

The likely failure is negative churn: suppressing the incumbent's original
mistakes allows previously lower-ranked, unsupported words to become the new
mistakes. Lab 19 did not rank all 2,315 actions during each update, so its loss
could not see that boundary moving.

### The sharp target was correct but insufficient

Singleton targets were one-hot on the sole candidate in both arms, so entropy
degeneracy was removed exactly as intended. Closure still fell sharply.

This weakens the hypothesis that the old closure failure was caused only by the
teacher utility. The target also needs enough leverage against every competing
action, and continued updates must preserve the incumbent behavior that already
put candidates near the top.

### The notebook's printed conclusion understates the arm separation

The notebook prints that the result "would not distinguish hard from soft
distillation" because neither arm beat its incumbent. The preregistered verdict
table does distinguish them: hard beat value on solve rate, closure, and Turn 2
regret for all three seeds.

The primary conclusion remains that both interventions failed. The secondary
conclusion should be that the soft value target failed more severely than the
matched hard control.

## Limitations

- There are three seed-level replications and 19 reserved answers. Per-state
  and per-answer rows are diagnostics, not independent replications.
- Lab 19 changes the incumbents through continued LoRA training. It does not
  test a fresh student, a separate action-value head, or full-logit
  distillation.
- The loss normalizes over twelve actions, not all 2,315 answers. The experiment
  cannot establish whether the target utilities would work under a full-action
  objective.
- Hard negatives were mined once from each frozen incumbent. The experiment did
  not refresh them as the trained policy's decision boundary moved.
- The broad training pool includes all available broad Turn 2 and 11-plus
  candidate source states, but it is still thinner than realistic deployment.
- The current artifacts do not directly measure whether each trained model's
  new top-ranked errors were absent from its training supports.

## Implications

Do not extend this run with more epochs or a different softmax temperature. The
value arm already fit its supported objective better and played worse, so
stronger optimization of the same objective is not justified.

The next step should be a no-training full-list diagnostic on the trained
adapters:

1. Rank all 2,315 answers on the fixed dev states for each incumbent and arm.
2. Measure the trained top action's teacher utility, candidate consistency, and
   whether that word appeared in any relevant training support.
3. Compare score changes for supported actions, original incumbent negatives,
   and newly emerged top-ranked actions.

That directly tests the negative-churn explanation. If new unsupported actions
replaced the mined negatives, the next intervention should refresh hard
negatives during training or optimize a model over the full action list. It
should also anchor the incumbent policy, for example with replay or a KL
penalty, so improving selected comparisons does not erase state consistency
and closure.

Broad strategy and sharp closure may still need different losses. The evidence
from this run says they cannot safely share the current twelve-action
continuation objective: hard damages both, and the softer value distribution
damages both more.
