# Lab 18 analysis: Dataset G state distribution

## Main conclusion

Dataset G changed the learned policy, but it did not improve the primary
held-out result enough to replace Dataset B.

On the 620-state paired battery, usable actions rose from 90 to 99:

```text
B-structured  90/620 = 14.5%
G-structured  99/620 = 16.0%
delta                  +1.45 percentage points
paired bootstrap 95% CI: -1.45 to +4.19 points
exact paired p-value: 0.374
```

The paired flips were 45 states won by G and 36 lost by G. This is a small,
uncertain shift rather than a general policy improvement.

The main hypothesis was that moving more supervision toward broad turn-2
states would improve broad-state behavior. The battery does not support that
claim. On states with more than 50 candidates, B produced 2 usable actions in
9 states and G produced none. G also reduced completed games from 5 of 19 to
4 of 19.

G did improve several secondary measures. It raised perturbation-pair
consistency from 4 of 34 to 7 of 34, fixed-state usable actions from 11 of 47
to 13 of 47, and usable turn-2 gameplay actions from 3 of 19 to 7 of 19.
None of those changes was decisive on its paired test, and the turn-2 gain did
not survive through complete games.

The result is best read as:

> Reallocating a fixed policy budget toward alternative openings and broad
> turn-2 states changed which states the model handled, but did not produce a
> reliable overall policy gain.

It says nothing about sequential learning. Both models trained on shuffled,
independent state-action rows.

## Findings

### The primary battery was a null

Both models evaluated the same 620 held-out states. None appeared in either
training corpus.

| Metric | B-structured | G-structured | Change |
| --- | ---: | ---: | ---: |
| Format valid | 575/620, 92.7% | 549/620, 88.5% | -4.2 points |
| History consistent | 90/620, 14.5% | 99/620, 16.0% | +1.45 points |
| Usable | 90/620, 14.5% | 99/620, 16.0% | +1.45 points |
| Teacher match | 46/620, 7.4% | 48/620, 7.7% | +0.3 points |

The usable and history-consistency rows are identical because every
history-consistent output in this battery was also a non-repeated answer-list
word.

The observed value is a 1.45-point gain with a confidence interval that
includes zero. The interpretation is that G changed individual decisions but
did not raise aggregate correctness reliably. The expected consequence is
small, unstable guardrail movement rather than a large gameplay gain. That is
what the other evaluations show.

The format regression matters. G gained nine usable states while losing 26
format-valid outputs. Reweighting the policy data did not fix the output
contract problem seen in Lab 17.

### Broad held-out states did not improve

Battery performance by candidate count was:

| Candidates | States | B usable | G usable |
| --- | ---: | ---: | ---: |
| 1-2 | 288 | 33, 11.5% | 35, 12.2% |
| 3-10 | 272 | 46, 16.9% | 53, 19.5% |
| 11-50 | 51 | 9, 17.6% | 11, 21.6% |
| 51-200 | 7 | 1, 14.3% | 0 |
| 201+ | 2 | 1, 50.0% | 0 |

G's training rows had a mean of 30.2 candidates, compared with 15.9 for B.
That extra broad-state exposure did not improve the nine battery states with
more than 50 candidates.

The sample is small at the broad end, so 0 of 9 is not a precise estimate.
It still contradicts the hoped-for direction. We should not claim that more
broad turn-2 supervision taught broad-state policy.

Turn 2 itself was also mixed. On the 36 battery turn-2 states, usable rate fell
from 7 of 36 to 6 of 36. The separate fixed-opening gameplay probe improved,
but the broader battery did not reproduce that gain.

### Perturbation behavior improved, but not decisively

The exact Lab 16 pairs moved as follows:

| Metric | B-structured | G-structured |
| --- | ---: | ---: |
| Both sides parsed | 24/34 | 27/34 |
| Sensitivity among parsed pairs | 18/24, 75.0% | 23/27, 85.2% |
| Paired consistency | 4/34, 11.8% | 7/34, 20.6% |
| Branch consistency | 19/68, 27.9% | 26/68, 38.2% |

The exact paired-consistency test was `p=0.453`. G gained three fully correct
pairs, but the discordant pattern was not strong enough to distinguish the
models.

The useful part is the direction. G parsed more pairs, changed actions more
often, and obeyed more branches. The expected consequence would be better
fixed-state behavior, which rose modestly from 11 to 13 usable states.

### Fixed states moved toward broader buckets

The 47 fixed states show a redistribution rather than a clean gain:

| Candidate bucket | B usable | G usable |
| --- | ---: | ---: |
| 1-2 | 4/16 | 2/16 |
| 3-10 | 6/17 | 7/17 |
| 11-50 | 0/8 | 1/8 |
| 51-200 | 1/6 | 3/6 |

G lost two near-terminal successes and gained four successes in larger
candidate sets. That pattern fits its training allocation better than the
aggregate battery does.

This result is encouraging as a diagnostic: G learned a different tradeoff.
It is not enough to call G the better policy because the larger 620-state
battery was inconclusive and completed gameplay got worse.

### Turn-2 gameplay improved without improving solve rate

Lab 18 re-ran both adapters under the same rule. A non-answer-list output ends
the game instead of querying an unchanged state repeatedly.

At turn 2, both models faced the same 19 states:

| Metric | B-structured | G-structured |
| --- | ---: | ---: |
| Format valid | 12/19 | 16/19 |
| In answer list | 8/19 | 13/19 |
| History consistent | 3/19 | 7/19 |
| Usable | 3/19 | 7/19 |

There were four G-only usable flips and no B-only flips. The exact paired
test was `p=0.125`. The sample supports a promising turn-2 direction, not a
settled result.

The gain did not carry through the games:

```text
B solved: BRICK, ROUND, SLATE, CRANE, SHEEP
G solved: SHORE, ROUND, SLATE, CRANE
```

Solve rate fell from 5 of 19 to 4 of 19. Invalid termination rose from 14 of
19 to 15 of 19. G made better first decisions in four additional games, then
failed often enough later that it finished one fewer game.

Later-turn call rates are descriptive because the models face different
survivor sets. B reached turn 3 in 7 games and G reached it in 11. By turn 4,
the counts were 4 and 3. These rows cannot support a direct later-turn policy
comparison.

### Auxiliary behavior held

`CHOOSE_VALID` stayed at 47 of 49, or 95.9%. `VALID_CANDIDATE` rose from 90 of
98 to 93 of 98, or 94.9%.

Dataset G did not erase the constraint-recognition behavior established in
Lab 17. This is a guardrail only. The auxiliary prompts expose the decoded
constraints and are easier than free generation.

### Training exposure was closely matched

Both models used 1,029 updates, an effective batch size of 16, four-example
microbatches, and the same LoRA and optimizer settings.

```text
B input tokens: 2,456,934
G input tokens: 2,414,866
G/B ratio:          0.983
```

G processed 49,971 supervised tokens. Its validation loss fell from 8.193 to
1.365. B's validation loss is not a valid comparator because the validation
rows differ.

MPS driver allocation reached roughly 11.30 GiB and stayed there through the
run. The compact-logit fix from Labs 9 and 17 continued to prevent full-prompt
vocabulary logits from exhausting memory.

## Surprises

The clearest surprise is the gap between turn-2 gameplay and the primary
battery. G more than doubled usable actions on the 19 fixed turn-2 states, but
it lost one usable action across the 36 battery turn-2 states. The fixed
opening probe alone would have overstated the benefit.

The broad-state result also went the wrong way. G devoted much more training
mass to broad states, yet scored 0 of 9 on held-out states with more than 50
candidates. More exposure did not make those decisions learnable under this
model and budget.

G did not merely collapse to one new output. Its most common battery outputs,
`FLAKE` and `TRUST`, appeared seven times each. The failure is distributed
across wrong and malformed actions rather than one dominant attractor.

Finally, whole-game filtering reduced unique states from 3,099 to 2,749 and
unique targets from 2,160 to 1,780 while increasing repeated visits. If
anything in G helped, we cannot assign it to turn-2 mass alone. Opening mix,
candidate breadth, repetition, and target diversity all changed together.

## Limitations

This is one training seed. The paired battery gives a much better readout than
19 games, but it does not measure variation across adapters trained on the same
corpus.

The intervention changes several data properties at once:

* alternative-opening share rises from 18.0% to 40.4%;
* turn-2 share rises from 25.5% to 39.1%;
* mean candidates rise from 15.9 to 30.2;
* unique states and targets fall;
* repeated visits rise;
* reserved-answer targets fall from 15 rows to zero.

These changes follow from whole-game filtering and the strict leakage boundary.
They prevent a single-factor causal claim.

The battery has only nine states above 50 candidates. It is strong for the
overall paired comparison but weak for broad-state estimates.

Dataset B and G use different validation state distributions. Validation loss
can diagnose convergence within each run, not relative policy quality.

The answer-only action space remains frozen. Part III will test the larger
allowed-guess vocabulary.

## Implications

Dataset G should not replace Dataset B as the default structured-policy
training corpus. Its primary gain was small and uncertain, and its complete
game solve rate was lower.

The evidence chain is:

> G shifted 13.6 percentage points of policy mass into turn 2 and more than
> doubled alternative-opening exposure -> turn-2 gameplay usability rose on
> one fixed probe -> the 620-state paired gain was only 1.45 points with a
> confidence interval spanning zero -> broad battery states and completed
> games did not improve -> data-distribution tuning is not the main remaining
> bottleneck.

Do not scale to G-2x. The measured problem is no longer a shortage of broad
teacher-game rows. More of the same distribution would increase cost without
evidence that the model can use those states.

Carry the structured representation forward, but keep B-structured as the
incumbent policy control. Lab 19 can revisit distillation with the stronger
state representation and the repaired evaluation stack. Its question should
be whether a richer teacher signal or action-scoring objective transfers policy
quality that hard one-word targets did not.

If we later return to Dataset G, the next experiment should isolate one factor,
such as opening mix or turn-2 allocation, rather than generating more coupled
whole-game data.
