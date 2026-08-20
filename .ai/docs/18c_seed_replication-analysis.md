# Lab 18c analysis: B-structured seed replication

## Main conclusion

The constrained B-structured policy replicates. Tier 1 usable rate was 30.3%,
31.0%, and 30.0% across seeds 42, 45, and 47. Candidate-rank percentile was
0.0283, 0.0291, and 0.0289. Tier 2 teacher match was 57.4%, 58.4%, and 57.6%.
The observed range was under one percentage point for both constrained success
rates and under 0.0008 for candidate-rank percentile.

This is not a solved Wordle policy. On Turn 2, Tier 2 teacher match was 19.4%,
16.7%, and 16.7%, below the 21.1% chance expectation for this battery. The model
reliably moves legal words toward the top of the answer ranking, but it does not
reliably choose the entropy teacher's preferred action when uncertainty is
largest.

The next experiment should use constrained full-game decoding before more
training. That will measure how much the replicated legality policy improves
actual play. If gameplay still fails at broad early states, the next training
objective should compare legal actions by teacher value rather than imitate one
target word.

## Findings

### The constrained capability is not a seed-42 accident

Tier 1 usable rate stayed between 30.0% and 31.0%. Candidate-rank percentile
stayed between 0.0283 and 0.0291, far from the 0.5 state-blind reference. Median
teacher rank stayed at 8 or 9 out of 2,315 answers.

Observed stability across three independently initialized adapters -> training
reliably learns to associate feedback with legal answer words -> Lab 18b exposed
a reproducible capability rather than a lucky checkpoint -> constrained
evaluation should remain the primary state-level policy measure.

The seed-level mean Tier 1 rate was 30.43%, with an observed sample standard
deviation of 0.49 points and range of 0.97 points. With three seeds, the 95%
upper confidence bound on the population standard deviation is still 3.10
points. The observed clustering is strong evidence of replication, but not a
precise variance estimate.

### Free generation consistently hides about half of the usable policy

Free-generation usable rates were 14.5%, 16.3%, and 16.3%. The corresponding
constrained gains were 15.8, 14.7, and 13.7 percentage points. Every seed's
Tier 1 rate exceeded every seed's free-generation rate.

Observed within-seed gaps of 13.7 to 15.8 points -> malformed or poorly grounded
generation consistently discards learned policy signal -> free generation is a
secondary interface metric, not a faithful primary measure of policy quality ->
future state-level comparisons should report constrained ranking first.

The exact free-generation behavior remains more variable than constrained
behavior. Its observed range was 1.77 points and its sample standard deviation
was 1.02 points, about twice the Tier 1 standard deviation. Seeds 45 and 47 had
the same 16.3% aggregate rate but disagreed on 48 individual states.

### Strategic ranking replicates overall, but not where games begin

Tier 2 teacher match was 57.4%, 58.4%, and 57.6%, compared with a fixed
per-state chance expectation of 52.2%. Each seed produced 356 to 362 correct
teacher choices out of 620. The descriptive z statistics ranged from 3.92 to
4.65, but the seed-level replication is the stronger evidence because battery
states are not independent training runs.

The aggregate hides a sharp difficulty split:

| Candidate count | Chance | Seed 42 | Seed 45 | Seed 47 |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 88.2% | 91.0% | 91.3% | 91.7% |
| 3-10 | 24.4% | 32.7% | 34.6% | 33.1% |
| 11-50 | 6.2% | 9.8% | 9.8% | 5.9% |
| 51-200 | 1.3% | 0.0% | 0.0% | 0.0% |

Turn 2 is the clearest deployed test. Teacher match was 19.4%, 16.7%, and
16.7%, below its 21.1% chance expectation in every seed. Candidate-rank
percentile at Turn 2 was still strong at 0.088 to 0.091, so the model recognized
which words were legal without ranking the teacher's strategic choice above
them.

Observed stable legality ranking but below-chance Turn 2 teacher choice ->
consistency filtering and strategic ranking are separate learned behaviors ->
constrained gameplay may avoid invented words but can still make poor opening
follow-ups -> any later training intervention should supervise relative value
among legal actions, especially broad Turn 2 candidate sets.

### Aggregate stability does not mean identical policies

Seeds differed on 68 to 72 Tier 1 outcomes in each pair, about 11% of the
battery, despite aggregate rates within one point. Tier 2 pairs differed on 33
to 39 states. Every paired confidence interval for the aggregate seed
difference included zero.

Observed state-level swaps under stable totals -> seeds learn similarly strong
policies but not the same policy -> one checkpoint cannot establish which
individual states or words are robust -> full-game evaluation should run every
seed rather than select the best validation loss.

### Validation loss varied more than constrained behavior

Final validation losses were 1.259, 1.287, and 1.326. Input exposure differed by
less than 0.06%, and supervised-token exposure differed by 19 tokens out of
about 50,300. Despite the validation-loss spread, constrained metrics remained
within one percentage point.

Observed loss variation with stable policy aggregates -> validation loss is not
a precise selector for the deployed constrained metrics -> do not choose one
seed as the representative model from loss alone -> report gameplay for all
three checkpoints.

### Dataset G still has no demonstrated advantage

The single G seed-42 Tier 1 rate was 30.0%, exactly the minimum of the replicated
B range of 30.0% to 31.0%. Its free-generation rate of 16.0% also lies inside
B's 14.5% to 16.3% range. G's Tier 2 rate of 54.5% is below every B seed, whose
range was 57.4% to 58.4%.

Observed G values relative to B's spread -> the Lab 18 Tier 1 and
free-generation differences are ordinary-sized seed effects -> Dataset G has
not shown a benefit -> do not spend more training budget on G without a
specific reason to replicate that treatment.

This lab still cannot claim a replicated B-over-G Tier 2 advantage because G
has only one seed.

## Surprises

The strongest result is how little the constrained metrics moved. The three
seeds used disjoint epoch shuffle seeds and independent LoRA initialization, yet
Tier 1, candidate-rank percentile, and Tier 2 each stayed in a narrow band.
Lab 18b's central result is much more stable than the earlier single-seed design
could establish.

The second surprise is the Turn 2 failure. The aggregate Tier 2 rate looks
respectable because 288 of 620 states have only one or two candidates. On the 36
Turn 2 states, none of the seeds beat chance. More broad-state visitation did
not solve this in Dataset G, and ordinary B seed variation does not solve it
either.

Free generation also varied more than constrained ranking. That supports Lab
18b's diagnosis directly: the unconstrained output interface adds noise on top
of a more stable learned ranking.

## Limitations

- Three seeds confirm the coarse capability but leave a wide upper confidence
  bound on population seed variance. The estimated standard deviations should
  not become universal constants.
- Seed 42 is the pre-existing successful run that motivated replication. Seeds
  45 and 47 are the direct replications; the all-seed mean is descriptive.
- The 620 states share answers, trajectories, and structural patterns. Paired
  state intervals describe disagreement on this battery, not uncertainty across
  the population of training runs.
- Tier 1 restricts actions to the 2,315 answer words. Wordle also permits
  information-seeking guesses outside the answer list.
- The battery is late-state heavy. Only 36 states are Turn 2, and only nine
  states have more than 50 candidates.
- State-level ranking does not establish game solve rate. Errors compound across
  turns and change which states the model visits.
- Dataset G remains a single-seed treatment.

## Implications

1. Build constrained full-game evaluation and run all three B seeds under the
   same gameplay rules. The named hypothesis is that removing malformed actions
   turns the replicated 30% Tier 1 capability into more solved games.
2. Report gameplay by turn and candidate count. The key failure to test is
   broad Turn 2 choice, not aggregate late-state accuracy.
3. Keep free generation as a secondary continuity metric. It is useful for
   interface quality but too noisy to represent learned policy on its own.
4. Do not run the original Lab 19 unchanged yet. If constrained gameplay still
   fails at Turn 2, redesign the training target around pairwise or listwise
   teacher values among legal actions.
5. Do not claim Dataset G is worse from one G seed. The supported conclusion is
   narrower: G has no demonstrated advantage, and its Tier 1 result sits inside
   B's replicated seed range.
