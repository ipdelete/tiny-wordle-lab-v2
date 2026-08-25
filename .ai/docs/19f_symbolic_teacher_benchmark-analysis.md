# Lab 19f analysis: symbolic entropy teachers

## Main conclusion

One-ply entropy is worth testing as the Lab 19e teacher, and the open-action
policy is better than the candidate-only policy under the frozen rule.

On the complete 2,315-answer universe, `open-action-entropy` solved all 2,315
answers with 8,092 penalized turns. `candidate-only-entropy` solved 2,304 and
used 8,330 penalized turns. Open action wins both components of the
lexicographic comparison: 11 more solves and 238 fewer turns.

The reserved battery clears the separate incumbent gate. Exact singleton
closure raises the seed-45 Lab 18d incumbent from 10/19 to 15/19.
Candidate-only entropy solves 18/19, and open-action entropy solves 19/19.
Both symbolic policies therefore beat the incumbent counterfactual.

The frozen outcome is `open_teacher_worth_testing`.
`lab19e_gate_c_may_proceed_unchanged`.

This result validates the symbolic teacher, not a learned policy. It says the
open-action target is worth a distillation test. It does not say the model will
learn or retain that policy.

## Findings

### The incumbent closure counterfactual is exact and raises the real baseline

The historical seed-45 answer-constrained policy solved 10/19. Replaying its
recorded calls and replacing the first action at a singleton state raised the
result to 15/19 with 92 penalized turns.

Six trajectories changed. `GHOST`, `KNIFE`, `SHEEP`, `ALLEY`, and `AUDIO`
became new solves. `CRANE` moved from Turn 5 to Turn 3. Four answers still
failed: `MIGHT`, `ROUND`, `WASTE`, and `POINT`.

Every reconstructed singleton contained the hidden answer as its sole member.
Choosing it produces `GGGGG` and terminates immediately, so no later model
state is invented.

Observed 10/19 becoming 15/19 under terminal-only substitutions -> closure
explains five incumbent failures but leaves four search failures -> a symbolic
teacher must beat 15 solves, not the historical 10 -> both entropy policies
clear the stronger comparison.

### Open-action entropy wins the exhaustive policy comparison

| Policy | Solves | Failures | Solve rate | Penalized turns | Mean solved turn |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate-only entropy | 2,304/2,315 | 11 | 99.525% | 8,330 | 3.582 |
| Open-action entropy | 2,315/2,315 | 0 | 100.000% | 8,092 | 3.495 |

The paired result is not driven only by the 11 recovered failures. Open action
finished earlier on 483 answers, tied candidate-only on 1,516, and finished
later on 316. The net paired difference was 238 fewer penalized turns.

Candidate-only needed Turn 6 for 47 solves and still failed 11 games. Open
action needed Turn 6 only twice and had no failures. It shifted work earlier:
1,139 answers finished on Turn 3 and 1,020 on Turn 4, compared with 999 and 919
for candidate-only.

Observed 2,315 versus 2,304 solves -> open actions remove the candidate-only
policy's horizon failures -> the extra action space improves complete games,
not only one-step entropy -> use open-action entropy as the unchanged Lab 19e
target.

### The reserved battery agrees with the universe result

| Policy | Reserved solves | Penalized turns | Turn distribution, 1 through 6 |
| --- | ---: | ---: | --- |
| Incumbent singleton closure | 15/19 | 92 | 0, 0, 5, 4, 3, 3 |
| Candidate-only entropy | 18/19 | 69 | 0, 1, 8, 9, 0, 0 |
| Open-action entropy | 19/19 | 68 | 0, 0, 8, 11, 0, 0 |

`WASTE` was the candidate-only failure. Open action solved it on Turn 4.
Open action also solved `BRICK` one turn earlier. It took one extra turn on
`SLATE`, `CRANE`, and `BANAL`, but still ended with one fewer penalized turn
overall because it recovered `WASTE`.

Candidate-only has the lower mean turn among its 18 solves, 3.444 versus 3.579.
That conditional mean hides its failure. The preregistered penalized total
correctly ranks open action, 68 versus 69.

Observed 19/19 and 68 turns versus 18/19 and 69 -> the small battery points in
the same direction as the exhaustive run -> the open-policy result is not an
artifact of unrelated universe answers -> the reserved gate passes.

### Noncandidate actions are uncommon but consequential

Open action selected a noncandidate at 164 of 2,478 tree states, 6.62%.
Candidate-only selected none by definition.

The policies both visited 1,704 candidate sets. Of the 295 shared
non-singleton sets, they differed on 111, or 37.63%. Every one of those
disagreements was an open-policy noncandidate choice. The open action had
strictly higher entropy in all 111 cases. The gain averaged 0.486 bits, with a
range of 0.035 to 1.600 bits.

Observed a 6.62% global exploratory-state rate alongside 11 recovered
candidate-only failures -> open exploration is a narrow intervention with
whole-game effect -> constraining the teacher to candidates discards useful
probes -> retain the 2,315-action teacher for Gate C.

### Ties do not explain the open-policy advantage

Entropy ties were common. Candidate-only had 670 tied states, a mean tie size
of 1.425, and a maximum of 6. Open action had 527 tied states, a mean tie size
of 106.41, and a maximum of 1,826. Large open ties occur in small candidate
sets because many legal guesses induce the same partition.

The frozen open tie-break prefers a remaining candidate, then lexical order.
Therefore an open noncandidate can win only when no candidate reaches the
maximum entropy. The 111 shared-state disagreements all had positive entropy
gains beyond the `1e-12` tie tolerance.

Observed no tie-equivalent disagreement -> noncandidate choices are not lexical
noise -> the solve gain comes from strictly better one-ply partitions -> an
open teacher teaches information unavailable to a candidate-only target.

### Candidate-only failures are horizon failures, not missed closure calls

The 11 candidate-only failures were `FOYER`, `GONER`, `GRAZE`, `MATCH`,
`SWORE`, `TATTY`, `WASTE`, `WATCH`, `WATER`, `WIGHT`, and `WILLY`. Seven
finished Turn 6 with one candidate and four with two candidates. Singleton
closure was never available before the final unsolved action; the singleton
appeared only after the turn budget was spent.

Open action solved all 11 on Turn 4 or Turn 5.

Observed late candidate collapse after the final allowed guess -> the
candidate-only policy gathers enough information one move too late -> adding a
closure rule cannot repair these games under the six-turn limit -> earlier
open probes address the actual failure.

### The exhaustive benchmark is small enough to rerun

The candidate-only and open trees contained 2,303 and 2,478 nodes. The two tree
builds took 0.570 seconds together. Full game materialization took 0.432 and
0.427 seconds. The recorded notebook runtime was 2.201 seconds, with no Torch
import.

The vectorized entropy calculation agreed with `EntropyExpert` probes to a
maximum absolute error of `2.665e-15`. The final policy and disagreement audit
cached 2,471 entropy states.

Observed exhaustive coverage in 2.201 seconds -> shared candidate-state trees
remove redundant per-answer work -> future teacher checks can run before model
training at negligible cost -> keep this symbolic benchmark as a preflight for
teacher changes.

## Surprises

### Candidate-only entropy is nearly perfect, but nearly perfect is not tied

Candidate-only solved 99.525% of the universe. A 19-answer battery could easily
miss that remaining 0.475%. It happened to contain one of the failures,
`WASTE`, but the exhaustive run is what establishes the 11-game gap.

The result weakens the idea that candidate-only entropy is an adequate simpler
substitute. It is very strong, but the frozen rule ranks solves before turn
count. Open action wins decisively on that component.

### A small exploratory rate changes the ceiling

Only 6.62% of open tree states chose outside the candidate set. That modest
rate recovered every candidate-only failure and reduced the penalized total by
238 turns.

The useful distinction is not "explore often." It is "allow exploration when a
noncandidate produces a strictly better partition." The candidate-preferring
tie-break kept open action conservative everywhere else.

### Mean solved turns can reverse the reserved interpretation

Candidate-only had a lower conditional mean among reserved solves, yet lost the
preregistered comparison because it failed `WASTE`. This is a concrete example
of why the failure penalty and lexicographic rule were frozen before the run.

## Limitations

- The open action space contains the 2,315 possible answers, not the larger
  legal Wordle guess list. The true unrestricted one-ply ceiling may be higher.
- Entropy assumes a uniform prior over remaining answers and optimizes one
  feedback step. It does not prove optimal expected game length under a
  frequency prior or a globally planned policy.
- The result uses `RAISE` as the fixed opener and a six-turn game. A different
  opener or horizon can change the visited trees and the policy ordering.
- The universe result is exhaustive for this answer list, so it has no sampling
  error. It does not measure robustness to a different lexicon or pattern
  implementation.
- The incumbent counterfactual covers only the 19 reserved seed-45 Lab 18d
  trajectories. Its exactness does not turn that battery into a population
  estimate.
- No learned model was loaded. The benchmark cannot establish target
  learnability, preservation under training, or deployed rank quality after
  distillation.
- Entropy ties use the existing expert convention of absolute tolerance
  `1e-12`. The vectorized implementation matched expert probes well inside that
  threshold, but the tie rule remains an operational floating-point
  definition.

## Implications

Lab 19e Gate C may proceed unchanged with the open-action entropy target. The
named hypothesis is now:

> A learned answer-constrained ranker can acquire the strict noncandidate
> entropy advantages that let the symbolic open policy solve all 2,315 answers,
> without losing incumbent ranking or singleton closure.

Gate C should still be judged as a learning experiment. The symbolic ceiling
does not excuse model regressions. Compare the trained policy with the frozen
incumbent on reserved solves, penalized turns, singleton closure, and full-list
rank preservation. Also check whether the learned policy improves on the 111
shared states where open and candidate-only teachers make strictly different
choices.
