# Lab 13 Analysis

## Main conclusion

The current corpus is primarily a constraint-recognition curriculum, not a
gameplay-policy dataset.

In the training split, `NEXT_GUESS` represents only 11.4% of examples and
10.6% of tokens. The other 88% of the training signal teaches
`VALID_CANDIDATE` and `CHOOSE_VALID`. Those auxiliary tasks can help the model
learn Wordle constraints, but they do not directly teach the deployed behavior:
choosing the next guess.

This also means aggregate validation accuracy is a weak proxy for gameplay.
A model can perform well on the dominant auxiliary tasks while remaining a poor
Wordle policy.

## Findings

### Policy examples mostly teach answer completion

The median `NEXT_GUESS` candidate count in the training split is 1. Of the
training policy examples, 88.3% have at most two remaining candidates.

The policy corpus therefore concentrates on emitting an answer after most of
the uncertainty has already been removed. It provides relatively little
practice choosing informative guesses from broad candidate sets.

### Early policy decisions are scarce

The training split contains no turn-1 `NEXT_GUESS` example and only 68 turn-2
examples. Turns 3 and 4 contribute 1,590 examples, or approximately 85% of all
training policy examples.

This distribution is particularly important because early decisions determine
the rest of a trajectory. If deployment asks the model to select an opening or
operate after only one feedback row, the corpus offers little direct
supervision for that behavior.

The deterministic opening and prompt deduplication explain part of this shape.
Many answers share the same early history and expert action, so those examples
collapse into a small number of unique policy states. That is mechanically
reasonable, but it means the resulting corpus is state-uniform rather than
representative of how frequently states occur across complete games.

### Deep histories exist, but late-game policy coverage is thin

Turns 5 and 6 provide 218 training policy examples, or 11.6% of
`NEXT_GUESS`. Across all tasks, only 6.3% of examples occur on turns 5-6, and
29.8% contain histories of depth three or greater.

Policy-specific history depth is healthier than the aggregate number suggests:
54.1% of training `NEXT_GUESS` examples contain histories of depth three or
greater. History depth by itself is therefore not obviously deficient.

The important interaction is that deep-history policy states have almost always
collapsed to tiny candidate sets. On turns 4-6, only 40 policy examples have
3-10 candidates and none has more than 10. Late-game states are present, but
the absolute count is small and their strategic diversity is narrow. This is a
more precise hypothesis for history-consistency and late-game failures than
simply saying that the corpus needs longer histories.

### Policy turn-by-difficulty coverage has clear holes

The training `NEXT_GUESS` matrix is:

| Turn | 1-2 candidates | 3-10 candidates | 11-50 candidates |
| ---: | ---: | ---: | ---: |
| 2 | 33 | 28 | 7 |
| 3 | 649 | 140 | 5 |
| 4 | 763 | 33 | 0 |
| 5 | 174 | 5 | 0 |
| 6 | 37 | 2 | 0 |

There are no training policy examples with 51 or more candidates. Only 12 have
11-50 candidates, and all of those occur on turns 2-3. This confirms that the
policy corpus mostly teaches completion after uncertainty has already been
removed. The general corpus's apparently healthy candidate-count distribution
comes from the auxiliary tasks and does not describe policy supervision.

### Apparent dataset size overstates state diversity

The full corpus contains 18,824 examples but only 2,304 distinct history
strings. Auxiliary task generation expands the same history into multiple
candidate-validation and choice examples.

One common first-turn `RAISE` feedback history contributes 362 examples.
There are no exact prompt/response duplicates and no ambiguous prompts, so the
corpus is mechanically clean. The issue is weighting: frequently expanded
auxiliary states consume much more training signal than their number of unique
gameplay decisions suggests.

Effective state reuse makes that weighting explicit:

| Task | Examples | Unique histories | Examples per history |
| --- | ---: | ---: | ---: |
| `VALID_CANDIDATE` | 10,516 | 2,303 | 4.566 |
| `CHOOSE_VALID` | 6,004 | 2,303 | 2.607 |
| `NEXT_GUESS` | 2,304 | 2,304 | 1.000 |

Each policy history receives one direct action label, while the same underlying
state receives an average of 7.173 auxiliary examples across the two constraint
tasks. Nominal task share therefore understates how repeatedly the corpus
reinforces auxiliary behavior on the same states.

### Target-frequency collapse is not a bottleneck

All 2,304 `NEXT_GUESS` targets are unique in the complete corpus. The ten most
common targets collectively account for only 0.43% of policy examples.

This eliminates the proposed hypothesis that policy behavior is dominated by a
small set of frequent target guesses. The unique-target result follows from the
symbolic expert's decision-tree structure: a selected candidate occupies one
branch of that tree and is not repeatedly selected at unrelated nodes.

The problem is therefore the distribution of policy states, not repeated
policy labels.

### Answer coverage is nearly complete

The corpus covers 2,314 of the 2,315 answer words. `RAISE` is the only missing
answer in the metadata.

Vocabulary coverage is not the obvious bottleneck. Covering nearly every answer
has not guaranteed coverage of the strategic decisions needed to reach those
answers.

### Train and dev differ most in task mix

Train and dev are close on turn, difficulty, and history depth. Their total
variation distances on those dimensions are approximately 0.02.

Task mix has a larger total variation distance of 0.064:

| Task | Train share | Dev share |
| --- | ---: | ---: |
| `VALID_CANDIDATE` | 56.1% | 54.5% |
| `CHOOSE_VALID` | 32.5% | 27.8% |
| `NEXT_GUESS` | 11.4% | 17.8% |

Task-specific metrics remain comparable, but aggregate train/dev metrics mix
different behaviors in different proportions.

## Connection to Part I behavior

These findings provide concrete dataset hypotheses for the model failures
measured in Part I:

| Observed model failure | EDA finding that could explain it |
| --- | --- |
| Repeated guesses and weak history consistency | Direct policy supervision is small relative to repeated auxiliary supervision on the same states |
| Poor early-game decisions | Training contains 68 turn-2 policy examples and no turn-1 policy example |
| Solve rate around 0.5% | Policy examples mostly occur after the candidate set has already collapsed to one or two words |
| Good curriculum accuracy but poor gameplay | Approximately 88% of examples and 89.4% of tokens train easier auxiliary behaviors |

These are causal hypotheses, not conclusions that EDA alone can prove. Lab 15
must test them by changing the data while holding the model and training setup
as constant as practical.

## Primary hypothesis for Lab 14

The model's gameplay weakness is caused less by missing answers or target-word
imbalance and more by insufficient supervision on genuine policy decisions,
especially early high-uncertainty states and later states that combine long
histories with nontrivial candidate sets.

## Dataset B implications

Dataset B requires two separate interventions. Reweighting can change which
known states dominate optimization, but it cannot create policy states absent
from the deterministic expert tree.

### Lever 1: Reweight existing states

1. Make `NEXT_GUESS` the primary training signal rather than allowing auxiliary
   constraint tasks to consume approximately 88% of examples and 89.4% of
   tokens.
2. Preserve complete answer-level teacher trajectories so state weighting
   reflects actual gameplay visitation rather than only unique decision-tree
   nodes.
3. Limit repeated auxiliary expansions from a single history or assign them a
   fixed budget so they cannot overwhelm policy learning.
4. Match train and dev task proportions and report task-specific metrics even
   when aggregate metrics are retained.

### Lever 2: Create missing strategic states

1. Deliberately add high-candidate-count policy states. This may require
   multiple opening guesses, off-policy but legal histories, or another
   controlled source of early-state diversity.
2. Increase the absolute and strategic diversity of turn-5 and turn-6 policy
   examples. Do not rebalance on history depth alone: 54.1% of policy examples
   already have depth-three-or-greater histories, but nearly all have tiny
   candidate sets.

Replaying more answers through the same deterministic expert will reproduce the
same decision tree and the same early-state collapse. Lab 14 therefore needs an
explicit state-generation policy, not merely a larger trajectory count.

### Controlled comparison

Preserve Dataset A unchanged. Define Dataset B with a recorded sampling recipe
for both levers, then use the fixed Lab 15 LoRA setup to test whether the new
policy allocation and state coverage improve deployed gameplay.

The new proportions should be derived from these observed weaknesses and the
deployment objective, not chosen merely to make the dataset look balanced.
