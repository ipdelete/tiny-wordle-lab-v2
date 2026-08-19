# Lab 13 Analysis

## Main conclusion

The current corpus is primarily a constraint-recognition curriculum, not a
gameplay-policy dataset.

In the training split, `NEXT_GUESS` represents only 11.4% of examples and
approximately 11% of tokens. The other 88% of the training signal teaches
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

### Late-game and long-history coverage is thin

Turns 5 and 6 provide 218 training policy examples, or 11.6% of
`NEXT_GUESS`. Across all tasks, only 6.3% of examples occur on turns 5-6, and
29.8% contain histories of depth three or greater.

Late-game states are present, but the absolute policy count is small. This is a
plausible explanation for failures involving long histories or failure to
exploit an obvious final candidate.

### Apparent dataset size overstates state diversity

The full corpus contains 18,824 examples but only 2,304 distinct history
strings. Auxiliary task generation expands the same history into multiple
candidate-validation and choice examples.

One common first-turn `RAISE` feedback history contributes 362 examples.
There are no exact prompt/response duplicates and no ambiguous prompts, so the
corpus is mechanically clean. The issue is weighting: frequently expanded
auxiliary states consume much more training signal than their number of unique
gameplay decisions suggests.

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

## Primary hypothesis for Lab 14

The model's gameplay weakness is caused less by missing answers or target-word
imbalance and more by insufficient supervision on genuine policy decisions,
especially early high-uncertainty states and later long-history states.

## Dataset B implications

Dataset B should:

1. Make `NEXT_GUESS` the primary training signal rather than allowing auxiliary
   constraint tasks to consume approximately 88% of examples and tokens.
2. Preserve complete answer-level teacher trajectories so state weighting
   reflects actual gameplay visitation rather than only unique decision-tree
   nodes.
3. Deliberately add high-candidate-count policy states. This may require
   multiple opening guesses, off-policy but legal histories, or another
   controlled source of early-state diversity.
4. Increase the absolute number of turn-5 and turn-6 policy examples and
   histories of depth three or greater.
5. Limit repeated auxiliary expansions from a single history or assign them a
   fixed budget so they cannot overwhelm policy learning.
6. Match train and dev task proportions and report task-specific metrics even
   when aggregate metrics are retained.
7. Preserve Dataset A unchanged as the control for the Lab 15 LoRA experiment.

The new proportions should be derived from these observed weaknesses and the
deployment objective, not chosen merely to make the dataset look balanced.
