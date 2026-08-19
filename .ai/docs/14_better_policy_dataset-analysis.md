# Lab 14 Analysis

## Main conclusion

Dataset B is a materially different policy curriculum, not a cosmetic resample
of Dataset A.

It changes the training signal from 10.7% policy tokens to 59.3%, expands train
policy coverage from 1,876 to 3,099 unique states, and fills the measured
high-uncertainty holes with 235 unique states at 11+ candidates and 40 at 51+.
It also preserves more deep and late-game states in absolute terms while
removing the repeated auxiliary amplification identified in Lab 13.

The dataset is suitable for the Lab 15 controlled LoRA test, with two important
conditions:

1. Lab 15 must equalize the training-token budget rather than epochs.
2. Evaluation must monitor target and action collapse because trajectory
   weighting has reintroduced target-frequency concentration.

Lab 14 establishes that the intervention changed the intended data dimensions.
It does not establish that those changes improve gameplay; that remains the
causal question for Lab 15.

## Findings

### Dataset A remained an immutable control

The notebook fingerprints all three Dataset A files before generation and
verifies the same fingerprints afterward:

| Split | SHA-256 |
| --- | --- |
| Train | `e54bb9cd2f17c2aef3ae0b6c835c987528525dac722a9b9e2076d4685e26dbf4` |
| Dev | `e241f2e9918058a7f7653c977c8314e01b792a0af3069dffd7351882dc948a3a` |
| Test | `986b15b8094d40ef24251ea4b10363ede6b202b3053c64f23e2a7e9496ee28cd` |

**Evidence chain:** unchanged fingerprints -> Dataset A is byte-for-byte
identical to the Lab 06 corpus -> Lab 15 can attribute differences to the new
corpus rather than accidental control regeneration -> train the control from
these exact files.

### Policy supervision now controls optimization

Training allocation changed as follows:

| Measure | Dataset A | Dataset B |
| --- | ---: | ---: |
| Policy rows | 1,876 | 5,669 |
| Policy example share | 11.4% | 64.7% |
| Policy tokens | 150,012 | 415,392 |
| Policy token share | 10.7% | 59.3% |

Dataset B auxiliary tasks are capped by underlying state. In train,
`CHOOSE_VALID` falls from 5,353 rows to 1,605 and `VALID_CANDIDATE` falls from
9,236 to 1,494.

**Evidence chain:** policy token share rises by 48.6 percentage points ->
gradients are now primarily driven by next-action imitation rather than
constraint classification -> the model should spend more capacity on deployed
behavior -> Lab 15 should measure gameplay and task-specific accuracy
separately.

### The strategic coverage gap is genuinely smaller

Rows and unique states tell different stories, so both were measured:

| Candidate count | A rows | A unique states | B rows | B unique states |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 1,656 | 1,656 | 2,071 | 1,741 |
| 3-10 | 208 | 208 | 2,058 | 1,123 |
| 11-50 | 12 | 12 | 963 | 195 |
| 51-200 | 0 | 0 | 573 | 36 |
| 201+ | 0 | 0 | 4 | 4 |

The strongest improvement is not the 1,540 weighted rows at 11+ candidates; it
is the increase from 12 to 235 unique states. Likewise, the original complete
absence of 51+ policy states is replaced by 40 unique states.

**Evidence chain:** unique high-uncertainty states increase from 12 to 235 ->
the model can observe more than terminal answer completion -> early strategic
choice now receives meaningful direct supervision -> expect the largest Lab 15
gain on turn-2 and broad-candidate evaluations.

### Deep and late-game coverage improves in absolute terms

| Measure | Dataset A unique states | Dataset B unique states |
| --- | ---: | ---: |
| History depth 3+ | 1,014 | 1,277 |
| Turns 5-6 | 218 | 270 |

Dataset B's deep-history *proportion* is lower because it adds many early
strategic states. The absolute number of unique deep and late states still
increases.

**Evidence chain:** deep-state counts rise while their share falls -> the lower
share is a denominator effect rather than removal of late-game supervision ->
Dataset B addresses early uncertainty without discarding the existing deep
curriculum -> compare Lab 15 performance by turn instead of relying on one
aggregate history-depth percentage.

### The two design components contribute differently

Dataset B train policy provenance is:

| Source | Rows | Unique states |
| --- | ---: | ---: |
| Canonical trajectories | 4,651 | 2,081 |
| `STARE` trajectories | 311 | 311 |
| `MOUND` trajectories | 211 | 211 |
| `GLYPH` trajectories | 181 | 181 |
| `FJORD` trajectories | 315 | 315 |

Canonical trajectories provide visitation weighting. The four controlled
openings add 1,018 unique states without adding duplicate hypothetical visits.
They therefore account for 32.8% of Dataset B's unique train policy states but
only 18.0% of policy rows.

**Evidence chain:** alternative sources add one row per new history while
canonical rows repeat encountered states -> reweighting and state creation
remain mechanically distinct inside one corpus -> a later source-based ablation
can isolate them if the combined Lab 15 intervention succeeds.

### Auxiliary amplification is removed

Effective train exposure per state changed from:

| Task | Dataset A examples/state | Dataset B examples/state |
| --- | ---: | ---: |
| `NEXT_GUESS` | 1.000 | 1.829 |
| `CHOOSE_VALID` | 2.510 | 1.000 |
| `VALID_CANDIDATE` | 4.330 | 1.000 |

Dataset B intentionally permits canonical policy repetition because it
represents answer-weighted state visitation. Auxiliary rows receive no such
amplification.

**Evidence chain:** auxiliary repetition falls to one while policy repetition
rises -> repeated optimization now follows game-state visitation rather than
candidate-option generation -> policy behavior should become easier to learn
than auxiliary shortcuts -> inspect whether Lab 15 reverses the Part I gap
between curriculum accuracy and gameplay.

### The gameplay boundary is fair at the exact-state level

The fixed 19 gameplay answers are excluded from Dataset B policy metadata.
More importantly, every exact state on their canonical expert paths is removed
from Dataset B. The measured train overlap is:

| Corpus | Reserved canonical path states in train |
| --- | ---: |
| Dataset A | 0 |
| Dataset B | 0 |

All examples sharing a first-guess/feedback branch remain in one split, so a
parent state and its deterministic descendants cannot cross split boundaries.
The notebook also independently reconstructs candidate sets and uncached expert
actions for 50 states.

**Evidence chain:** both corpora have zero exact reserved-path overlap ->
Dataset B cannot win merely by memorizing the expert test trajectory ->
gameplay differences can support the data-distribution hypothesis -> retain
this assertion in every regenerated Dataset B.

## Surprises

### Visitation weighting reintroduces target concentration

Lab 13 eliminated target concentration as an explanation for Dataset A:

| Measure | Dataset A | Dataset B |
| --- | ---: | ---: |
| Train policy rows | 1,876 | 5,669 |
| Unique targets | 1,876 | 2,160 |
| Maximum target frequency | 1 | 121 |
| Top-10 target row share | 0.5% | 12.3% |

This is not an accidental duplicate-generation bug. Common states are visited
by many possible answers, and the deterministic expert assigns the same action
to each visit. Nevertheless, the optimizer sees target frequency, not the
reason for it.

**Evidence chain:** the top ten actions now supply 12.3% of policy rows ->
realistic visitation weighting creates a new collapse pressure -> a model may
overproduce common teacher actions even though state coverage improved -> Lab
15 must report generated-guess frequency, repetition, and state consistency.

If collapse appears, the next controlled variant should cap or transform
canonical visit weights while preserving the same unique states. The current
Dataset B should remain frozen as the first test.

### Dataset B is smaller despite being strategically broader

Dataset A train contains 16,465 rows and 1,397,673 tokens. Dataset B train
contains 8,768 rows and 700,843 tokens. Dataset B has about half the training
tokens but substantially more policy states.

This weakens the intuition that better coverage requires a larger corpus. It
also creates a training-volume confound if both corpora are trained for the same
number of epochs.

### The alternative openings mostly add nonterminal depth

The four alternative sources add 1,075 `3-10` states across all splits, in
addition to 199 `11-50`, 37 `51-200`, and 6 `201+` states. Their primary value
is therefore not only broad turn-2 coverage; they also create histories where
meaningful uncertainty survives beyond the opening.

This supports the refined Lab 13 hypothesis about long histories combined with
nontrivial candidate sets.

## Limitations

### Lab 15 tests the combined redesign

Dataset B changes policy allocation, canonical visitation weighting, auxiliary
budgets, state coverage, and split construction together. The `source` column
keeps the components identifiable, but one A-versus-B training run cannot say
which component caused an improvement.

The first Lab 15 result should therefore be described as a test of the combined
data hypothesis. A reweight-only or alternative-state-only ablation becomes
justified after the combined intervention shows a benefit.

### Training volume must be controlled

Equal epochs would expose the Dataset A model to about twice as many training
tokens as Dataset B. Lab 15 must use an equal total training-token budget or an
equivalent fixed optimizer-step protocol. Otherwise, corpus composition and
optimization volume remain confounded.

### Alternative openings are deliberately off-policy

`STARE`, `MOUND`, `GLYPH`, and `FJORD` create legal histories that the canonical
teacher does not normally visit. This increases robustness and strategic
coverage, but transfer to the deployment path is still an empirical question.
Failure to improve would not prove that high-uncertainty states are useless; it
could mean their prompt histories are too far from the model's on-policy
distribution.

### Opening policy is outside this comparison

Neither train corpus contains the no-history state. The current benchmark asks
the model to generate turn 1, which would mix an untrained opening decision into
the policy comparison. Lab 15 should seed the same `RAISE` guess and feedback
for both models, then score learned decisions from turn 2 onward.

This narrows the conclusion. Lab 15 will test post-opening policy, not whether
Dataset B teaches the model to choose an opening.

### The internal test split is too small for headline conclusions

Branch grouping produces 8,768 train rows, 1,135 dev rows, and only 21 internal
test rows. The 21-row test split is an integrity artifact, not a credible final
evaluation set. The fixed 19-game gameplay evaluation remains the relevant
end-to-end test.

### The teacher has a known evaluation failure

The canonical expert fails to solve reserved answer `WASTE` within six turns.
This means exact imitation of this teacher does not imply a 100% solve ceiling
on the fixed gameplay set. Report `WASTE` separately when interpreting model
failures.

### Hard trajectories remain selectively represented

Ten development answers fail under the canonical expert and are excluded from
the canonical policy component. Alternative trajectories retain locally
expert-chosen eligible states even when their full trajectory does not solve.
This asymmetry is documented, but it means Dataset B is not a uniform sample of
all hard games.

## Implications for Lab 15

The pre-registered Lab 15 comparison should:

1. Train Dataset A and Dataset B from the same base model with identical LoRA
   rank, target modules, optimizer, learning rate, batch construction, and
   checkpoint-selection rule.
2. Equalize total training tokens or optimizer steps, not epochs.
3. Seed the same `RAISE` opening for both models, then use the same gameplay
   code from turn 2 onward. Preserve the zero reserved-path overlap assertion.
4. Use the metric hierarchy below instead of choosing whichever result improves.
5. Keep Dataset B frozen for the first comparison.

### Metric hierarchy

The primary metric is the **post-opening history-consistent non-repeated guess
rate**. The numerator counts model calls that produce a format-valid five-letter
guess, do not repeat an earlier guess, and satisfy every prior feedback row. The
denominator is every model call after the seeded `RAISE` turn across all held-out
games.

This produces many per-turn observations and measures the behavior Dataset B
directly trains. A 19-game solve rate is too noisy to carry the comparison by
itself.

Secondary metrics are:

1. solve rate from the fixed `RAISE` opening;
2. candidate-set reduction on valid non-repeated guesses, reported separately
   for broad and narrow candidate sets;
3. held-out `NEXT_GUESS` policy quality by turn and candidate-count bucket.

Guardrails are format-valid output rate, repeat rate, generated top-10 guess
share, frequency of the most common Dataset B teacher targets, and auxiliary
task accuracy. These diagnose regressions but do not replace the primary
metric.

The strongest prediction is not a uniform accuracy increase. Dataset B should
improve early broad-candidate decisions, valid policy actions, and end-to-end
gameplay even if aggregate curriculum accuracy falls because auxiliary tasks
receive less optimization.

Possible outcomes have distinct interpretations:

| Lab 15 result | Interpretation | Next experiment |
| --- | --- | --- |
| Gameplay and broad-state policy improve | The combined data hypothesis is supported | Ablate reweighting versus alternative states |
| Policy accuracy improves but gameplay does not | Prompt/state representation or sequential consistency remains limiting | Proceed to Lab 16 error analysis and Lab 17 representation tests |
| Common-target output collapse appears | Canonical visitation weights are too concentrated | Cap or transform visit weights without changing unique states |
| Neither policy nor gameplay improves | Data distribution was not the dominant bottleneck | Inspect optimization fit and representation before generating more data |

Dataset B is ready to test, but it is evidence for a better experiment, not yet
evidence for a better model.
