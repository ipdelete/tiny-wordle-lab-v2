# Lab 16 analysis

## Main conclusion

State insensitivity is the dominant failure.

Under the training prompt, Model B changed its action on 5 of 31 perturbation
pairs where both outputs parsed, a sensitivity rate of 16.1%. Model A changed
on 1 of 31, or 3.2%. Dataset B therefore made generation somewhat more
responsive to state.

That response was not useful policy conditioning. Neither model produced one
pair where both generated actions were consistent with their corresponding
states. Model B had 23 state-insensitive invalid pairs, 3 state-insensitive
partial pairs, and 5 state-misinterpretation pairs. No model produced a
wrong-branch swap.

The evidence supports an explicit state representation in Lab 17. Dataset B
should remain frozen. It changed what the model emits and made Model B more
sensitive to feedback, but the raw history still does not control generation
reliably enough for play.

## The perturbation experiment

The notebook generated 7,796 reachable candidate pairs. It then capped
correlated pairs from the same parent and branch guess, stratified by state
scope and branch source, and selected 34 pairs.

| Pair scope | Controlled off-policy | Fixed opening | Teacher | Total |
| --- | ---: | ---: | ---: | ---: |
| Broad | 10 | 1 | 0 | 11 |
| Mixed | 10 | 1 | 0 | 11 |
| Narrow | 10 | 0 | 2 | 12 |

Each pair shares a parent and branch guess. Its child states differ at exactly
one feedback position, both feedback branches are reachable, and neither
training prompt appears in Dataset A or Dataset B train.

The broad controlled pairs retained an average of 80.6 and 62.5 candidates in
their two branches. The perturbation set therefore tests the broad uncertainty
regime that Dataset B was designed to improve.

## State sensitivity versus correctness

The headline metrics use only pairs where both outputs parsed. Unparseable
pairs are reported separately so format changes cannot inflate sensitivity.

| Model | Interface | Both parse | Parsed pairs | Sensitivity | Either action consistent | Both actions consistent |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Training | 91.2% | 31 | 3.2% | 0.0% | 0.0% |
| A | Deployment | 85.3% | 29 | 13.8% | 0.0% | 0.0% |
| B | Training | 91.2% | 31 | 16.1% | 12.9% | 0.0% |
| B | Deployment | 82.4% | 28 | 10.7% | 3.6% | 0.0% |
| Base | Deployment | 88.2% | 30 | 10.0% | 6.7% | 0.0% |

The base model did not produce parseable words under the training-format task
prompt, so only its deployment result is a useful control.

Model B's training sensitivity exceeded Model A by 12.9 percentage points.
That is evidence that Dataset B affected state response. The absolute rate
remained low: 83.9% of parsed pairs produced the same action after one valid
feedback mark changed.

More importantly, paired consistency was zero for every model and interface.
Action changes did not produce two correct branch-specific choices.

The observed chain is:

> 16.1% sensitivity -> Dataset B increased state response -> 0% paired
> consistency -> response does not implement the feedback constraints ->
> represent the constraints explicitly and retest the same pairs.

## What Model B did on the paired states

Model B's 31 parsed training-format pairs were classified as:

| Diagnosis | Pairs | Share |
| --- | ---: | ---: |
| State-insensitive invalid | 23 | 74.2% |
| State-insensitive partial | 3 | 9.7% |
| State misinterpretation | 5 | 16.1% |
| State conditioned | 0 | 0.0% |
| Wrong-branch swap | 0 | 0.0% |

The three partial pairs emitted one unchanged word that happened to fit one
branch but not the other. That is not evidence of conditioning because the
action did not respond to the feedback change.

Of the five parsed pairs where Model B changed its action, only one produced an
action consistent with its own branch. In that pair, feedback after `DRIFT`
changed between `YGGBB` and `BGGBB`; the model changed from `TRIPE` to `BRING`.
`BRING` fit its branch and `TRIPE` did not fit the other.

There were no pairs where both changed actions fit their own branches, and no
pairs where both actions fit the opposite branches. The wrong-branch
hypothesis is unsupported. The model is not systematically swapping branch
semantics. It mostly emits an invalid attractor regardless of branch.

## Broad, mixed, and narrow states

Model B under the training prompt produced:

| Pair scope | Parsed pairs | Sensitivity | Either consistent | Both consistent |
| --- | ---: | ---: | ---: | ---: |
| Broad | 11 | 18.2% | 9.1% | 0.0% |
| Mixed | 11 | 27.3% | 27.3% | 0.0% |
| Narrow | 9 | 0.0% | 0.0% | 0.0% |

The strongest response appeared in mixed pairs, where one branch was broad
and the other narrow. Even there, no pair produced two branch-correct actions.

Narrow states were not easier. Model B emitted the same action for every
parsed narrow pair, including the available teacher-branch examples. That
agrees with Lab 15: the three usable fixed-state calls occurred only after
uncertainty collapsed, but they did not reflect a general narrow-state policy.

## Feedback-change type

Model B's training-format response varied by the mark transition:

| Feedback change | Parsed pairs | Sensitivity | Either consistent | Both consistent |
| --- | ---: | ---: | ---: | ---: |
| B/G | 12 | 0.0% | 8.3% | 0.0% |
| B/Y | 12 | 33.3% | 16.7% | 0.0% |
| Y/G | 7 | 14.3% | 14.3% | 0.0% |

The model reacted most often to a gray/yellow distinction and did not change
once for a gray/green distinction. This weakens a simple salience story in
which green feedback always controls output more strongly.

The samples are small, especially the seven Y/G pairs. Treat this pattern as a
Lab 17 diagnostic target, not a settled ranking of feedback difficulty. A
structured representation should make all three transitions explicit, then
reuse these pairs to test whether B/G sensitivity rises without sacrificing
correctness.

## Fixed-state failure census

Constraint failure remained the main fixed-state problem. Under the training
interface:

| Model | Constraint violation | Outside answer lexicon | Invalid format | Repeated history guess | Usable |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 30 | 7 | 5 | 5 | 0 |
| B | 33 | 8 | 2 | 1 | 3 |

These categories assign one primary failure per call. The orthogonal
constraint check found violations in 89.4% of fixed states for both A and B
under the training prompt.

Marginal violation counts show that the failure is not confined to one Wordle
rule:

| Violation | Model A | Model B |
| --- | ---: | ---: |
| Gray or excess letter | 39 | 28 |
| Green not preserved | 25 | 26 |
| Yellow count missing | 7 | 12 |
| Yellow reused in the same position | 15 | 8 |

Dataset B reduced gray/excess-letter and yellow-position violations, but green
preservation did not improve and missing yellow counts increased. A
representation that only lists green positions would address one symptom, not
the full constraint problem.

## Repetition was not fixed

| Model | Lab 15 repeat rate | Parsed-output repeat rate |
| --- | ---: | ---: |
| A | 70.5% | 70.5% |
| B | 11.6% | 71.6% |
| Base | 0.0% | 61.5% |

Lab 15 counted repeats only for answer-list guesses. Lab 16 adds every parsed
output to the diagnostic `seen` set. Model B's apparent repeat improvement
disappeared: its corrected rate is 71.6%, slightly worse than Model A's 70.5%.

The attractors changed, but collapse severity did not:

```text
Model A: BRAIN 76, BAYOU 10
Model B: BRISE 35, PETEL 21, RASHY 20, BETEL 19
Base:    CRAKE 74, CRANE 17
```

This supports the state-insensitivity result. A policy that maps many histories
to four outputs will often remain unchanged under a one-mark perturbation.

## Prompt transfer

Model B was more sensitive under the training prompt than deployment:

```text
training sensitivity:   16.1%
deployment sensitivity: 10.7%
```

Its either-consistent rate also fell from 12.9% to 3.6%. Prompt transfer loses
some learned state response.

This does not make deployment prompting the primary cause. Model B still
produced 0% paired consistency under the exact training representation. Lab 17
must improve how constraints are represented during learning and deployment,
not merely copy the current training wording into gameplay.

## Hypotheses weakened or eliminated

### Model B is already state-sensitive but reads feedback backward

No pair had both outputs consistent with the opposite branches. The
wrong-branch swap rate was zero. There is no evidence of systematic branch
inversion.

### Green changes dominate model attention

Model B changed on 0 of 12 parsed B/G pairs under the training prompt. It
changed on 4 of 12 B/Y pairs. Green salience does not explain this sample.

### Dataset B solved repetition

The corrected parsed-output repeat rate was 71.6%, not 11.6%. Dataset B
diversified the attractor set but did not solve collapse.

### Prompt mismatch explains the failure

The training prompt improved sensitivity, yet 83.9% of parsed pairs remained
unchanged and paired consistency was zero. The failure exists before prompt
transfer.

### Better policy coverage did nothing

Model B was more state-sensitive than Model A under the controlled training
prompt, 16.1% versus 3.2%, and achieved four pairs with one directly consistent
action where A achieved none. Better coverage changed learning. It was not
sufficient for a valid policy.

## Limitations

The experiment has 34 pairs. After excluding unparseable outputs, each
model-interface comparison has 28 to 31 observations.

Thirty of the 34 pairs use controlled off-policy branch guesses. Only two use
the candidate-only teacher action and two use the fixed `RAISE` opening. The
experiment measures response to valid unseen states, but much of it is an
extrapolation test.

Pairs were capped at two per parent and branch guess, leaving 28 distinct
parent/guess groups. The observations are still not fully independent.

The sensitivity estimates do not include confidence intervals. Differences of
one or two pairs should not drive a representation decision.

The base model cannot follow the training-format task prompt, so it only
provides a deployment-interface control.

Consistency uses the frozen 2,315-answer candidate-only rules from Lab 15. Part
III will test the larger legal action space separately.

The feedback-transition groups are small. B/G has 12 parsed Model B training
pairs, B/Y has 12, and Y/G has 7.

## Implications for Lab 17

Lab 17 should test an explicit derived-state representation against the same
frozen adapters or newly controlled adapters. The representation should expose:

* fixed green letters by position;
* present letters with excluded positions;
* minimum and maximum letter counts;
* absent letters only when duplicate-letter evidence permits that conclusion;
* previous guesses as a separate no-repeat set;
* the remaining candidate count.

The primary Lab 17 representation metric should reuse these 34 perturbation
pairs:

```text
both-consistent rate under explicit state
minus
both-consistent rate under raw history
```

The raw-history baseline is 0% for every model and interface. Sensitivity is a
secondary diagnostic. Raising sensitivity without raising paired consistency
would reproduce Lab 16's state-misinterpretation failure.

The strongest next hypothesis is:

> Raw feedback history does not expose Wordle constraints in a form that
> Qwen3-0.6B can reliably bind to its generated action. An explicit derived
> state should increase paired consistency, not merely make outputs change
> more often.

Keep Dataset B frozen until that representation hypothesis is tested.
