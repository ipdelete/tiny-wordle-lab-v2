# Part III: Expanding the Wordle action space

Part II treats the 2,315 original Wordle answers as both the hidden solution
space and the model's action space.

Those are different domain concepts:

```text
ANSWERS          = possible hidden solutions
ALLOWED_GUESSES  = legal player actions
```

The distinction matters most when many answers remain. A strong Wordle player
may choose a legal word that cannot be the answer because its letters divide
the remaining candidates better. This is an information-gathering probe.

The current symbolic teacher cannot choose such a word. It scores only the
remaining answer candidates as possible actions. Dataset B exposes the model
to more high-uncertainty states, but the labels still come from this restricted
teacher.

This suggests a follow-on hypothesis:

> Policy quality may be limited because the teacher and evaluator treat
> possible answers as the complete action set, excluding legal
> information-gathering guesses and scoring normal-mode probes as failures.

Part III will test that hypothesis before changing any training corpus.

## Keep the Lab 15 experiment frozen

Lab 15 compares Dataset A with Dataset B. Both datasets use the same
candidate-only teacher and the same 2,315-word answer vocabulary. The Lab 15
metric also uses that answer list when checking model outputs.

We will not revise Dataset B or the Lab 15 metric while that experiment is in
progress. Doing so would change the intervention after preregistration.

Lab 15 should continue to describe its check as `in_answer_lexicon`. This is a
known limitation, not an implementation bug. Its result remains useful because
both models are evaluated under the same restricted rules.

Part III begins after the Dataset A versus Dataset B result is recorded.

## Lab 23: Answer space versus action space

This lab corrects the domain model before changing code or data. Students will
distinguish possible hidden answers from legal player actions and identify
where the current implementation treats those sets as interchangeable.

The lab will also separate action legality from policy quality. A normal-mode
probe may be legal and informative even when it cannot be the answer. A repeat
may be legal but strategically useless. Hard-mode constraints are another
property rather than the definition of a valid Wordle action.

The deliverable is a written model of the game state, action space, and
evaluation properties that later labs will implement.

## Lab 24: Build and verify the original Wordle vocabulary

The project should retain two explicit files:

```text
data/wordle-answers-original.txt
data/wordle-guesses-original.txt
```

`wordle-answers-original.txt` contains the 2,315 possible solutions already
used by the course. `wordle-guesses-original.txt` should contain the complete
original legal action set, including all 2,315 answers.

The intended course definition is:

```text
answer space:  2,315 words
action space: 12,972 words
```

We should not attempt to reproduce the current New York Times vocabulary. It
has changed over time and does not provide the stable experimental definition
this course needs. The original source-derived vocabulary is fixed,
reproducible, and matches the historical answer list used in Part I and Part
II.

Available public copies still need verification. They do not all contain the
same words:

* `deedy/wordle-solver` currently has a 12,972-row
  `official_wordle_all.txt`.
* `tabatkins/wordle-list` describes its list as source-derived and is
  MIT-licensed, but its `words` file currently contains 14,855 entries.
* The `cfreshman` lists separate the 2,315 original answers from the additional
  allowed guesses and provide an independent comparison.

We must explain the 1,883-word difference between the 12,972-entry and
14,855-entry lists before choosing a source. Repository popularity is not
evidence that two files represent the same game vocabulary.

Before adding the action list, validate that:

* it contains exactly 12,972 unique lowercase ASCII words;
* every entry has exactly five letters;
* it contains all 2,315 original answers;
* the additional set contains exactly 10,657 words;
* its membership matches a second independent source;
* the source revision is pinned to a commit;
* the imported file has a recorded SHA-256 digest;
* its provenance and redistribution terms are documented.

The Deedy repository is useful for comparison, but GitHub currently reports no
repository license. We should not assume that a public file can be copied into
this project without recording suitable redistribution terms. The Tab Atkins
repository has an MIT license, but its current list does not match the proposed
12,972-word experiment.

The runtime representation should load the complete action set directly:

```python
ANSWERS = load_words("data/wordle-answers-original.txt")
ALLOWED_GUESSES = load_words("data/wordle-guesses-original.txt")

assert len(ANSWERS) == 2_315
assert len(ALLOWED_GUESSES) == 12_972
assert ANSWERS <= ALLOWED_GUESSES
```

Do not store only the 10,657 additional guesses and require every caller to
remember the union.

### Separate hidden states from legal actions

Candidate filtering always operates over possible answers:

```python
candidates = filter_answers(ANSWERS, history)
```

Teacher action selection may operate over either action space:

```python
candidate_only_actions = candidates
full_actions = ALLOWED_GUESSES
```

This gives us two symbolic policies:

```text
Candidate-only teacher
Scores remaining possible answers as guesses.

Full-action teacher
Scores every legal guess against the remaining possible answers.
```

For a state with answer candidates \(C\) and a legal guess \(g\), both teachers
partition \(C\) by the feedback pattern produced by \(g\). The teachers differ
only in which guesses they may score.

The full-action teacher can choose a probe that is not in \(C\). It does not
change the candidate definition. The hidden answer always remains one of the
2,315 answer words.

## Lab 25: Compare candidate-only and full-action teachers

### Define the teacher objective precisely

Maximum feedback entropy and minimum expected candidates remaining are related
but not identical objectives. The experiment should report both, but one must
be declared as the action-selection rule.

For each fixed state, record:

| Metric | Candidate-only teacher | Full-action teacher |
| --- | ---: | ---: |
| Feedback entropy | | |
| Expected candidates remaining | | |
| Worst feedback partition | | |
| Selected guess is a possible answer | | |
| Selected guess is outside the answer set | | |

Use the same tie-breaking rule for both teachers. A difference caused by
unstable ordering is not evidence that the larger action space helps.

Scoring 12,972 guesses against as many as 2,315 candidates is more expensive
than the current search. A cached guess-by-answer feedback matrix can make the
comparison practical. Performance work must not change the scoring semantics.

### Compare teachers before generating data

The first Part III experiment is symbolic. It does not train an LLM.

Compare both teachers on a large, fixed collection of states covering:

* turn number;
* candidate-count buckets;
* history depth;
* early high-uncertainty states;
* later states with long histories and nontrivial candidate sets;
* canonical trajectories;
* controlled off-policy histories.

The state sample should not come only from candidate-teacher trajectories.
That would evaluate the full-action teacher on states selected by its
competitor. Include fixed states generated independently of either policy.

The fixed-state comparison measures local policy quality. It is not enough on
its own. Run complete games against the same answer set with:

* the same opening treatment;
* the same answer sample;
* the same six-turn limit;
* the same tie-breaking rule;
* the same entropy or expected-remainder objective.

Report:

| Metric | Candidate-only teacher | Full-action teacher |
| --- | ---: | ---: |
| Solve rate | | |
| Mean turns on wins | | |
| Mean candidates after each turn | | |
| Mean feedback entropy | | |
| Mean expected candidates remaining | | |
| Worst-case candidates remaining | | |
| Guesses outside the answer set | | |
| Games using at least one probe | | |

A full-action teacher may improve one-step information gain while using an
extra turn to probe. The decision should depend on complete-game behavior, not
entropy alone.

### Decide which Wordle rules evaluation represents

Normal Wordle and hard mode do not define the same legal policy.

In normal mode, any word in `ALLOWED_GUESSES` is a legal action. A probe can be
legal and useful even when the word itself cannot satisfy all previous
feedback.

Hard mode requires the player to reuse revealed hints, although its exact
rules are not identical to simply requiring that every guess remain a possible
answer.

Future evaluation must record separate properties rather than combining them
under one ambiguous `usable` flag:

| Property | Meaning |
| --- | --- |
| `format_valid` | The model produced one parseable five-letter word. |
| `in_allowed_guesses` | The game accepts the action. |
| `in_answers` | The guess could be a hidden solution. |
| `not_repeated` | The guess does not repeat an earlier action. |
| `feedback_consistent` | The word itself satisfies all prior feedback. |
| `candidate_reduction` | The action removes possible answers. |

A full-action probe may be format-valid, legal, non-repeated, and highly
informative while both `in_answers` and `feedback_consistent` are false.

For normal-mode evaluation, legal action rate should require
`in_allowed_guesses`, not `in_answers`. Feedback consistency remains a useful
behavioral diagnostic, but it must not automatically invalidate a legal probe.

Repeated guesses are also legal in the game, but strategically wasteful. Keep
legality and policy quality as separate measurements.

## Lab 26: Analyze teacher policy and target learnability

A stronger symbolic policy may be a worse curriculum for a 0.6B language
model. Expanding the action space creates several risks:

* the output vocabulary grows from 2,315 to 12,972 words;
* high-value probes may be obscure words the model rarely saw in pretraining;
* policy targets may become more diffuse;
* exact teacher imitation may become harder;
* tokenizer fragmentation may make some actions difficult to generate;
* target concentration may move to a new set of frequent probe words.

If the symbolic experiment shows a material gain, inspect the proposed teacher
targets before regenerating a corpus. Measure:

* the fraction of targets outside `ANSWERS`;
* target frequency and top-k concentration;
* tokenizer length for answer and probe targets;
* repeated use of a small probe vocabulary;
* policy changes by candidate-count bucket;
* the number of existing Dataset B states whose label changes;
* expected candidate reduction gained by each changed label.

This analysis answers a second question:

> Does the larger teacher action space produce a policy that this model can
> plausibly learn?

## Lab 27: Build Dataset C

Do not regenerate Dataset B in place. It is a frozen Part II artifact.

Create Dataset C only if the full-action teacher produces a meaningful
complete-game improvement and the target analysis does not reveal an obviously
unlearnable policy.

Dataset C should preserve Dataset B's successful design choices:

* policy examples remain the majority of optimization signal;
* auxiliary expansion remains capped;
* high-uncertainty states remain represented;
* controlled openings create missing strategic states;
* branch-aware splitting prevents trajectory leakage;
* reserved gameplay answers remain excluded from training paths;
* every row retains provenance.

The intervention changes the policy action space and teacher labels, not the
state-coverage experiment established in Lab 14.

Dataset C provenance should add:

```text
teacher_action_space
teacher_objective
teacher_guess_in_answers
teacher_guess_in_allowed_guesses
allowed_guess_list_sha256
```

This makes later ablation possible without reconstructing how each target was
created.

## Lab 28: Test the expanded action space with LoRA

If Dataset C passes the design checks, compare it with the best frozen Part II
dataset under the same controls used in Lab 15:

* same base checkpoint;
* same LoRA configuration;
* same optimizer;
* same scheduler;
* same random seed;
* substantially matched optimizer steps and input-token exposure;
* same fixed evaluation states;
* same fixed-opening game answers;
* same generation settings.

The primary model metric should use a fixed denominator and should count all
legal actions, including non-answer probes. Report results by candidate-count
bucket so the experiment can test its strongest prediction:

> Expanding the teacher action space should help most on broad-candidate
> states where information-gathering probes have room to outperform plausible
> answers.

Secondary metrics should include:

* fixed-opening solve rate;
* legal action rate;
* repeat rate;
* candidate reduction;
* probe frequency;
* feedback consistency as a diagnostic;
* output concentration;
* exact teacher agreement;
* performance on narrow states where guessing a possible answer may be better
  than probing.

Evaluate the training prompt and deployment prompt separately. A policy learned
under the training representation may still fail to transfer through the
deployment interface.

## Lab 29: Failure analysis and final integration

Record the expected interpretations before training:

| Result | Interpretation |
| --- | --- |
| Full-action teacher barely improves symbolic play | Keep the candidate-only teacher. The larger action space does not justify added complexity. |
| Teacher improves, model does not | The policy is stronger but not learnable by this model or representation. |
| Offline policy improves, gameplay does not | The remaining problem is sequential deployment, prompt transfer, or compounding error. |
| Broad-state play improves without narrow-state regression | The action-space hypothesis is supported. |
| Probe targets dominate model outputs | The expanded curriculum introduced a new policy-collapse mode. |
| Solve rate improves but mean turns worsens | Probes improve reliability at the cost of efficiency. Decide which objective the course values. |
| Both teacher and model improve | Retain the full action space and test it in later trajectory, distillation, full-SFT, or RL work. |

Part III should not assume that a larger vocabulary is better. It asks whether
the restricted action space is a real bottleneck, whether removing that
restriction improves the symbolic policy, and whether the small model can
learn the improved policy without creating a new failure mode.
