# Part II — From Training Techniques to a Working Wordle Model

The first twelve labs focused on learning the mechanics of modern LLM training.

We built the Wordle environment, created datasets, trained models with full supervised fine-tuning and LoRA, explored distillation, designed reward functions, and applied GRPO. Along the way, we learned how each technique changes a model and how to measure those changes.

The resulting models, however, are still poor Wordle players.

That is not a failure of Part I.

The objective of Part I was to learn the training techniques in isolation. We intentionally accepted imperfect gameplay so we could focus on the mechanics of SFT, parameter-efficient fine-tuning, distillation, and reinforcement learning without constantly redesigning the task around the model.

Part II changes the objective.

From this point forward, the training techniques are no longer the subject of the course. They are tools we already know how to use.

Our new goal is:

> **Take the small model we already have and make it genuinely good at Wordle.**

To do that, we will combine the training techniques from Part I with a more traditional data-science workflow.

Instead of immediately training another model, we will study the data and behavior we already have. We will visualize distributions, identify weak areas of the dataset, analyze model failures, formulate hypotheses, and run controlled experiments.

The workflow for Part II becomes:

**Observe → Analyze → Hypothesize → Experiment → Measure → Iterate**

Every major change should answer a specific question.

If we change the training distribution, we should know why.

If we modify the prompt representation, we should know which failure we expect it to address.

If we return to full SFT, distillation, or GRPO, we should have evidence that the technique is appropriate for the remaining problem.

The objective is not simply to obtain a higher score.

The objective is to understand what caused the improvement.

---

## Lab 13 — Exploratory Data Analysis of the Training Corpus

Before generating more data, we need to understand the data we already have.

In this lab, we will treat our Wordle training corpus like any other machine-learning dataset. We will use Pandas and visualization tools to explore its structure and distribution.

We will examine:

* examples by task
* training examples versus training tokens
* `NEXT_GUESS` frequency
* examples by Wordle turn
* candidate-count distributions
* difficulty buckets
* history depth
* target-guess frequencies
* answer coverage
* repeated or near-duplicate states
* training versus validation distributions

We will also cross important dimensions, such as:

`turn × candidate_count`

and:

`task × difficulty`

The goal is to identify whether our training corpus actually reflects the behavior we expect the model to learn.

By the end of the lab, students should be able to make evidence-based statements such as:

> Our model performs poorly on late-game decisions because those states represent only a small fraction of the training data.

or:

> `NEXT_GUESS` is not actually rare, but its target distribution is dominated by a handful of guesses.

This lab produces hypotheses, not a new model.

---

## Lab 14 — Designing a Better Policy Dataset

Lab 13 tells us what the current dataset looks like.

Lab 14 asks what it should look like.

We will redesign the training corpus around actual Wordle gameplay rather than accepting whatever distribution naturally falls out of our data generator.

Possible interventions include:

* increasing the proportion of `NEXT_GUESS`
* balancing examples across turns
* increasing late-game states
* increasing multi-turn histories
* balancing candidate-count ranges
* reducing excessive opening-move duplication
* limiting extremely frequent target guesses
* increasing state diversity
* generating complete teacher trajectories
* extracting multiple training examples from each trajectory

Rather than choosing arbitrary percentages, we will derive the new distribution from the weaknesses discovered in Lab 13.

We will also compare the original and redesigned datasets visually.

The result of this lab will be two clearly defined corpora:

**Dataset A:** the original training distribution

**Dataset B:** the redesigned policy-focused distribution

That creates the basis for a controlled experiment.

---

## Lab 15 — Testing the Data Hypothesis with LoRA

Now we train.

We will use LoRA because it gives us a relatively inexpensive way to test whether our dataset redesign actually matters.

Everything except the dataset should remain as constant as practical:

* same base model
* same LoRA rank
* same target modules
* same optimizer
* same learning rate
* same training schedule
* same evaluation harness

We will compare the original LoRA model against a new model trained on the policy-focused dataset.

Evaluation will include more than aggregate validation accuracy.

We will measure:

* accuracy by task
* `NEXT_GUESS` performance
* valid guess rate
* history consistency
* repeat rate
* candidate-set reduction
* solve rate
* turns required on solved games

This lab tests a specific hypothesis:

> Poor gameplay is caused in meaningful part by the distribution of our training data.

If the new dataset significantly improves behavior, we have evidence that data distribution was an important bottleneck.

If it does not, we have eliminated one possible explanation.

Both results are useful.

---

## Lab 16 — Error Analysis of the Improved Model

Once the new model has been trained, we stop looking only at averages.

We inspect its failures.

We will collect failed games and classify failure modes such as:

* invalid output
* repeated guess
* guess inconsistent with history
* legal but low-information guess
* failure to exploit an obvious late-game solution
* collapse onto a frequent training guess
* failure after duplicate-letter feedback
* failure with long histories

We will group these failures and measure how frequently each occurs.

We may discover, for example, that valid formatting and repetition are largely solved but multi-turn consistency remains poor.

That changes the next intervention.

This lab teaches an important ML engineering principle:

> Once a model becomes partially competent, aggregate metrics become less useful than understanding the remaining failure distribution.

---

## Lab 17 — Improving State and History Representation

The model can only reason over information it can reliably interpret.

In this lab, we investigate whether the way we represent Wordle state is making the task unnecessarily difficult.

We may compare representations such as:

Raw history:

`CRANE -> BYBBG`

versus more explicit derived state:

`GREEN=_R___`

`PRESENT=A`

`ABSENT=CNE`

or structured forms that include candidate count and previous guesses.

We will test whether exposing derived symbolic state dramatically improves behavior.

This becomes an experiment in representation learning.

If a structured representation works much better, then the model may not be failing at policy selection at all. It may be spending too much capacity reconstructing constraints from compact Wordle feedback.

We can later remove some of that scaffolding and measure what the model can infer for itself.

---

## Lab 18 — Learning from Full Game Trajectories

So far, many of our examples have treated states independently.

But Wordle is sequential.

In this lab, we train on complete teacher games.

A single trajectory may look like:

`state_0 → guess_1`

`state_1 → guess_2`

`state_2 → guess_3`

`state_3 → answer`

Each intermediate state becomes a training example while preserving the full history leading to it.

This exposes the model to the actual causal structure of gameplay.

We will compare trajectory-trained models against models trained primarily on isolated states.

Questions include:

* Does history consistency improve?
* Does repetition decrease?
* Does performance improve at turns 3–6?
* Does the model learn to change strategy as the candidate set shrinks?

This lab shifts the problem from static prediction toward sequential policy learning.

---

## Lab 19 — Distillation Revisited

We explored distillation in Part I to understand how teacher knowledge can be transferred.

Now we revisit it with a stronger purpose.

Our symbolic Wordle solver can generate high-quality policies and complete trajectories.

We will use it as a teacher and ask:

> Can a small neural model imitate a genuinely competent Wordle policy?

We will generate teacher decisions across diverse states and train the student to reproduce them.

Evaluation may include:

* teacher/student top-1 agreement
* KL divergence
* history consistency
* policy quality
* actual solve rate

We will also examine places where exact teacher imitation may not be necessary.

If several guesses are strategically equivalent, exact-match accuracy may underestimate policy quality.

This gives us an opportunity to distinguish imitation quality from game quality.

---

## Lab 20 — Full SFT on the Validated Dataset

By this point, we should have much stronger evidence about what data and representation work.

Only now do we return to the more expensive full supervised fine-tuning experiment.

We will train the entire model using the best dataset and representation discovered in Labs 13–19.

Then we compare:

* LoRA
* full SFT
* distilled policy

The question is no longer:

> Is full SFT better than LoRA?

Instead, it becomes:

> Once the training data is good, how much additional capability do we gain by allowing the entire network to adapt?

We will compare performance against:

* training time
* trainable parameter count
* checkpoint size
* solve rate
* consistency
* policy quality

This provides a much more meaningful cost-versus-quality comparison than the original Part I experiment.

---

## Lab 21 — Reinforcement Learning Revisited

In Part I, GRPO attempted to improve a policy that was still fundamentally weak.

Now we revisit RL with a competent supervised starting point.

The reward can focus on genuine gameplay quality:

* valid output
* no repeated guesses
* history consistency
* candidate-set reduction
* efficient solving
* successful completion

Because the base policy should already understand the game, reinforcement learning is no longer being asked to discover Wordle from scratch.

Instead, it can refine behavior.

We will compare the pre-RL and post-RL policies and answer:

* Does solve rate increase?
* Are games solved in fewer turns?
* Does the policy choose more informative guesses?
* Does RL introduce regressions?
* How far does the policy move from the supervised model?
* Is the improvement worth the added complexity?

This is a much more realistic use of post-training reinforcement learning.

---

## Lab 22 — Final Evaluation and Ablation Study

The final lab answers the question that matters most:

> What actually made the model better?

We will run the best model against a held-out Wordle evaluation set and produce a final scorecard.

Metrics may include:

* solve rate
* mean turns on wins
* valid output rate
* history consistency
* repeat rate
* candidate reduction
* runtime
* checkpoint size

But we will also perform ablations.

For example:

* remove policy balancing
* remove structured state
* remove trajectory data
* remove distillation
* remove RL
* compare LoRA and full SFT

Each ablation removes one improvement and measures the impact.

This prevents us from ending the course with a pile of interventions and no idea which ones mattered.

The final result should tell a causal story:

> Better data produced the largest improvement. Structured state solved most consistency failures. Trajectory training improved later turns. Distillation transferred policy quality. Full SFT added a smaller incremental gain. RL improved turn efficiency but was not responsible for basic competence.

The exact conclusions will depend on what the experiments show.

That uncertainty is intentional.

---

## The Goal of Part II

By the end of Part II, we should have done more than create a good Wordle player.

We should have experienced the full lifecycle of applied model development:

**build → train → evaluate → fail → inspect → understand → redesign → retrain → measure**

Part I taught us how the tools work.

Part II teaches us how to decide when and why to use them.

The final model is useful evidence that our process worked.

But the more important result is that we can explain, with data and controlled experiments, how a weak model became a competent one.

---

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

## Establish a reproducible original vocabulary

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

## Separate hidden states from legal actions

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

## Define the teacher objective precisely

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

## Compare teachers before generating data

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

## Decide which Wordle rules evaluation represents

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

## Test whether the stronger teacher is teachable

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

## Gate Dataset C on measured teacher improvement

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

## Run a controlled model experiment

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

## Interpret the result without moving the goalposts

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
