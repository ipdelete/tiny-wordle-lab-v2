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

If it does not, the intervention was not sufficient. That does not prove that
data distribution was irrelevant or that better coverage is unnecessary.

Both results are useful.

---

## Lab 16 — Error Analysis of the Policy Models

Once the new model has been trained, we stop looking only at averages.

We inspect its failures.

Lab 15 showed that better policy coverage changed the output distribution but
did not produce usable broad-candidate play. Model B reached 6.4% usable under
the training prompt and 2.1% under the deployment prompt. Prompt transfer
matters, but the model already fails under its training interface.

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

The central question is whether the model ignores the state or reads it
incorrectly:

* **state insensitivity:** the generated action rarely changes when a valid
  feedback branch changes;
* **state misinterpretation:** the action changes, but the new guess violates
  the constraints encoded by the changed history.

We will construct paired, reachable Wordle states that differ by one valid
feedback branch. Arbitrary feedback edits are not suitable because they can
create impossible histories.

For these pairs, report:

```text
state perturbation sensitivity =
changed generated actions / valid paired state perturbations
```

Cross sensitivity with history consistency. An unchanged guess can remain
valid in both states, so action stability alone is not a failure. A changed but
inconsistent guess shows that the model responds to state while interpreting
it incorrectly.

We may discover, for example, that valid formatting is largely solved while
multi-turn consistency remains poor.

That changes the next intervention.

State insensitivity supports making the state representation more explicit in
Lab 17. State misinterpretation supports targeted constraint-learning examples
before changing the interface.

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

## Lab 19d — Diagnose full-list rank drift

Lab 19 showed that a model can improve its twelve-action distillation objective
while becoming worse over the deployed 2,315-answer action space. Before
training another model, this lab uses the persisted score matrices and
checkpoints to locate that failure.

On paired Turn 2 states, compare each trained arm with its incumbent:

* where the trained top action ranked under the incumbent;
* whether newly dominant actions came from below the mined top 32;
* how candidate, candidate-teacher, and open-teacher ranks moved;
* whether good actions lost score, unsupported actions gained score, or both;
* how the sole candidate's rank changed on visited singleton states;
* whether newly dominant bad actions appeared in that seed's training support.

The main question is:

> Did frozen hard negatives become stale as the policy changed, or did
> continued training cause a broader loss of the incumbent ranking?

This is a no-training diagnostic. Its result determines how Lab 20 constructs
corrective data.

---

## Lab 20 — Correct policy-created states

Static expert trajectories do not cover every state the deployed policy creates.
Lab 19 also showed that a frozen pre-training view of the policy's mistakes can
become obsolete during continued training.

This lab asks:

> Does expert correction on states reached by the policy improve full-game
> behavior more than the same amount of additional static expert data?

This is supervised learning, not RL. A symbolic teacher labels the states after
collection. The changed variable is where those states came from.

### Freeze the rollout contract

Record the policy checkpoint, observation schema, prompt version, answer
vocabulary, decoder, action parser, transition rules, answer split, and trace
format. Preserve every policy-created state before querying the teacher.

### Compare matched additional data

Start all arms from the same frozen incumbent:

| Arm | New labeled states |
| --- | --- |
| `rollout_correction` | States reached during fixed model-driven games, relabeled by the symbolic teacher |
| `static_random` | Expert-generated states sampled without reference to policy rollouts |
| `static_matched` | Expert-generated states matched by answer branch, turn, and candidate-count stratum |

Persist each eligible unique rollout state before querying the teacher, query
it once, and retain its `visit_count`. Cap that count at a preregistered value,
then expand the correction corpus into formatted training presentations.
`static_matched` receives the same capped weight as its corresponding rollout
state. `static_random` samples the same total presentation count with
replacement from a manifested development-only pool.

All arms use the same number of formatted training presentations, padded-token
budget, optimizer updates, schedule, and held-out evaluation. Preserve unique
state counts, visit weights, teacher disagreement, collection cost, and overlap
with the base corpus.

Train three seed-matched arm triplets. The primary estimate is the equal-weight
mean of the three seed-paired held-out solve-rate differences:
`rollout_correction - static_random`.

The correction gate passes only when:

* the mean paired solve-rate gain is at least five percentage points;
* rollout correction improves all three seed pairs;
* the pooled answer-level paired bootstrap 95% interval excludes zero.

The bootstrap resamples held-out answer IDs while retaining all three arm
outcomes and all three seed pairs for each sampled answer. It never resamples
arms independently.

`static_matched` explains whether a gain comes from coarse state difficulty or
from policy-specific state differences beyond branch, turn, and candidate
count.

If the gate fails, preserve the traces and diagnose the result. Do not iterate
the correction loop or treat it as an RL baseline without new evidence.

---

## Lab 21 — Part II checkpoint and handoff

Part II ends by answering:

> Which supervised choices produced deployed capability, and which only
> improved an offline objective?

Run the frozen models against one held-out scorecard covering:

* answer-constrained solve rate and turns on wins;
* free-generation validity;
* history consistency and repeats;
* Turn 2 action value and candidate reduction;
* singleton closure;
* runtime, trainable parameters, and checkpoint size.

Then record the supported ablations. Compare static versus policy-created data,
compact versus structured state, isolated versus trajectory examples, and the
incumbent against the failed distillation arms. Do not describe a failed
intervention as a component of the best model.

The handoff freezes:

* the retained Part II checkpoint and seed;
* the retained corpus and representation;
* the answer-only action vocabulary;
* the Lab 20 correction result and traces;
* the unresolved rank-drift and interface failures.

Part III begins from that record. Full SFT waits until Part III has tested the
action vocabulary and selected the final supervised recipe. Completion-level
RL does not return here; Part IV studies policy learning through complete
environment trajectories.

---

## The Goal of Part II

By the end of Part II, we should have done more than create a good Wordle player.

We should have experienced the full lifecycle of applied model development:

**build → train → evaluate → fail → inspect → understand → redesign → retrain → measure**

Part I taught us how the tools work.

Part II teaches us how to decide when and why to use them.

The final model is useful evidence that our process worked.

But the more important result is that we can explain, with data and controlled experiments, how a weak model became a competent one.

Part III continues this work in Lab 22 by testing whether the project defined
Wordle's action space too narrowly. See
[Part III: Expanding the Wordle action space](part3-prd.md).
