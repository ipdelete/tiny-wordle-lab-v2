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

Part III continues this work by testing whether the project defined Wordle's
action space too narrowly. See [Part III: Expanding the Wordle action
space](part3-prd.md).
