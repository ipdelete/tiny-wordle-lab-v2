# Marin research methodology

Reviewed repository: [`marin-community/marin`](https://github.com/marin-community/marin) at commit
[`3f2976e5311c26f540bbc6e61d68a20200cd86f5`](https://github.com/marin-community/marin/tree/3f2976e5311c26f540bbc6e61d68a20200cd86f5)
(commit date 2026-08-28). Review date 2026-08-28. Every Marin repository link below is pinned
to that commit. The Karpathy comparison uses
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) at
[`228791fb499afffb54b46200aca536f79142f117`](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117),
which was the repository head at review time.

This report describes how Marin does research. It does not describe how to install
or operate Marin. Claims are marked:

- **FACT** for something stated or implemented in a cited source.
- **INTERPRETATION** for a reading of that evidence.
- **RECOMMENDATION** for advice to this Wordle repository.

## Practical summary

Marin uses a formal process because it coordinates large datasets, shared
infrastructure, expensive training runs, and many contributors. Its useful
lesson for this repository is experimental discipline, not distributed
orchestration.

### Research loop

The documented Marin lifecycle is:

1. Preregister the hypothesis, expected outcome, scale, changes, and metrics in
   a GitHub issue.
2. Implement the data, training, evaluation, and reporting work as an explicit
   dependency graph.
3. Put a small sanity run and a dry run of the full plan in a pull request.
4. Review the code, evidence, and expected cost before the expensive run.
5. Run the approved experiment.
6. Return the results and interpretation to the issue.
7. Retain negative results in the research record.

Its autonomous research skill uses a related loop:

```text
Forage -> Propose -> Run -> Interpret -> Promote -> Seal
```

A persistent logbook records the hypothesis queue, blocked ideas, falsified
hypotheses, promoted results, exact commands, and milestone commits.

### Cheap evidence protects expensive runs

Marin moves through quick comparisons, reference-scale tuning, compute sweeps,
larger scaling ladders, and cheap canaries before a full run. Promotion rules
are written before results arrive.

The Wordle equivalent is:

```text
tiny answer canary
small validation battery
repeated validation
untouched final holdout
```

A broken prompt, reward, adapter, or evaluator should fail before it consumes a
long GPT-OSS-20B run.

### Controlled experiments and hero runs

Marin runs two kinds of research. Small experiments are controlled,
preregistered, and gated. Large runs may change while training because
restarting them is prohibitively expensive. Marin calls the latter the
Tootsie Roll process.

Their retrospectives acknowledge that mid-run changes weaken attribution.
Wordle experiments are restartable, so this repository should restart after a
recipe change rather than adopt the Tootsie Roll process.

### Data belongs inside the experiment

Marin separates dataset identity from experiment policy. Catalog entries,
processed artifacts, and mixture weights are different concerns. An experiment
chooses a mixture without mutating the underlying datasets.

For Wordle:

- canonical answers and legal guesses remain fixed;
- subsets and difficulty buckets belong to experiment configuration;
- SFT sampling weights belong to the training recipe;
- a changed mixture becomes a new experiment.

### Metric disagreement is evidence

Marin has found failures by comparing metrics that should have moved together.
Training loss changed while validation loss stayed flat in a shuffle failure.
A data mixture improved validation loss while downstream tasks became worse.

Equivalent Wordle warning signs include:

- SFT loss falls while solve rate stays flat;
- legal-action rate rises while clue consistency falls;
- training reward rises while candidate probability mass falls;
- deterministic play improves while stochastic play regresses;
- aggregate results improve while repeated-letter results decline.

These disagreements should trigger diagnosis rather than be averaged into one
score.

### What to borrow

- Write the hypothesis and expected direction before each run.
- Freeze promotion rules before seeing results.
- Run cheap canaries before large evaluations or training jobs.
- Keep negative results in an experiment ledger.
- Record commands, model revisions, prompt hashes, data hashes, and seeds.
- Separate canonical data from sampling and mixture policy.
- Promote only after staged gates pass.
- Extract proven research code into the core package after the experiment.

### What not to borrow

- distributed artifact executors;
- cluster schedulers and distributed locks;
- multi-region storage machinery;
- W&B project hierarchies;
- scaling-law machinery without a model-size axis;
- multi-terabyte deduplication infrastructure;
- stale cache reuse after an identity mismatch;
- mid-flight changes to restartable experiments.

This repository is closer to autoresearch in scale. Marin contributes a
stronger method for hypotheses, staged evidence, provenance, and promotion.

## Mission and research philosophy

**FACT.** Marin describes itself as "a research program, software platform, and community
for the research and development of foundation models" and commits to "openly sharing *all*
of the process knowledge required to build these models"
([`README.md#L15-L17`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md#L15-L17)).

**FACT.** The stated core value is open development: "We document our processes, experiments,
and decisions as they happen. Every step, from raw data to the final model, is recorded.
Failed experiments are part of that record."
([`README.md#L19`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md#L19)).

**FACT.** Marin's own model system prompt defines the project as "an open lab for building
foundation models collaboratively." It spells out the mechanism: share code, data, experiments,
and documentation in real time; preserve issues, pull requests, execution traces, and W&B
reports; and let contributors work on architectures, algorithms, datasets, or evaluations
([`docs/system-prompts/05-18-2025.md#L25-L30`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/system-prompts/05-18-2025.md#L25-L30)).

**FACT.** The pipeline in scope is spelled out as nine stages, from curating raw sources
through crawling, text extraction, quality classifiers, filtering, deduplication,
tokenization, training, and evaluation
([`docs/explanations/lm-pipeline.md#L7-L15`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lm-pipeline.md#L7-L15)).
Marin composes existing tools rather than replacing them: trafilatura and resiliparse for
HTML, fastText for filtering, Levanter for training, lm-evaluation-harness for evaluation
([`lm-pipeline.md#L17-L25`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lm-pipeline.md#L17-L25)).

**FACT.** The current program is a frontier mixture-of-experts pretraining effort at roughly
5e24 model FLOPs, alongside Delphi, an open scaling suite spanning 3e18 to 1e23 FLOPs whose
released artifacts include checkpoints, deterministic mixture pipelines, forkable recipe code,
a development methodology published as an agent skill, and plot-ready figure data
([`README.md#L25-L42`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md#L25-L42)).

**INTERPRETATION.** "Open lab" here means something narrower and more demanding than open
weights. Marin publishes the decision trail, not only the artifact. The 32B retrospective
makes this concrete by documenting a self-inflicted evaluation contamination and a bad data
shuffle in the same document that reports the benchmark wins
([`docs/reports/marin-32b-retro.md#L333-L371`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L333-L371)).

## The experiment model in code

**FACT.** An experiment has two representations. Conceptually it is "a unit of inquiry with a
particular hypothesis or goal," captured by a GitHub issue tagged `experiments`. Structurally
it is a DAG of steps, recorded as one file per experiment under `experiments/`, named after
the issue number
([`docs/explanations/experiments.md#L1-L19`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/experiments.md#L1-L19)).

**FACT.** The execution model is lazy artifacts. An `ArtifactStep[T]` is a frozen handle whose
identity is `name` plus `version`; constructing one runs nothing. Storage address is an explicit
`{prefix}/{name}/{version}` path with no content hash
([`docs/explanations/lazy-artifacts.md#L1-L39`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L1-L39)).

**FACT.** The identity boundary is explicit. Values written as literals in `build_config`, such
as model architecture and hyperparameters, enter the artifact fingerprint. Values pulled from
the `StepContext`, such as output paths, storage prefix, region, and compute resources, are
execution choices and are excluded
([`lazy-artifacts.md#L41-L98`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L41-L98)).
Changing the TPU therefore never re-fingerprints a checkpoint
([`lazy-artifacts.md#L96-L98`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L96-L98)).

**FACT.** Cache invalidation is advisory, not automatic. When the runner finds an existing
artifact whose recorded fingerprint differs from the current one, it logs a warning and serves
the cached output. Producing a new result requires the author to bump `version`; a `dev`
version string opts out of caching entirely
([`lazy-artifacts.md#L131-L149`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L131-L149)).
This is a deliberate reversal of the previous content-addressed executor, in which a single
hyperparameter change silently re-addressed every downstream step and paths carried no
human-readable version
([`lazy-artifacts.md#L205-L214`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L205-L214)).

**FACT.** `StepRunner.run` applies a cache check and a distributed lock before executing each
step
([`lazy-artifacts.md#L119-L123`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L119-L123)).
The implementation handles caching, locking, heartbeats, and status writes explicitly rather than
through decorators, "so the control flow is easy to follow and debug"
([`lib/marin/src/marin/execution/step_runner.py#L4-L11`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_runner.py#L4-L11)).

**FACT.** Execution resources do not change step identity. A lowered `StepSpec` carries an
optional `ResourceConfig`. The runner sends a step with explicit resources, or a
`RemoteCallable`, to Fray; otherwise it runs the callable in the runner thread
([`step_spec.py#L67-L82`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_spec.py#L67-L82)).
The runner keeps the lock and status file while the remote job executes, and writes the
artifact result inside that job
([`step_runner.py#L430-L469`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_runner.py#L430-L469)).

**FACT.** `ArtifactStep.adopt` brings pre-existing data into the graph without recomputation and
still writes a provenance record at the canonical path, so the drift check governs the alias
([`lazy-artifacts.md#L151-L177`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L151-L177)).
The experiment guidelines instruct authors to use `adopt` after a full experiment runs, to keep
heavy artifacts visible in the dependency graph while preventing accidental re-execution
([`docs/explanations/guidelines.md#L126-L128`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L126-L128)).

**FACT.** Launch is separated from definition. Experiment drivers print the lowered plan by
default and require an explicit `--run` to execute, and `--version` has no silent default
because a deferred dataset at a mutable version would rebuild a multi-terabyte cache
([`lib/marin/src/marin/experiment/cli.py#L4-L20`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/cli.py#L4-L20)).

**INTERPRETATION.** The reproducibility consequence is that Marin trades automatic correctness
for legibility and cost control. A stale cache can be served after a warning, which is a real
hazard. The 32B GSM8k contamination shows the broader danger of trusting a cached dataset after
preprocessing has changed
([`marin-32b-retro.md#L335-L348`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L335-L348)).
In exchange, storage paths are readable, reruns are cheap, and a version bump is an auditable
human decision rather than an invisible side effect of editing a config.

## The scientific loop actually practiced

Marin's infrastructure supports many patterns. The evidence below separates what the code
can do from what the record shows was done.

**FACT.** The documented experiment lifecycle is preregistration followed by staged review.
An issue states the change set, the regime such as "1.4B parameter models for 28B tokens," the
evaluation metrics, and "a hypothesis or goal, which is an a priori prediction of what the
outcomes will be," described explicitly as preregistration
([`guidelines.md#L77-L92`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L77-L92)).
A PR then runs a small-scale sanity check plus a dry run of the full experiment "so that one can
review the full experiment and review the estimated cost before running it." Review covers both
the code and the sanity-check output. Only an approved PR gets the full run, and results and
analyses go back onto the issue
([`guidelines.md#L66-L76`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L66-L76)).

**FACT.** The published index records outcomes as one-line conclusions attached to issues, and
the negative results are kept. Examples include "Llama3 tokenizer is the best" and, for MuP,
"not worth it compared to our heuristic version"
([`docs/reports/index.md#L63-L107`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/index.md#L63-L107)).
Data ablations are recorded the same way, including "No major improvement compared to control"
([`index.md#L112-L120`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/index.md#L112-L120)).

**FACT.** The 74-line `exp1078_reproduce_dclm_7b1x.py` is representative: it names the paper it
replicates, cites the upstream architecture config, derives the step count, assembles train and
validation data, and calls `train_lm`
([`experiments/tutorials/exp1078_reproduce_dclm_7b1x.py#L1-L74`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/tutorials/exp1078_reproduce_dclm_7b1x.py#L1-L74)).

**FACT.** The most disciplined comparison protocol in the repository is the Agent MoE gate
system. A variant is compared against a fixed baseline using two metrics pulled from W&B,
`eval/paloma/macro_loss` and `throughput/tokens_per_second` averaged over the last 100 steps,
with a requirement that `run.state` be `finished` before final metrics are read
([`experiments/grug/moe/agent.md#L16-L31`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L16-L31)).
Gate 1 requires an effective speedup at two small scales, d512 at 3.82e17 FLOPs and d768 at
2.81e18. Gate 2 adds d1024 at 1.16e19 and d1280 at 3.46e19, requires speedup at all four, then
refits a scaling law with the asymptote pinned at 1.6 and requires the projection to beat fixed
baseline losses at 1e21 and 1e23 FLOPs
([`agent.md#L33-L51`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L33-L51)).

**FACT.** Promotion is a written rule with an escape hatch. Passing both gates is normally
sufficient; low curvature around each isoflop minimum and stability improvements can also
support promotion "even if loss is neutral at small scale"; and discretionary factors such as
memory footprint, inference latency, KV-cache size, and serving compatibility may influence the
decision even when loss criteria are met
([`experiments/grug/moe/README.md#L150-L178`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L150-L178)).

**FACT.** The Agent MoE digest reports 80 sub-issues, including 32 classified "Did not work."
It separates outcome from issue state because open issues can already contain useful results
([`docs/reports/agent-moe-experiments.md#L25-L39`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/agent-moe-experiments.md#L25-L39)).
One row records a measured MHA win that was declined to keep GQA's fourfold smaller KV cache
([`agent-moe-experiments.md#L61`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/agent-moe-experiments.md#L61)).

**FACT.** The large runs use a different loop, named the Tootsie Roll process: "we didn't fully
know the best recipe, so we just started training with what we had, and planned to adapt along
the way"
([`docs/reports/marin-8b-retro.md#L40-L48`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md#L40-L48)),
restated for 32B as "start training, instrument heavily, and make evidence-driven changes
mid-flight"
([`marin-32b-retro.md#L26`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L26)).
The 32B run is reported as four phases with step ranges, token counts, and one-line change
descriptions, with a footnote excluding discarded diagnostic bursts from cumulative totals
([`marin-32b-retro.md#L37-L63`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L37-L63)).

**INTERPRETATION.** Marin runs two loops at once. Small-scale screening is preregistered,
gated, and quantitative. The hero run is opportunistic and diagnostic, closer to operating a
system under observation than to running a controlled trial. The retrospectives are honest that
the second loop is a compromise: "While we aspire to a clean run, we anticipate that our first
release at each scale will often involve changes mid-run"
([`marin-32b-retro.md#L514`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L514)).
The gates exist because the hero run cannot be a controlled experiment, so control is pushed
down to the cheap scales.

## Data methodology

**FACT.** Dataset catalogs and mixture weights are separated by policy. Each dataset family
lives in one module under `experiments/datasets/` and exposes handles only; "Mixture weights are
policy, not catalog: keep them in a separate `<NAME>_MIXTURE_WEIGHTS` constant keyed the same
way, and let the experiment pick weights"
([`experiments/AGENTS.md#L5-L26`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md#L5-L26)).
The same file forbids changing a handle's `name`, `version`, `pin`, or path strings, because
"those are artifact identities"
([`experiments/AGENTS.md#L22-L23`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md#L22-L23)).

**FACT.** The Nemotron catalog shows provenance handled by pinning. Seven quality splits map to
globs over one pinned raw download, and each llama3-tokenized split is pinned to an existing
cache so referencing it "never re-tokenizes the multi-TiB corpus." A comment preserves an
upstream path typo verbatim because it is the real location
([`experiments/datasets/nemotron.py#L1-L61`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datasets/nemotron.py#L1-L61)).

**FACT.** The Datakit reference pipeline is an explicit DAG from normalized sources to a
per-cluster, per-quality store, with per-source stages for tokenize, embed, domain assignment,
quality scoring, decontamination, and MinHash, plus combining stages for global exact dedup,
fuzzy candidate search, full-text verification, the decontamination filter, and the store
([`experiments/datakit/README.md#L1-L90`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/README.md#L1-L90)).

**FACT.** Deduplication is two-stage and conservative. Exact dedup selects a canonical record
without copying text. Fuzzy dedup first finds candidate clusters, then marks a duplicate only
after a direct comparison with normalized full text
([`datakit/README.md#L39-L57`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/README.md#L39-L57)).

**FACT.** Identity is scoped so that changing the source set does not invalidate unrelated work:
"A source-set change gives a new global exact-dedup output and a new store identity. It does not
change the identity of tokenization, embedding, quality, decontamination, or MinHash steps"
([`datakit/README.md#L64-L66`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/README.md#L64-L66)).

**FACT.** Named constants separate source-local boilerplate detection from cross-source
decontamination based on document frequency and recurrence across at least three sources
([`experiments/datakit/decontam/config.py#L1-L16`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/decontam/config.py#L1-L16)).

**FACT.** Mixture comparison is done with microannealing: take a mostly trained model, run a
short cooldown with roughly 70 percent original mix and 30 percent a candidate high-quality
source, against a 100 percent pretraining-mix control. The reported result is a metric
divergence: oversampling high-quality data improved loss on high-quality validation sets but
degraded task performance, and nothing beat the control until FLAN was mixed in, with the best
setting at 70 percent pretraining, 15 percent FLAN, 15 percent high quality
([`marin-8b-retro.md#L275-L303`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md#L275-L303)).

**FACT.** Data-ordering quality is treated as a first-class variable. A training-loss phase shift
at roughly 190k steps, with validation loss flat, was traced to an affine or LCG index
permutation producing correlated batches; switching to a Feistel permutation removed the phase
shift and improved Paloma validation losses
([`marin-32b-retro.md#L353-L369`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L353-L369),
[`#L417-L438`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L417-L438)).

**INTERPRETATION.** The reasoning that identified the shuffle bug is transferable at any scale:
train loss moved while eval loss did not, which localizes the cause to the data stream rather
than the model. That is a diagnostic rule about which metric pair disagrees, and it costs
nothing to apply.

## Training methodology

**FACT.** `train_lm` forces the identity-bearing decisions to be explicit. Model, optimizer,
dataset weights, batch size, sequence length, `z_loss_weight`, `evals`, and `resources` are all
required keyword arguments and "the helper defaults none of them," with `evals=None` as an
explicit opt-out because "there is no implicit default suite"
([`lib/marin/src/marin/experiment/train.py#L101-L134`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/train.py#L101-L134)).

**FACT.** Staged runs are expressed as dependencies. `init_from` chains a run onto another
checkpoint, making it a dependency and seeding the initialization path
([`train.py#L140-L142`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/train.py#L140-L142)).
A mutable `dev` version namespaces the checkpoint per user so concurrent authors do not clobber
each other, while a calendar version keeps the shared name
([`train.py#L151-L157`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/train.py#L151-L157)).

**FACT.** Marin distinguishes a scaling law from a scaling heuristic: "A scaling law tells you
**what** to train for a compute budget. A scaling heuristic tells you **how** to train each
candidate," and a new heuristic is needed for a new optimizer, training method, or architecture,
while dataset changes can usually reuse an existing one
([`docs/recipes/add_scaling_heuristic.md#L1-L16`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L1-L16)).

**FACT.** The heuristic workflow is six ordered steps: signs of life via a quick A/B at about
130M parameters, tune reference hyperparameters at a fixed reference point of roughly 130M
parameters and 2.5B tokens, define scaling rules as a frozen dataclass, run an IsoFLOP sweep,
train a scaling ladder at larger budgets, then promote to a daily canary
([`add_scaling_heuristic.md#L18-L27`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L18-L27),
[`#L49-L181`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L49-L181)).

**FACT.** Hyperparameter transfer uses formulas anchored at a reference point, including
learning-rate scaling and a beta2 rule that holds token half-life constant
([`add_scaling_heuristic.md#L109-L116`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L109-L116)).
The MoE recipe records a fitted learning-rate formula, the 17-cell sweep that produced it, and
R squared of 0.996
([`experiments/grug/moe/README.md#L54-L90`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L54-L90)).

**FACT.** Post-sweep checks require stability at every size, no regression against the old
heuristic, and an optimum that does not sit at a grid boundary
([`add_scaling_heuristic.md#L158-L161`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L158-L161)).
The iteration rule is to reproduce and fix a problem at the smallest scale before scaling up
([`add_scaling_heuristic.md#L185-L190`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L185-L190)).

**FACT.** Resource accounting appears as budgets and constraints rather than a dollar meter.
The heuristic protocol requires `estimate_memory_bytes()`
([`add_scaling_heuristic.md#L76-L80`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L76-L80)),
the canary is deliberately held at roughly 30M parameters and 1B tokens "so it stays cheap"
([`add_scaling_heuristic.md#L180-L181`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L180-L181)),
cross-region copies share a process-global 10 GB `TransferBudget` that raises when exhausted
and require explicit human permission beyond that
([`experiments/AGENTS.md#L35-L47`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md#L35-L47)),
and the root guidelines state plainly that "storage and bandwidth are major cost drivers for
this project"
([`AGENTS.md#L93-L95`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/AGENTS.md#L93-L95)).

## Evaluation methodology

**FACT.** Three evaluation paths exist: in-loop training evals through Levanter's
lm-evaluation-harness integration logged to W&B, post-hoc evals over an OpenAI-compatible served
endpoint using an Evalchemy fork, and containerized agent benchmarks through Harbor
([`docs/explanations/evaluation.md#L12-L20`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/evaluation.md#L12-L20)).

**FACT.** Evaluations are cached artifacts, not ad-hoc scripts. One `EvalGroup` becomes one
`EvalchemyResult` addressed by `evaluation/evalchemy/{model}/{group_id}`, "so a pipeline picks up
exactly the evals it needs and each is cached and reused," and the in-loop suite and post-hoc
groups draw from the same task menu
([`evaluation.md#L42-L50`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/evaluation.md#L42-L50)).
`eval_steps` builds one lazy result per group and `eval_report` aggregates typed results into one
report artifact
([`lib/marin/src/marin/experiment/evaluation.py#L200-L240`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/evaluation.py#L200-L240)).

**FACT.** Task sets are named and versioned in code, with shot counts and aliases fixed per task,
for example `agieval_lsat_ar` at 3-shot, `arc_easy` at 10-shot, and `hellaswag` registered twice
at 0-shot and 10-shot under distinct aliases
([`experiments/evals/task_configs.py#L11-L27`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/evals/task_configs.py#L11-L27)).
`CORE_TASKS` is the default and `CORE_TASKS_PLUS_MMLU` extends it
([`evaluation.md#L54-L58`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/evaluation.md#L54-L58)).

**FACT.** In-loop metrics go beyond accuracy: bits per byte, raw log probability, choice log
probability, and length-normalized choice probability, each with its formula
([`evaluation.md#L60-L68`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/evaluation.md#L60-L68)).
Intermediate evaluation is scheduled in the experiment definition, for example
`EvalSuite(CORE_TASKS, every=10000)`
([`exp1078_reproduce_dclm_7b1x.py#L65`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/tutorials/exp1078_reproduce_dclm_7b1x.py#L65)).

**FACT.** Result comparison is published with the harness named, the aggregation rule stated
("Average" is a simple mean over shown tasks), per-task numbers for six models, and both mean
rank and mean reciprocal rank as secondary summaries, plus four explicit caveats about prompt
style, base-model status, modality, and task selection
([`marin-32b-retro.md#L440-L499`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L440-L499)).

**FACT.** Contamination was detected through prompt fragility rather than an inflated score.
Under the default harness prompt the model was about 22 points worse than the weakest baseline on
GSM8k, while OLMes-style prompts looked reasonable. The cause was a cached Dolmino math bundle
containing GSM8k test items in `test.json`, formatted in OLMes style, which raised surprisal on
the original prompt's structured tags. The report also notes MATH was poor with no reason to
believe it was contaminated
([`marin-32b-retro.md#L335-L351`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L335-L351)).

**INTERPRETATION.** The contamination episode is the strongest methodological lesson in the
corpus, and it inverts the usual expectation. Training on the test set made the score worse,
because the contaminated format did not match the evaluation format. A score that is anomalous
in either direction, and unstable across prompt formats, is evidence about the data pipeline.

## Tracking, failure handling, and observability

**FACT.** Naming is a convention with teeth. Experiment files are
`experiments/exp${GITHUB_ISSUE_NUMBER}_${DESCRIPTOR}.py`, take no arguments beyond executor
flags, and running one "should launch all the relevant jobs for this experiment from start to
finish." Artifact variables are named for what they produce, and full GCS paths are discouraged
in favor of importing the producing handle so the dependency structure stays explicit
([`guidelines.md#L103-L131`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L103-L131)).

**FACT.** The layer split for records is stated: GitHub is the narrative layer and W&B is the
data and report layer. Runs requiring explicit run-to-run comparison must share a W&B project,
scope must be decided early because runs cannot reliably be moved later, the same experiment ID
must appear in W&B run names, logbook entries, and issue comments, and claims in GitHub must
match the final W&B values
([`.agents/skills/wandb-reporting/SKILL.md#L8-L45`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/wandb-reporting/SKILL.md#L8-L45)).

**FACT.** Deterministic reuse is enforced by the runner. A step is skipped when its output path
holds a completed record, and a distributed lock prevents two processes building the same step
([`lazy-artifacts.md#L119-L123`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L119-L123)).
Failures are collected rather than swallowed: the scheduler accumulates a failure list and raises
`RuntimeError` naming the count, chained from the first exception, and it treats a cached output
that disappeared after its inputs were pruned as a failure
([`step_runner.py#L233-L349`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_runner.py#L233-L349)).
Failed states are not retried automatically. The runner raises `PreviousTaskFailedError` unless
the caller explicitly enables `force_run_failed`
([`step_runner.py#L360-L380`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_runner.py#L360-L380)).
The repository-wide error policy is to let exceptions propagate and never swallow them
([`AGENTS.md#L172-L177`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/AGENTS.md#L172-L177)).

**FACT.** Continuous health checks run as daily "ferries." The canary is a fixed 9.9M-parameter
MoE chosen to exercise routing, attention, and optimizer state "without turning an accelerator
smoke test into a multi-billion-parameter run," configured by environment variables set in a
GitHub Actions workflow
([`experiments/ferries/canary_ferry.py#L4-L87`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/ferries/canary_ferry.py#L4-L87)).
A data ferry runs the full download through tokenize pipeline daily on FineWeb-Edu, with a
validation script that "confirms row counts and dedup fraction across stages"
([`experiments/ferries/OPS.md#L1-L52`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/ferries/OPS.md#L1-L52)).

**FACT.** The stated lesson from the 32B run is "Instrument heavily. Our ability to diagnose and
address issues mid-flight was greatly aided by extensive logging and monitoring"
([`marin-32b-retro.md#L517`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L517)).

## Human and agent roles

**FACT.** The guidelines address both developers and AI agents
([`guidelines.md#L1-L3`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L1-L3)).
The repository ships 39 loadable agent skills under `.agents/skills/`, and `AGENTS.md` instructs
agents to check for a matching skill "before starting any non-trivial task"
([`AGENTS.md#L28-L33`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/AGENTS.md#L28-L33)).

**FACT.** The Agent MoE workflow grants bounded autonomy: create branches, commit and push
without asking, create experiment issues and post comments, submit Iris jobs and kill only its
own jobs, and run experiments through both gates. "Do not stop to ask for confirmation at any
step. If something fails, diagnose and retry or report the failure, do not block waiting for
input"
([`agent.md#L3-L14`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L3-L14)).
Each experiment issue records the exact initiating prompt and is attached to a shared tracking
issue
([`agent.md#L121-L151`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L121-L151)).

**FACT.** The `run-research` skill defines a six-phase loop, Forage, Propose, Run, Interpret,
Promote, Seal, backed by a logbook at `.agents/logbooks/<topic>.md` containing "a living
hypothesis queue" updated as hypotheses are proposed, blocked, falsified, or promoted, on a
long-lived research branch with commit or tag snapshots at milestones
([`.agents/skills/run-research/SKILL.md#L17-L49`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/run-research/SKILL.md#L17-L49)).
Its practical rules include "Record exact command lines for every headline number" and "Treat
failures and negative results as first-class data. Record dead ends and excessive hyperparameter
sensitivity; skip routine bugs or undertuning"
([`run-research/SKILL.md#L99-L105`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/run-research/SKILL.md#L99-L105)).

**FACT.** Research code and production code have different quality bars by design: on a research
branch, "Ad-hoc scripts, temporary config knobs, and copy/paste are acceptable," while
production-facing code keeps the `AGENTS.md` bar
([`run-research/SKILL.md#L62-L67`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/run-research/SKILL.md#L62-L67)).

**FACT.** Agent-created PRs and issues get an `agent-generated` label, and agent comments carry
a robot marker
([`AGENTS.md#L99-L106`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/AGENTS.md#L99-L106)).

**INTERPRETATION.** Autonomy and governance are separate. The Agent MoE contract lets an agent
create branches and issues, submit jobs, diagnose failures, and complete both gates without
confirmation. Marin's general experiment process still uses issue preregistration and PR review
before a full run, and production work from a research branch is extracted into a clean branch
and PR
([`guidelines.md#L66-L92`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L66-L92),
[`run-research/SKILL.md#L78-L97`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/run-research/SKILL.md#L78-L97)).

**INTERPRETATION.** Marin's agents are autonomous within a fixed contract, but their work stays
attributable and reviewable. Karpathy's agent never pauses, and its record is a five-column TSV
plus git history
([`program.md#L64-L112`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L64-L112)).
The difference is not agent capability. It is that a Marin experiment consumes shared cluster
capacity and feeds a published model, so provenance and cost control dominate.

## Strengths and tradeoffs

**Strengths, all evidenced above.** Preregistered hypotheses on issues. Negative results kept
and indexed. Promotion criteria written down before results arrive. Human-readable artifact
addresses. Identity separated from execution so hardware changes do not invalidate results.
Cheap gates protecting expensive runs. Published retrospectives that name their own mistakes.

**Tradeoffs.** Advisory drift means a stale cache can be served after the runner logs a warning
([`lazy-artifacts.md#L131-L141`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L131-L141));
the 32B contamination records the related risk of reusing stale processed data. The hero-run
process accepts mid-flight changes that make phase-to-phase attribution difficult, which the
retrospective concedes
([`marin-32b-retro.md#L513-L514`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L513-L514)).
The gate system depends on a scaling law with a pinned asymptote of 1.6
([`agent.md#L46-L51`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L46-L51)),
so promotion decisions inherit that assumption. The surrounding machinery is large: eleven
libraries under `lib/`, Pulumi infrastructure projects, a cluster scheduler, and a search service.

**INTERPRETATION: what is excessive for a single-model Wordle lab.** The lazy artifact graph,
distributed locking, mirror filesystems and transfer budgets, Iris job submission, isoflop sweeps
and scaling ladders, Feistel shuffles, MinHash fuzzy dedup, daily canary ferries, W&B project
partitioning, and multi-region storage all solve problems this repository does not have. A
Wordle evaluation over 2,315 answers completes in 17 seconds locally
(`results/baseline-random/run.json`). Adopting an executor to cache that would be pure cost.

## Mapping to this repository

This repository already has a deterministic evaluator that writes a `run.json` per experiment
recording config, git commit and diff hash, input file SHA-256 values, and a summary block with
`solved`, `penalized_turns`, `illegal_actions`, `repeat_actions`, and timing
(`results/baseline-random/run.json`). It has 31 recorded result directories, two versioned
prompts under `prompts/wordle-player/`, dataset findings in `docs/dataset-findings.md`, and
Karpathy-derived measurement design in `docs/autoresearch-measures.md`.

### Borrow now

**RECOMMENDATION.** Write the hypothesis before the run. Marin requires an a priori prediction
recorded as preregistration
([`guidelines.md#L84-L87`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L84-L87)).
Part III of the local plan already commits to freezing the Lab 15 comparison rather than revising
Dataset B mid-experiment, which is the same principle. Add a `hypothesis` string and an
`expected_direction` to the experiment config so it lands in `run.json` alongside the result.

**RECOMMENDATION.** Write the promotion rule before the result. The MoE recipe states its gates
and its discretionary factors in advance
([`grug/moe/README.md#L150-L178`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L150-L178)).
Keep the existing lexicographic rule, more solved games first and fewer penalized turns as the
tiebreak, as the sole gate, and list any discretionary factors such as prompt length or latency
explicitly as secondary.

**RECOMMENDATION.** Keep negative results indexed, not deleted. Marin's report index carries
one-line conclusions including failures
([`docs/reports/index.md#L63-L120`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/index.md#L63-L120)),
and the MoE digest reports 32 "Did not work" outcomes as a headline number
([`agent-moe-experiments.md#L29-L36`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/agent-moe-experiments.md#L29-L36)).
The ledger proposed in `docs/autoresearch-measures.md` covers this; add a one-line conclusion
column and keep discarded runs listed.

**RECOMMENDATION.** Separate catalog from policy. Marin keeps dataset handles in the catalog and
mixture weights in the experiment
([`experiments/AGENTS.md#L14-L17`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md#L14-L17)).
Here the analogue is keeping `data/wordle-lexicon.jsonl` and the two original word lists as the
catalog while any answer subsetting, difficulty bucketing, or turn balancing lives in the
experiment definition. Part III's separation of `ANSWERS` from `ALLOWED_GUESSES` is the same
distinction applied to the action space.

**RECOMMENDATION.** Hash inputs and refuse to compare across mismatches. Marin's contamination
went undetected because a cached dataset was trusted
([`marin-32b-retro.md#L335-L344`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L335-L344)).
The evaluator already records input SHA-256 values; add a comparison helper that refuses to
report a head-to-head result when the answer-list or lexicon hashes differ between two runs.

**RECOMMENDATION.** Watch for metric divergence. Two Marin findings turned on disagreeing
metrics: loss improving while task performance degraded in microannealing
([`marin-8b-retro.md#L283-L285`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md#L283-L285)),
and train loss shifting while eval loss stayed flat in the shuffle anomaly
([`marin-32b-retro.md#L355-L363`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L355-L363)).
Here, treat legal-action rate rising while solve rate is flat, or solve rate rising only on
repeated-letter answers, as a signal to investigate rather than a result to report.

**RECOMMENDATION.** Add a cheap canary. Marin holds its daily canary at about 30M parameters and
1B tokens so it stays cheap
([`add_scaling_heuristic.md#L180-L181`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L180-L181)).
The analogue is a fixed small answer subset with a fixed seed that every change runs first, to
catch evaluator or prompt regressions before a full 2,315-game run.

### Defer

**RECOMMENDATION.** Defer a formal artifact graph. The lazy-artifact system exists because
tokenizing multi-terabyte corpora is expensive
([`nemotron.py#L34-L43`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datasets/nemotron.py#L34-L43)).
Revisit only when a single dataset build exceeds roughly ten minutes and is reused across several
experiments.

**RECOMMENDATION.** Defer scaling-law gating. Marin's gate 2 needs four compute-optimal points
and a fitted curve
([`agent.md#L41-L51`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L41-L51)).
With one target model and one task, there is no scaling axis to fit. The concept worth keeping
early is normalizing quality against cost, which for Wordle means solve rate against model calls
or tokens per game, not against FLOPs.

**RECOMMENDATION.** Defer a published report site. Marin publishes analysis sites to durable
hosting with a machine-readable index
([`docs/reports/index.md#L12-L19`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/index.md#L12-L19)).
Markdown under `docs/` plus notebooks is sufficient at this size.

### Reject

**RECOMMENDATION.** Reject advisory drift that serves a stale cache after only a warning. Marin
accepts it because rebuilding is expensive
([`lazy-artifacts.md#L138-L141`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L138-L141)).
Here, recomputation is cheap and the failure mode is worse than the cost. Recompute and compare
hashes.

**RECOMMENDATION.** Reject mid-flight recipe changes as a default. The Tootsie process is a
response to a run that cannot be restarted
([`marin-8b-retro.md#L42-L44`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md#L42-L44)).
A Wordle experiment can be restarted, so it should be, and Part III already commits to not
revising an intervention after preregistration.

**RECOMMENDATION.** Reject a broad benchmark suite. Marin's 19-task table exists because a base
model must be general
([`marin-32b-retro.md#L440-L455`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L440-L455)).
Keep the narrow deterministic evaluator as the single arbiter and keep behavioral measures as
explanatory diagnostics, as `docs/autoresearch-measures.md` already argues.

**RECOMMENDATION.** Reject cluster-shaped tooling. Iris submission, mirror filesystems, transfer
budgets, and distributed locks
([`experiments/AGENTS.md#L28-L49`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md#L28-L49))
solve multi-region, multi-user contention. Notebook-first local work is the right surface here.

## Marin compared with Karpathy autoresearch

| Dimension | Marin | Karpathy autoresearch |
| --- | --- | --- |
| Scope | Full pipeline: curation, filtering, dedup, tokenization, pretraining, posttraining, evaluation ([`lm-pipeline.md#L7-L15`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lm-pipeline.md#L7-L15)) | One single-GPU training script derived from nanochat ([`README.md#L11-L14`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md#L11-L14)) |
| Mutable surface | Experiment definitions, model, heuristic, and optimizer files; promotions land in `model.py`, `heuristic.py`, or `optimizer.py` ([`grug/moe/README.md#L165-L175`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L165-L175)) | `train.py` only; `prepare.py` and the evaluation harness are read-only, no new dependencies ([`program.md#L25-L31`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L25-L31)) |
| Budget normalization | Fixed FLOP budgets at four compute-optimal scales, with wall clock entering through measured throughput ([`agent.md#L33-L51`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L33-L51)) | Fixed 5 minutes of steady-state training excluding startup and compilation; runs past 10 minutes are killed and discarded ([`program.md#L23`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L23), [`#L108`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L108)) |
| Primary metric | `eval/paloma/macro_loss` plus throughput, combined into effective wall-clock speedup ([`agent.md#L21-L25`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L21-L25), [`#L53-L85`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L53-L85)) | `val_bpb`, vocab-size independent so architecture changes compare fairly; VRAM is a soft constraint ([`README.md#L17`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md#L17), [`program.md#L33-L35`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L33-L35)) |
| Orchestration | Lazy artifact DAG lowered to `StepRunner`, dispatched to Iris and Fray with locks and heartbeats ([`lazy-artifacts.md#L100-L129`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L100-L129)) | `uv run train.py > run.log 2>&1` in a shell loop ([`program.md#L94-L105`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L94-L105)) |
| Provenance | Issue plus PR plus experiment file plus `{name}/{version}` artifact records plus W&B runs and reports ([`guidelines.md#L62-L76`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L62-L76), [`lazy-artifacts.md#L31-L35`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L31-L35)) | One commit per idea plus a five-column untracked `results.tsv` with status `keep`, `discard`, or `crash` ([`program.md#L64-L88`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L64-L88)) |
| Promotion | Two gates plus curvature and stability criteria plus a documented discretionary decision ([`grug/moe/README.md#L150-L178`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L150-L178)) | Lower `val_bpb` advances the branch, otherwise `git reset`; a qualitative simplicity criterion breaks near-ties ([`program.md#L37`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L37), [`#L103-L104`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md#L103-L104)) |
| Intended scale | 5e24 model FLOPs, 500B+ parameter MoE, TPU pods and multislice ([`README.md#L27-L29`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md#L27-L29)) | One NVIDIA GPU, about 12 experiments per hour and about 100 overnight ([`README.md#L23`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md#L23), [`#L64`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md#L64)) |

**FACT.** Autoresearch names its own portability limit: fixing wall clock rather than compute
means "your runs (and results) become not comparable to other people running on other compute
platforms"
([`README.md#L64`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md#L64)).

**INTERPRETATION.** The two designs answer different questions. Autoresearch asks which
configuration wins on this machine in five minutes, and optimizes iteration count. Marin asks
which change will still win three orders of magnitude away, and spends its budget on
comparability and provenance. This repository sits closer to autoresearch in scale and closer to
Marin in aim, because a Wordle policy has to be compared across prompts, datasets, and training
methods over months, so run records need to outlive the machine that produced them. The concrete
adaptation, already argued in `docs/autoresearch-measures.md`, is to replace the fixed wall-clock
budget with a fixed answer set and rollout budget, so that model latency does not affect measured
policy quality.

## Recommended reading order

1. [`README.md#L15-L42`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md#L15-L42) for mission and current program. Two minutes.
2. [`docs/explanations/guidelines.md#L55-L131`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md#L55-L131) for the experiment lifecycle. This is the highest-value page for a small lab.
3. [`experiments/grug/moe/agent.md#L16-L85`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md#L16-L85) and [`experiments/grug/moe/README.md#L150-L178`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md#L150-L178) for a fully specified gate and promotion protocol.
4. [`docs/reports/marin-32b-retro.md#L333-L371`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L333-L371) and [`#L511-L518`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md#L511-L518) for the contamination and shuffle failures with their lessons.
5. [`docs/reports/marin-8b-retro.md#L275-L303`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md#L275-L303) for the microannealing data-mixture ablation and its metric divergence.
6. [`docs/explanations/lazy-artifacts.md#L1-L149`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md#L1-L149) for identity, fingerprints, and caching, once the process is understood.
7. [`experiments/tutorials/exp1078_reproduce_dclm_7b1x.py#L1-L74`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/tutorials/exp1078_reproduce_dclm_7b1x.py#L1-L74) as the shortest complete experiment definition.
8. [`docs/recipes/add_scaling_heuristic.md#L1-L200`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md#L1-L200) if scaling behavior ever becomes relevant.
9. [`karpathy/autoresearch/program.md`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md) for the minimal contrast, roughly five minutes.

## Bibliography

All Marin links pinned to `3f2976e5311c26f540bbc6e61d68a20200cd86f5`; all autoresearch links
pinned to `228791fb499afffb54b46200aca536f79142f117`.

**Documentation (Marin)**

- [`README.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/README.md): mission, open development, current program, worked example.
- [`docs/explanations/guidelines.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/guidelines.md): issue and experiment lifecycle, preregistration, experiment PR rules.
- [`docs/explanations/experiments.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/experiments.md): experiment as unit of inquiry and as DAG.
- [`docs/explanations/lazy-artifacts.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lazy-artifacts.md): handles, fingerprints, advisory drift, adoption.
- [`docs/explanations/lm-pipeline.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/lm-pipeline.md): nine-stage pipeline and tool choices.
- [`docs/explanations/evaluation.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/evaluation.md): evaluation modes, task sets, in-loop metrics.
- [`docs/explanations/marin-prefix.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/explanations/marin-prefix.md): artifact storage addressing.
- [`docs/system-prompts/05-18-2025.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/system-prompts/05-18-2025.md): first-party definition of the open lab and its participation model.
- [`docs/recipes/add_scaling_heuristic.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/recipes/add_scaling_heuristic.md): scaling law versus heuristic, sweep and ladder workflow, definition of done.
- [`AGENTS.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/AGENTS.md) and [`experiments/AGENTS.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/AGENTS.md): agent conventions, catalog versus policy, cost rules.
- [`.agents/skills/run-research/SKILL.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/run-research/SKILL.md), [`background-research`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/background-research/SKILL.md), [`wandb-reporting`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/.agents/skills/wandb-reporting/SKILL.md): research loop, prior-work search, tracking policy.

**Source (Marin)**

- [`experiments/tutorials/exp1078_reproduce_dclm_7b1x.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/tutorials/exp1078_reproduce_dclm_7b1x.py): representative experiment definition.
- [`experiments/grug/moe/agent.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/agent.md) and [`experiments/grug/moe/README.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/grug/moe/README.md): autonomous gates, effective speedup, promotion criteria, fitted scaling rules.
- [`experiments/datasets/nemotron.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datasets/nemotron.py): catalog handles, pinning, provenance.
- [`experiments/datakit/README.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/README.md) and [`experiments/datakit/decontam/config.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/datakit/decontam/config.py): dedup and decontamination policy.
- [`experiments/evals/task_configs.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/evals/task_configs.py): named task sets with shot counts.
- [`experiments/ferries/canary_ferry.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/ferries/canary_ferry.py) and [`experiments/ferries/OPS.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/experiments/ferries/OPS.md): daily canaries and pipeline validation.
- [`lib/marin/src/marin/experiment/train.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/train.py), [`evaluation.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/evaluation.py), [`cli.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/experiment/cli.py), [`execution/step_runner.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_runner.py): training API, eval aggregation, plan-versus-run CLI, scheduler and failure handling.
- [`lib/marin/src/marin/execution/step_spec.py`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/lib/marin/src/marin/execution/step_spec.py): lowered step identity, dependencies, dispatch, and resource specification.

**Technical reports (Marin)**

- [`docs/reports/index.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/index.md): curated index of experiment issues with one-line conclusions.
- [`docs/reports/marin-8b-retro.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-8b-retro.md): Tootsie process, phases, microannealing ablation.
- [`docs/reports/marin-32b-retro.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/marin-32b-retro.md): four-phase run, contamination and shuffle failures, benchmark table, lessons.
- [`docs/reports/agent-moe-experiments.md`](https://github.com/marin-community/marin/blob/3f2976e5311c26f540bbc6e61d68a20200cd86f5/docs/reports/agent-moe-experiments.md): 80 agent-run experiments with outcome classification, machine-generated.

**Papers cited by Marin**

These are direct links to the papers behind recipes or comparisons discussed above. They provide
background, not evidence for claims about Marin's own process.

- [DataComp-LM](https://arxiv.org/abs/2406.11794): recipe reproduced by the representative experiment.
- [Nemotron-CC](https://arxiv.org/abs/2412.02595): major source in the 8B and 32B data mixtures.
- [OLMo 2](https://arxiv.org/abs/2501.00656): comparison model and source of evaluation conventions.
- [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046): Marin-authored optimizer study referenced in the 32B retrospective.

**Karpathy autoresearch (primary)**

- [`README.md`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md): design choices, fixed budget, metric, portability caveat.
- [`program.md`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md): the agent contract covering mutable surface, budget, results ledger, and the keep-or-reset loop.

**Local repository context (not evidence about Marin)**

- `docs/autoresearch-measures.md`, `docs/dataset-findings.md`, `results/*/run.json`, `README.md`.
