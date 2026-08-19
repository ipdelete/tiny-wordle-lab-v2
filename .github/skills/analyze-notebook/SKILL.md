---
name: analyze-notebook
description: Execute and analyze a Jupyter notebook. Use when asked to run a notebook, preserve its cell outputs, export executed results, or write an evidence-based notebook analysis.
---

# Execute and analyze a notebook

Produce two durable artifacts from one notebook:

- `.ai/docs/<notebook-stem>-cell-outputs.md`, with every cell, output, and plot;
- `.ai/docs/<notebook-stem>-analysis.md`, with conclusions grounded in the executed evidence.

Use the repository's existing environment and notebook conventions. Treat the
executed notebook as the source of truth.

## 1. Resolve the notebook

Use the path named by the user. When no path is named, select a notebook only
when the conversation identifies one unambiguously.

Read its title, objective, inputs, and expected artifacts. Inspect the relevant
project configuration and prior result documents needed to interpret it.

**Complete when:** the notebook path, environment command, execution working
directory, and analysis objective are explicit.

## 2. Execute in place

Run the notebook from the working directory its relative paths expect. For this
repository, execute notebooks with:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute <notebook-path> \
  --output <notebook-filename> \
  --output-dir <notebook-directory> \
  --ExecutePreprocessor.timeout=600
```

Use a longer timeout when the notebook documents a legitimate long-running
training or evaluation step. Preserve outputs in the notebook itself.

If execution fails, read the failing cell and traceback. Fix an in-scope
notebook defect and rerun from the beginning; report an external blocker
plainly when execution requires unavailable data, credentials, hardware, or a
service.

**Complete when:** every executable cell finishes without an error output and
the executed notebook is saved at its original path.

## 3. Export the cell transcript

Create `.ai/docs/`, then export the executed notebook:

```bash
uv run jupyter nbconvert \
  --to markdown \
  <notebook-path> \
  --output <notebook-stem>-cell-outputs.md \
  --output-dir .ai/docs
```

Keep the generated `<notebook-stem>-cell-outputs_files/` directory beside the
Markdown file. The transcript must retain Markdown cells, code cells, text and
table outputs, and all plot references.

Validate that:

- the transcript contains every notebook section and code cell;
- every generated image reference resolves to an existing asset;
- no output contains an execution error.

**Complete when:** the Markdown transcript is a faithful, self-contained index
of the executed notebook and every linked local asset exists.

## 4. Analyze the evidence

Read the objective and every executed output before writing
`.ai/docs/<notebook-stem>-analysis.md`.

Structure the analysis around:

1. **Main conclusion** - the strongest answer the evidence supports.
2. **Findings** - quantified observations tied to the notebook's questions.
3. **Surprises** - hypotheses weakened or eliminated by the results.
4. **Limitations** - what the notebook did not measure or cannot establish.
5. **Implications** - the next intervention or experiment justified by the
   evidence.

Use an **evidence chain** for each material claim:

> observed value -> interpretation -> expected consequence -> next experiment

Compare the dimensions implicated by known model symptoms rather than
performing unrelated data archaeology. Prefer task-specific and
cross-dimensional distributions when aggregate values can hide the deployed
behavior. Distinguish nominal example counts from token allocation, unique
states, repeated state exposure, and realistic state visitation.

Treat plausible stories as hypotheses. Kill a hypothesis when the measurements
contradict it, and record the contradiction as a result. Recommend measured
interventions without inventing arbitrary target percentages.

**Complete when:** every conclusion cites executed evidence, every important
surprise is recorded, limitations are explicit, and each proposed next step
tests a named hypothesis.

## 5. Verify the artifacts

Review the notebook and both documents together. Ensure values in the analysis
match the latest execution and generated links remain valid. Run the smallest
existing repository check that covers any notebook code changed while resolving
execution failures.

Commit or push only when the user requests it.

**Complete when:** the executed notebook, transcript, analysis, and assets are
consistent and the worktree contains only intended changes.
