## Wordle vocabulary

The original source lists are kept unchanged:

- `data/wordle-answers-original.txt` contains the 2,315 possible answers.
- `data/wordle-guesses-original.txt` contains all 12,972 legal guesses.

`data/wordle-lexicon.jsonl` adds answer membership, English Zipf frequency,
and dictionary-backed parts of speech to each legal guess. Its records follow
`data/wordle-lexicon-record.schema.json`.

Build the enriched lexicon with:

```bash
uv run python scripts/fetch_lexical_sources.py
uv run python scripts/build_wordle_lexicon.py
```

The fetch step caches source archives under `.cache/wordle-lexicon/`. The
builder checks sources in this order:

1. The English Kaikki extract of Wiktionary
2. Moby Part-of-Speech II
3. GCIDE 0.54
4. `omlx-qwen-38-27b` for words missed by all dictionaries

The first source that classifies a word wins. The metadata file records
aggregate coverage by source. Before running the optional Qwen fallback, words
absent from all three dictionaries keep an empty POS array.

Classify those remaining words and rebuild with:

```bash
uv run python scripts/classify_unmatched_pos.py
uv run python scripts/build_wordle_lexicon.py
```

Qwen labels are stored separately in `data/wordle-pos-qwen.jsonl`. Final
lexicon records contain only the resulting POS array; model and prompt details
stay in `data/wordle-pos-qwen.metadata.json`.

The generator treats a `wordfreq` result of `0.0` as missing and writes
`null`. It writes records in alphabetical order so repeated builds produce
the same files.

Frequency values come from
[`wordfreq`](https://github.com/rspeer/wordfreq). Lexical labels come from the
[Kaikki English dictionary](https://kaikki.org/dictionary/English/),
[Moby Part-of-Speech II](https://www.gutenberg.org/ebooks/3203), and
[GCIDE](https://gcide.gnu.org.ua/).

The Dictionary of the Scots Language is useful for manually investigating
remaining words, but its terms prohibit systematic downloading. It is not part
of the automated pipeline.

## Research harness

The research harness evaluates a policy against the same Wordle rules, answer
set, action budget, and result format. Its default benchmark plays all 2,315
answers. This is an exact deterministic comparison for a fixed policy; it does
not use sampled answer batteries or statistical significance tests.

Run the built-in baselines with:

```bash
uv run tiny-wordle-lab-v2 evaluate \
  --policy random \
  --experiment-id baseline-random

uv run tiny-wordle-lab-v2 evaluate \
  --policy frequency \
  --experiment-id baseline-frequency

uv run tiny-wordle-lab-v2 evaluate \
  --policy candidate-entropy \
  --experiment-id baseline-candidate-entropy

uv run tiny-wordle-lab-v2 evaluate \
  --policy open-entropy \
  --experiment-id baseline-open-entropy
```

Candidate entropy chooses only among answers that remain consistent with the
history. Open entropy scores every one of the 12,972 legal guesses and prefers
a remaining answer when several actions have the same maximum entropy.

Each evaluation writes:

```text
results/<experiment-id>/
├── run.json
└── games.jsonl
```

`run.json` records the configuration, summary, source hashes, Git state,
runtime environment, and the hash of `games.jsonl`. The result files validate
against `schemas/experiment-result.schema.json`. The `results/` directory is
ignored by Git; schemas and small test fixtures remain versioned.

Compare completed runs with:

```bash
uv run tiny-wordle-lab-v2 compare results/*/run.json
```

The comparison orders runs by solved games and then by penalized turns. A win
costs the number of opportunities used; a failure costs seven. Illegal actions
consume an opportunity and receive no feedback. Repeated legal guesses consume
an opportunity and receive normal feedback.

The initial full-list baselines are:

| Policy | Solved | Penalized turns | Illegal | Repeats |
| --- | ---: | ---: | ---: | ---: |
| Random, seed 0 | 2/2,315 | 16,196 | 0 | 0 |
| Frequency-ranked candidate | 2,270/2,315 | 9,301 | 0 | 0 |
| Candidate-only entropy | 2,304/2,315 | 8,330 | 0 | 0 |
| Open entropy, 12,972 actions | 2,315/2,315 | 8,020 | 0 | 0 |

Two independent executions produced identical per-game artifacts for every
policy.

The policy seam is intentionally small:

```python
class Policy(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    def choose(self, observation: Observation) -> str: ...
```

`Observation` exposes the public game history, previous policy outputs,
remaining opportunities, and the answer candidates consistent with the
history. It never exposes the hidden answer. Future model adapters can use or
ignore the derived candidate set without changing the evaluator. The
descriptor records the effective policy name and parameters in each run.

### Untouched language-model baseline

The LiteLLM adapter sends only public Wordle history to a chat model. It does
not provide the hidden answer or the harness's derived candidate list. Raw
model text crosses the policy boundary, so malformed, illegal, and repeated
guesses remain visible to the evaluator.

Run a small GPT-OSS-20B smoke test through the local gateway:

```bash
uv run tiny-wordle-lab-v2 evaluate \
  --policy litellm \
  --model gpt-oss-20b \
  --answers foyer,banal,sissy \
  --experiment-id gpt-oss-20b-smoke
```

The adapter reads `LITELLM_MASTER_KEY` from `~/src/wmd-router/.env` by default.
Use `--env-file` or `--api-base` to select another local configuration.

### Prompt optimization before training

The model-facing Wordle prompt lives in
`prompts/wordle-player/v1-baseline.md`; evaluations record its source path and
SHA-256. `prompt_optimization/` contains a standalone SkillOpt exercise that
uses frozen train, validation, and final-holdout answer splits. It optimizes
only that Markdown prompt against the deterministic harness, then freezes the
selected prompt before model-training experiments begin.

See `prompt_optimization/README.md` for the pinned dependency and run command.
The first completed run found no validated improvement, so the original
baseline prompt remains selected and frozen. The proposed `slate` opener tied
the baseline and was rejected rather than promoted.

A controlled follow-up used Sonnet 5 as the optimizer while retaining
GPT-OSS-20B as the target. Its more detailed strategy prompt also failed the
validation gate, dropping from 7/16 to 6/16 solves. Both results are preserved
under `prompt_optimization/`.