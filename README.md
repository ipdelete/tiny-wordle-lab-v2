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