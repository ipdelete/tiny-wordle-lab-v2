# Wordle dataset findings

This document records the main findings from
[`01_exploratory_data_analysis.ipynb`](../notebooks/01_exploratory_data_analysis.ipynb).
The notebook contains the pandas code and full outputs behind each number.

## Dataset snapshot

The analysis uses `data/wordle-lexicon.jsonl`:

| Measure | Value |
| --- | ---: |
| Legal words | 12,972 |
| Original answers | 2,315 |
| Guess-only words | 10,657 |
| Duplicate words | 0 |
| Words with a Zipf frequency | 8,926 |
| Words missing a Zipf frequency | 4,046 |
| SHA-256 | `c833f0be0c5328df33efc7fed8217561b1f6a19e8a6864823406827706786a98` |

Original answers make up 17.85% of the legal vocabulary. The remaining 82.15%
are valid guesses but cannot be hidden answers in the original Wordle list.

## Findings

### The answer list favors familiar words

Original answers have a mean Zipf frequency of 3.60. Guess-only words average
2.18. Their quartiles tell the same story:

| Vocabulary | 25th percentile | Median | 75th percentile |
| --- | ---: | ---: | ---: |
| Original answers | 2.96 | 3.52 | 4.18 |
| Guess-only words | 1.49 | 1.99 | 2.64 |

Only 40 answers have a recorded Zipf frequency below 2.0, compared with 3,329
guess-only words. The most frequent words are almost all answers, including
`about`, `their`, `there`, `which`, and `would`.

This is a large selection effect. A legal word is not automatically a
plausible answer.

### Missing frequency is confined to guess-only words

All 2,315 answers have a recorded Zipf frequency. The 4,046 missing values,
31.19% of the full dataset, belong entirely to guess-only words.

Missing frequency is therefore not random. Treating missing values as an
average frequency would make obscure legal guesses look more answer-like than
the data supports.

### The legal vocabulary contains many inflected forms

The final letter `s` appears in 3,958 legal words but only 36 original answers.
Common suffixes in the full vocabulary include `es`, `ed`, `er`, `ts`, `ks`,
and `ds`.

The final-`s` gap reflects the many plural and inflected forms allowed as
guesses. Not every final-`s` word is a plural, but the size of the gap makes
the curation rule visible.

### Answers repeat letters less often

Repeated letters occur in 32.4% of answers and 36.6% of guess-only words.
Answers average 4.65 unique letters per word, compared with 4.60 for
guess-only words.

The difference is real but modest. Repeated letters help distinguish the two
groups, but they are not rare enough to reject an answer candidate on their
own.

### Vowel count barely separates the groups

Answers average 1.77 vowels, while guess-only words average 1.81. Counting
`a`, `e`, `i`, `o`, and `u` alone does little to identify plausible answers.

Vowels still matter for information gain during play. They are weak evidence
for answer plausibility.

### Parts of speech overlap

Every word has at least one part-of-speech label. Nouns are the largest group,
followed by verbs and adjectives. A word may have several labels, so these
counts overlap and should not be added as if they were exclusive categories.

Parts of speech describe the vocabulary, but frequency and morphology show a
clearer difference between answers and guess-only words.

## Implications for experiments

1. Keep legal actions and possible answers as separate sets. A policy may use
   an obscure legal guess for information, but it should not assign that word
   the same answer probability as a familiar candidate.
2. Frequency is a useful answer prior. It should break ties or rank surviving
   candidates, not override feedback consistency.
3. Do not impute missing Zipf values with the dataset mean. Missing values
   identify a distinct guess-only population in this dataset.
4. Evaluate repeated-letter answers separately. Aggregate solve rate can hide
   policies that gather new letters well but handle duplicate constraints
   poorly.
5. Report results against the fixed answer list. Sampling uniformly from all
   legal guesses would measure a different task.

## Limits

These findings describe the original 2,315-answer and 12,972-guess lists in
this repository. They do not describe later Wordle answer-list revisions.

Zipf frequency comes from `wordfreq`, and parts of speech combine dictionary
sources with the recorded fallback classification. These fields are useful
annotations, not Wordle rules. The analysis also makes no claim about model
quality or causal effects on solve rate.

## Reproducing the analysis

Start JupyterLab from the repository root:

```bash
scripts/run_jupyter.sh
```

Then run every cell in
`notebooks/01_exploratory_data_analysis.ipynb`. The notebook reads the
canonical JSONL file directly.
