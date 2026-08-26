# Lexicon sources

`wordle-lexicon.jsonl` combines the unchanged original Wordle lists with
frequency and lexical data from these sources:

- [wordfreq](https://github.com/rspeer/wordfreq) supplies English Zipf
  frequencies. The package is Apache-2.0 and documents the licenses of its
  underlying corpora.
- [Kaikki](https://kaikki.org/dictionary/English/) supplies the primary
  English Wiktionary extraction. Wiktionary content is available under
  CC BY-SA and GFDL; Kaikki asks research users to cite Wiktextract.
- [Moby Part-of-Speech II](https://www.gutenberg.org/ebooks/3203) supplies
  fallback POS labels. Project Gutenberg distributes this work as public
  domain in the United States.
- [GCIDE 0.54](https://gcide.gnu.org.ua/) supplies historical fallback
  entries. GCIDE 0.54 is distributed under GPL-3.0-or-later.

The builder uses the first source that classifies a word. Aggregate source
counts and source versions are recorded in `wordle-lexicon.metadata.json`; the
word records contain only the resulting POS labels.

Words missed by all dictionaries may be classified by the local
`omlx-qwen-38-27b` model. Generated labels remain separate in
`wordle-pos-qwen.jsonl`.

The Dictionary of the Scots Language is not an automated source. Its terms
allow limited research extracts but prohibit systematic downloading without
an explicit license.
