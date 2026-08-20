# Labs 18b and 18c: diagnostics on the Lab 18 null

These two labs are not new curriculum. They are diagnostics on a result we do not
trust yet. `part2-prd.md` is unchanged, and Lab 19 keeps its number.

## Why these exist

Labs 14, 17, and 18 each changed the training data and each produced a smaller
effect than the last. Lab 18's primary readout was null: 90/620 to 99/620 usable,
delta +1.45 points, 95% CI -1.45 to +4.19, exact p = 0.374.

Two explanations have never been separated.

1. The policy is genuinely weak, and the data interventions are at their ceiling.
2. The policy is better than free generation reveals, and our metric is dominated
   by the model failing to spell a real word.

Evidence for the second: on the 620-state battery, B-structured parses as five
letters 92.7% of the time but produces an actual answer-list word only 46.1% of
the time. The other 46.6% are strings like `BAGGE`, `WESHT`, and `GALEL`. At the
same time the auxiliary tasks, where the model picks from a supplied list, score
95.9% (`CHOOSE_VALID`) and 91.8% (`VALID_CANDIDATE`). Given options the model is
near perfect. Asked to produce one, it hallucinates about half the time.

A third explanation has also never been tested: none of it is real, because every
conclusion since Lab 15 rests on a single seed.

18b addresses the first pair. 18c addresses the third.

## Ordering

18b runs first, and not only because it is cheaper.

18c measures seed noise, and seed noise has to be measured on the interface we
intend to keep. If constrained ranking turns out to be substantially lower
variance than free generation, then running 18c first would carefully quantify
the noise band of a metric we are about to abandon. 18b decides the measurement.
18c then replicates on it.

## Lab 18b: constrained ranking probe

No training. Existing adapters only.

### Hypothesis

If the answer lexicon is imposed at decode time rather than left to the model,
state-conditioned correctness rises substantially. If it does not, the policy is
weak on its own terms and no further data intervention is worth running.

### Method

For each of the same 620 held-out states, score every one of the 2,315 answer
words under the model and keep the full log-probability vector. The artifact is a
620 by 2315 matrix per model, about 5.7 MB. Every downstream question is then a
filter over that one matrix, so the analysis reruns without touching the GPU.

Scoring uses summed log P(word tokens + EOS | prompt), which is the true string
likelihood and is what a real constrained decoder maximizes. Length-normalized
mean log-probability is recorded as a sensitivity check, because words tokenize
into 1, 2, or 3 tokens and unnormalized sums tilt toward shorter tokenizations.

### Tiers

- Tier 0, free generation. Already measured. 90/620 for B-structured and 99/620
  for G-structured.
- Tier 1, argmax over all 2,315 answers. Answers whether lexicon grounding is the
  bottleneck.
- Tier 2, argmax restricted to the consistent candidate set. History consistency
  is 100% by construction here, so the metric becomes teacher match. This
  separates "knows which words are legal" from "knows which legal word is best."
- Rank statistics. Where the teacher word lands in the full ranking, mean
  reciprocal rank, and top-k hit rates. If the teacher word sits at rank 3 of
  2,315, free generation is discarding a great deal of signal. These cost nothing
  extra and are richer than top-1.

### Models

- B-structured, the incumbent working representation
- G-structured, Lab 18's arm
- Qwen3-0.6B base with no adapter, scored through the same structured interface

The base control is the one that matters most. If an untrained model ranks about
as well as our adapters, the gain came from the lexicon restriction and not from
anything we trained, and that reframes four labs. B-raw is optional and off by
default, since it needs its own raw interface and Lab 17 already showed it is
worse.

### Statistics

Same 620 states throughout, so everything is paired. Reuse Lab 18's
`paired_battery_metric`: flip counts, paired bootstrap 95% CI, and exact McNemar.

### Preregistered readings

- Constrained is close to free generation. Grounding is not the bottleneck, the
  policy is genuinely weak, and Part II's data labs are at their ceiling. This is
  a stop signal and we report it as one.
- Constrained is far above free generation. We have been scoring a decoding
  failure since Lab 15, and Labs 15 through 18 need rereading under the new
  interface.
- Constrained is far above free generation, but the base model matches our
  adapters. The lexicon constraint did the work, not the training.
- Tier 2 teacher match is near chance, meaning 1 over candidate count. The model
  learned consistency filtering and never learned the entropy policy. That is a
  different problem than the one we have been attacking.
- Ranking is state-blind, meaning the same few words top the ranking regardless of
  feedback. Lab 16's finding reappearing in the likelihood surface.

### Cost

About 16 minutes per model on MPS, so roughly 50 minutes for three. Measured, not
estimated: the transformer forward dominates and the softmax is free, so the only
lever is total scored positions. Words are bucketed by token length so no padding
positions are scored.

The Lab 09 rule still applies. Gather target-token log-probabilities per chunk
and never materialize full-vocabulary logits across all positions.

## Lab 18c: seed replication

Two additional B-structured seeds under an otherwise identical config, giving
three runs of the same arm.

### Question

Do Lab 17's 4/34 paired result and Lab 18's +1.45 point delta fall inside
B-structured's own seed-to-seed spread?

### Staging

One arm's noise band is enough to invalidate. If B's spread on the primary
battery is wider than the effects we have been reporting, everything since Lab 15
is noise and we stop. Only if the band is tight do we spend two more runs on G
seeds to properly validate the delta.

About 40 minutes per seed including evaluation.

### Interface

Whatever 18b establishes as primary. If constrained ranking wins, 18c reports
seed spread under constrained ranking, with free generation kept as a secondary
column so the Lab 15 through 18 record stays comparable.

### What it cannot do

Three seeds give a crude spread, not a precise variance estimate. The goal is a
decision, not a publication-grade interval: is the effect we have been chasing
larger than the noise, yes or no.
