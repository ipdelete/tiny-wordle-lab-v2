# SQ31 Wordle environment analysis

## Main conclusion

SQ31 passes its sampling-policy gate. Temperature 1.0 produced mixed win/loss
groups for 11 of 16 calibration answers, or 68.8%, and 9 of 16 confirmation
answers, or 56.2%. The confirmation Wilson lower bound was 33.2%, well above
the preregistered 2% minimum. Sparse terminal reward therefore supplies usable
group-relative variation for SQ34 without reward shaping.

The caveat is large. Greedy trie decoding agreed with the frozen Lab 18d
full-string argmax on only 45.5% of 77 recorded states. SQ34 can train the trie
policy, but improvement will not automatically mean that the deterministic
Lab 18d evaluation decoder improved. Both decoders must remain separate
reported outcomes.

## Findings

### The environment reproduces the prior experiment

The extracted representation reproduced 3,565 unique structured states byte
for byte. The environment then replayed 78 seed 42 calls, 77 seed 45 calls, and
78 seed 47 calls with matching feedback, candidate counts, repeat flags, and
terminal outcomes.

This closes the compatibility question for the shared in-vocabulary domain.
The notebook also demonstrated the intentional difference outside that domain:
the old benchmark normalizes lowercase output, while SQ31 terminates it as a
contract violation.

### The stochastic policy is normalized and reproducible

The tokenizer produced 2,315 distinct trie leaves. The trie has 5,679 total
nodes, 349 branching nodes, and a maximum depth of four tokens including EOS.
The exact distribution normalized over all 2,315 answers.

Across 512 samples, the binned empirical distribution was 0.0174 total
variation from the exact distribution. That is small enough to support the
claim that the sampler implements the declared trie policy rather than an
approximation or a second undocumented decoder.

### Memory reached a flat plateau

The largest exact-walk forward reached 5.64 GiB of MPS driver memory. All 40
exact-walk iterations then stayed at exactly 5.64 GiB. Sampling settled near
2.64 GiB after its first iteration. The process cap was 128 GiB.

The compact last-position logits removed the dangerous
`batch x sequence x vocabulary` allocation. The measured plateau supports
running the bounded SQ34 collector with the same kernel, although SQ34's
gradient-bearing current-policy and reference-policy loops still need their
own soaks.

### Temperature 1.0 gives the strongest reward diversity

Calibration mixed-group fractions were:

| Temperature | Mixed groups | Solve rate | All zero | All one |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 37.5% | 48.4% | 37.5% | 25.0% |
| 0.75 | 56.2% | 46.9% | 25.0% | 18.8% |
| 1.00 | 68.8% | 37.5% | 25.0% | 6.2% |
| 1.25 | 56.2% | 31.2% | 37.5% | 6.2% |
| 1.50 | 56.2% | 23.4% | 43.8% | 0.0% |

Temperature 1.0 won on mixed groups, then confirmed at 56.2% mixed groups and
42.2% stochastic solve rate on the separate confirmation answers. This is the
right trade for GRPO. Lower temperatures solve more often but produce too many
all-one groups. Higher temperatures lose more often and produce too many
all-zero groups. Neither extreme yields an advantage signal.

At temperature 1.0, confirmation games averaged 5.17 total turns including the
fixed opening and 8.27 model forwards. Repeats occurred on 3.1% of policy
actions. Mean sampled action surprisal rose from 2.17 nats on Turn 2 to 3.72
nats on Turn 4, then fell to 3.01 nats on Turn 6.

## Surprises

The decoder disagreement is larger than expected. A 45.5% match means greedy
trie decoding and full-string argmax choose different words on more than half
of the recorded Lab 18d states. This does not invalidate trie-policy training,
but it sharply limits what a positive SQ34 result would prove.

The full gate was also cheaper than the earlier estimate. The notebook finished
in 536 seconds while preserving hundreds of GiB of system headroom. Compact
logits changed the practical cost of the experiment, not only its safety.

## Limitations

The calibration and confirmation estimates each use 16 development answers
with four episodes per answer. The answer is the independent unit, so the
intervals remain wide.

The 42.2% confirmation solve rate is not a held-out model comparison. It comes
from development answers selected for the sampling gate. It must not be
compared directly with the deterministic 10/19 reserved-answer baseline.

Action entropy by turn is estimated from mean sampled surprisal. It is an
unbiased policy-entropy estimate across sampled actions, not an exact
distribution calculation at every visited state.

SQ31 exercised inference only. SQ34 still needs separate fixed-shape soaks for
current-policy recomputation, reference-policy scoring, and the LoRA update.

## Implications

SQ34 is unblocked with temperature 1.0, group size four, and sparse 0/1 return.
The first run should keep the preregistered diversity-collapse stop. If mixed
groups disappear as the policy sharpens, stop rather than changing temperature
inside the run.

Evaluate both stochastic trie gameplay and deterministic full-string gameplay
at every saved checkpoint. If trie gameplay improves but full-string gameplay
does not, the result is still useful: sparse game reward trained the declared
behavior policy, but the update did not transfer across decoder definitions.

Before the first optimizer update, run the three remaining memory soaks under
the same watchdog. The sampling kernel is safe; that result does not transfer
automatically to gradient-bearing loops.
