# Lab 17 analysis: structured-state LoRA

## Main conclusion

The derived-state package improved policy learning, but the primary paired
experiment is not decisive on one seed.

B-structured produced valid actions for both branches on 4 of 34 perturbation
pairs. B-raw produced none. Branch consistency rose from 4 of 68 to 19 of 68,
and the pair-level bootstrap interval for B-structured was 16.2% to 39.7%,
compared with 1.5% to 11.8% for B-raw. The exact paired-consistency test was
`p=0.125`, though, so 4 paired successes do not establish a reliable effect by
themselves.

The guardrails point in the same direction. Fixed-state usable rate rose from
3 of 47 to 11 of 47, and fixed-opening solve rate rose from 0 of 19 to 5 of 19.
This is the first Part II intervention that improved controlled state
consistency and completed games at the same time.

The right claim is:

> The derived-state package improved policy learning under the same examples,
> targets, optimizer updates, and supervised outputs.

The notebook cannot assign that gain to explicit constraints alone. The
intervention also adds candidate count and increases processed input tokens
from 1,300,155 to 2,456,934, a ratio of 1.890.

## Findings

### State changes affect the action now

B-raw changed its action on 5 of 31 parsed pairs, or 16.1%. B-structured
changed its action on 18 of 24 parsed pairs, or 75.0%.

That jump matters because Lab 16's dominant failure was state insensitivity.
The structured model no longer maps nearly every neighboring state to the same
action. The expected consequence is more branch-specific behavior, and the
consistency counts show some of it:

| Metric | B-raw | B-structured |
| --- | ---: | ---: |
| Both branches parsed | 31/34 | 24/34 |
| Paired consistency, all pairs | 0/34 | 4/34 |
| Branch consistency, all branches | 4/68 | 19/68 |
| Parsed-pair sensitivity | 5/31 | 18/24 |

The improvement is incomplete. Ten B-structured pairs had at least one
unparseable output, versus three for B-raw. Outputs such as `BETN`, `ACHT`,
`FIDIC`, `BEYOND`, and `LETHEN` show that stronger state response came with
weaker format control on this probe.

### The gains concentrate in mixed states

The four paired successes were not evenly distributed:

| Pair scope | Pairs | Parsed | Paired consistent | Consistent branches |
| --- | ---: | ---: | ---: | ---: |
| Broad | 11 | 7 | 1 | 4 |
| Mixed | 11 | 9 | 3 | 11 |
| Narrow | 12 | 8 | 0 | 4 |

Mixed states account for three paired successes and 11 of the 19 consistent
branches. Broad states improved from zero consistent branches under B-raw to
four, including one fully correct pair. That is real movement on the failure
Lab 15 exposed, but 1 of 11 broad pairs is still poor policy performance.

The feedback-change breakdown is too small for a semantic ranking. Among
parsed B-structured pairs, all B/G groups had high sensitivity, while Y/G
groups had none. Parse failures and small cells make that a prompt for a later
targeted test, not evidence that the model has learned one transition and
ignored another.

### Fixed-state performance improved beyond the paired probe

B-structured produced usable actions on 11 of 47 fixed states, or 23.4%.
B-raw's training-interface baseline was 3 of 47, or 6.4%.

The bucket counts show where the extra successes came from:

| Candidate bucket | States | B-structured usable |
| --- | ---: | ---: |
| 1-2 | 16 | 4 |
| 3-10 | 17 | 6 |
| 11-50 | 8 | 0 |
| 51-200 | 6 | 1 |

Lab 15's three B-raw successes all occurred in 1-2 candidate states.
B-structured added six successes in 3-10 candidate states and one in a
51-200 candidate state. The 11-50 bucket remained at zero. Structured state
therefore extended usable behavior beyond near-terminal positions, but it did
not produce dependable broad-state play.

### Gameplay improved under a matched interface

Both models played the same 19 fixed-opening games through their respective
training-format prompts. B-raw solved none. B-structured solved five:

```text
BRICK  turn 6
ROUND  turn 2
SLATE  turn 4
CRANE  turn 4
SHEEP  turn 3
```

B-structured's call-level usable and history-consistency rates were both
10 of 84, or 11.9%. B-raw scored zero on both measures across 95 calls.
Parsed-output repetition fell from 42.1% to 27.4%.

The model still has obvious attractors. `CODY` appeared 25 times, and several
invalid strings repeated five times each. Five solved games are a useful
guardrail result, not evidence of a generally usable Wordle policy.

### Auxiliary accuracy rose as expected

`CHOOSE_VALID` accuracy rose from 65.3% to 95.9%, while
`VALID_CANDIDATE` rose from 76.5% to 91.8%.

These prompts hand B-structured the decoded constraints needed to answer the
question. The result confirms that the adapter can use the representation, but
it is not an independent transfer result. The paired and gameplay evaluations
carry the policy claim.

### Training completed without the Lab 09 memory failure

The first attempt projected logits for every prompt position and killed the
MPS kernel. The corrected path used `logits_to_keep` for response-predicting
positions, disabled the KV cache, and retained an effective batch size of 16
through four-example gradient microbatches.

MPS driver allocation reached 10.69 GiB by step 75 and then stayed at
10.69 GiB through step 1,029. Validation loss fell from 7.328 before training
to 1.259 at the final checkpoint. The flat memory trace distinguishes the
fixed allocation plateau from the earlier full-logit growth.

## Surprises

The intervention did more than increase sensitivity. Lab 16 left open the
possibility that explicit fields would merely make the output change more
often while remaining wrong. B-structured did make more individually
consistent actions, produced four fully correct perturbation pairs, and solved
five games. That weakens the claim that Qwen3-0.6B cannot use state constraints
at all.

The parse regression is the unpleasant part. Both-parse rate fell from 91.2%
to 70.6% even though the response contract did not change. More explicit input
did not make output formatting easier.

Narrow states were not the easiest perturbation scope. They produced zero
paired successes, while mixed states produced three. Candidate count alone
does not explain where the model succeeds.

## Limitations

This is one adapter seed over 34 paired probes. The exact paired test does not
reject equal paired-consistency performance at conventional thresholds.

The intervention bundles three changes:

1. decoded Wordle constraints;
2. symbolic candidate count;
3. 1.890 times as many input tokens.

The experiment holds rows, targets, row weights, splits, optimizer updates,
and supervised tokens fixed. It does not hold input-token exposure or
inference dependencies fixed.

The repository's 2,315-answer list still defines legal actions in this frozen
experiment. Part III will add the larger allowed-guess vocabulary.

Branch bootstrap intervals resample the 34 pairs and preserve within-pair
correlation. They describe each model separately. They are not a replacement
for replication across training seeds.

The microbatched structured run preserves the 16-example effective batch and
optimizer schedule, but its floating-point accumulation order differs from
the original B-raw run.

## Implications

Lab 17 has answered the representation question well enough to move on, with
one qualification: treat the result as promising rather than settled.

The evidence chain is:

> explicit derived state raised sensitivity from 16.1% to 75.0% on parsed
> pairs -> branch consistency rose from 4/68 to 19/68 -> fixed-state and game
> results improved too -> state representation was one bottleneck, but broad
> policy and format control remain weak.

Lab 18 should keep B-structured as the representation control and test the
PRD's trajectory hypothesis. Full teacher trajectories should show whether
sequential exposure improves the mixed and broad states where Lab 17 remains
unreliable.

If later work needs to isolate the Lab 17 mechanism, run two ablations against
the same Dataset B rows:

* remove or hold `CANDIDATE_COUNT` constant;
* compare a compact structured encoding with a length-matched raw control.

A second B-structured seed would answer a different question: whether 4 of 34
paired successes and 5 of 19 solved games reproduce. That replication is the
cleanest way to firm up the statistical claim before attributing the gain to
the representation package.
