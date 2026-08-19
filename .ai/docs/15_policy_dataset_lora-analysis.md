# Lab 15 analysis

## Main conclusion

Dataset B changed model behavior, but it did not produce a usable Wordle
policy.

On the pre-registered 47-state training-format evaluation, Model B made 3
usable calls while Model A made none. That is a 6.4 percentage-point
difference with a paired 95% bootstrap interval of 0.0 to 14.9 points. All
three successes occurred with only one or two candidates left. Model B made no
usable call in any state with three or more candidates.

Neither adapter solved any of the 19 fixed-opening games. The practical
hypothesis that Dataset B would improve broad-candidate policy behavior is not
supported by this run.

Better policy coverage did change the learned behavior. Model B produced three
usable narrow-state calls where Model A produced none, performed better under
the training-format prompt, and often emitted guesses with strong hypothetical
candidate reduction. Those guesses almost never satisfied the supplied
history. Better coverage was not sufficient to produce a state-conditioned
broad-candidate policy.

Prompt transfer made the result worse, but it cannot explain most of the
failure. Model B fell from 6.4% usable under the training prompt to 2.1% under
the deployment prompt. Training-format performance was already catastrophically
low.

## Experimental controls held

Both runs completed 1,029 full batches from independent, identically seeded
Qwen3-0.6B plus LoRA initializations.

| Measurement | Dataset A | Dataset B |
| --- | ---: | ---: |
| Optimizer steps | 1,029 | 1,029 |
| Processed input tokens | 1,381,125 | 1,300,155 |
| Processed supervised tokens | 42,117 | 50,308 |
| Minimum batch size | 16 | 16 |
| Training time | 785.1 seconds | 690.4 seconds |
| Final within-corpus validation loss | 0.7622 | 1.8972 |

The B/A input-token ratio was 0.941, within the pre-registered 0.90 to 1.10
range. Dataset B exposed more supervised response tokens despite fewer total
input tokens because its task distribution contains much more policy
supervision.

The validation losses are not directly comparable because each model uses its
own corpus and development split. Within each run, Model A improved from
8.5384 to 0.7622 and Model B improved from 7.2586 to 1.8972. Model B reached
1.8915 at step 600 and ended at 1.8972. The final checkpoint remained fixed by
the experiment rather than selected after seeing validation behavior.

## Primary result

| Model | Fixed states | Usable calls | Usable rate |
| --- | ---: | ---: | ---: |
| Base | 47 | 0 | 0.0% |
| Dataset A | 47 | 0 | 0.0% |
| Dataset B | 47 | 3 | 6.4% |

The paired B minus A estimate was 6.4 percentage points. Its 95% bootstrap
interval was 0.0 to 14.9 points. The interval touches zero, and the location of
the three successes matters more than the aggregate:

| Answer | Turn | Candidates | Generated | Teacher | Exact |
| --- | ---: | ---: | --- | --- | --- |
| FLING | 4 | 1 | FLING | FLING | yes |
| WASTE | 6 | 2 | WASTE | TASTE | no |
| POINT | 4 | 1 | POINT | POINT | yes |

Dataset B succeeded only after uncertainty had nearly disappeared. It did not
solve the broad-state policy problem that motivated Dataset B:

| Candidate bucket | States | Model A usable | Model B usable |
| --- | ---: | ---: | ---: |
| 1 to 2 | 16 | 0.0% | 18.8% |
| 3 to 10 | 17 | 0.0% | 0.0% |
| 11 to 50 | 8 | 0.0% | 0.0% |
| 51 to 200 | 6 | 0.0% | 0.0% |

The observed value is 3 narrow-state successes and zero broad-state successes.
The interpretation is that extra high-uncertainty policy rows did not make
high-uncertainty decisions usable. The expected consequence is continued
failure during complete games, which is exactly what the gameplay evaluation
found.

## Complete-game behavior

| Model | Games | Solved | Solve rate |
| --- | ---: | ---: | ---: |
| Base | 19 | 1 | 5.3% |
| Dataset A | 19 | 0 | 0.0% |
| Dataset B | 19 | 0 | 0.0% |
| Symbolic teacher ceiling | 19 | 18 | 94.7% |

Every game began with the fixed `RAISE` opening. Both adapters then made five
model calls per game and failed all 19 answers. Neither adapter recorded one
usable gameplay call at any turn.

The base model's single solve is not evidence that it has a better policy. It
generated `CRANE` for 17 calls and happened to solve one answer on turn 2.
That is output concentration meeting a favorable target.

## Dataset B changed the failure mode

Model A collapsed almost completely to `BRAIN`:

| Model A gameplay output | Count |
| --- | ---: |
| BRAIN | 76 |
| BAYOU | 10 |
| Unparsed output | 9 |

Its format-valid and answer-lexicon rates were both 90.5%, but its recorded
repeat rate was 70.5% and its history-consistency rate was zero.

Model B used four outputs:

| Model B gameplay output | Count |
| --- | ---: |
| BRISE | 35 |
| PETEL | 21 |
| RASHY | 20 |
| BETEL | 19 |

All 95 outputs parsed as five-letter words, but only 20.0% belonged to the
2,315-answer lexicon. Those 19 answer-list calls were all `BETEL`. Model B's
recorded repeat rate fell to 11.6%, but its history-consistency rate remained
zero.

The apparent repeat improvement needs care. The gameplay code records a repeat
only when the output is in the answer lexicon, and it adds only answer-list
guesses to `seen`. Repeated invalid outputs such as `BRISE` therefore do not
count as repeats. Model B emitted `BRISE` 35 times, so the recorded 11.6% rate
understates output repetition.

For every model, the ten most frequent generated words account for 100% of
parsed gameplay outputs. Model A used only two parsed words and Model B used
four. Dataset B did not remove output collapse. It moved the collapse from one
mostly valid answer to a small family of word-like outputs.

One of Model B's four outputs, `BETEL`, is among Dataset B's ten most frequent
teacher targets. Dataset B's top teacher targets account for 20.0% of Model
B's parsed gameplay outputs. This is evidence that visitation weighting
affected generation, but the neighboring forms `PETEL`, `BRISE`, and `RASHY`
show that imitation is unstable.

## Prompt transfer is real but not the main bottleneck

| Model | Interface | Answer-lexicon rate | History consistency | Usable rate |
| --- | --- | ---: | ---: | ---: |
| A | Training | 74.5% | 0.0% | 0.0% |
| A | Deployment | 74.5% | 0.0% | 0.0% |
| B | Training | 78.7% | 6.4% | 6.4% |
| B | Deployment | 51.1% | 2.1% | 2.1% |
| Base | Training | 0.0% | 0.0% | 0.0% |
| Base | Deployment | 57.4% | 2.1% | 2.1% |

Model B's usable rate fell from 6.4% with the training prompt to 2.1% with the
deployment prompt. Its answer-lexicon rate fell by 27.6 percentage points.
This supports the known prompt-interface hypothesis.

The training-format result is still only 3 usable calls out of 47. Fixing the
deployment representation alone cannot explain the failure. The model already
fails under the exact interface used for supervised training.

## Candidate reduction is not policy validity

On the fixed states, Model B's mean generated-to-teacher reduction ratio was
0.72 under the training prompt. In the broad buckets it was 0.86 for 11 to 50
candidates and 0.86 for 51 to 200 candidates. These numbers look encouraging
until they are paired with zero history consistency and zero usable calls in
both buckets.

The reduction metric scores any generated answer-list word against the current
candidate set. It does not require the word to satisfy the history. Model B
often generated words that would split the candidates well in isolation but
were invalid for the supplied state.

The observed value is strong hypothetical reduction with zero valid broad
decisions. The interpretation is that the model learned something about
generally informative Wordle words without learning to condition that choice
on history. Lab 16 should inspect whether the model ignores feedback tokens,
loses track of repeated-letter constraints, or maps many histories to the same
latent action.

## Auxiliary guardrails

| Model | CHOOSE_VALID | VALID_CANDIDATE |
| --- | ---: | ---: |
| Base | 0.0% | 0.0% |
| Dataset A | 83.7% | 74.5% |
| Dataset B | 65.3% | 76.5% |

Dataset B retained `VALID_CANDIDATE` accuracy and improved it by 2.0
percentage points. `CHOOSE_VALID` fell by 18.4 points. Capping auxiliary
examples did not erase constraint classification, but it weakened the task
that asks the model to choose between candidates.

This is consistent with the policy result. Dataset B preserved recognition
better than selection.

## Hypotheses weakened or eliminated

### More policy supervision is sufficient

Dataset B made policy examples the majority of the training signal and added
missing high-uncertainty states. Model B still made no usable broad-state call
and solved no games. Better policy coverage changed the output distribution
and produced three narrow-state successes, but it was not sufficient for
state-conditioned broad-candidate play. This experiment does not show that
coverage was irrelevant or unnecessary.

### Deployment prompting is the whole problem

The deployment prompt reduced Model B's usable rate, but the training prompt
produced only three narrow-state successes. Representation transfer matters,
but failure begins before deployment.

### Dataset B fixed target collapse

Model B stopped repeating `BRAIN`, but it generated only four words across 95
gameplay calls. The new distribution changed the attractors rather than
producing state-dependent choices.

### Candidate reduction implies a good policy

Model B's broad-state reduction ratios were high while every broad-state call
was unusable. Information value must be reported with legality and state
consistency.

## Limitations

The primary set contains only 47 states, including 14 broad states. The paired
interval is correspondingly wide.

The 19-game solve rate is too small to estimate rare success precisely.

`in_answer_lexicon` is stricter than normal Wordle because the repository does
not yet contain the full legal-guess vocabulary. Part III will test that
separate action-space question without changing this frozen experiment.

The fixed states come from candidate-only teacher trajectories. They do not
cover every legal off-policy history.

The repeat guardrail ignores repeated outputs outside the answer lexicon.
Lab 16 should add a raw parsed-output repetition measure without changing the
stored Lab 15 result.

Reduction ratios include answer-list words that violate history. They measure
hypothetical information value, not valid action quality.

Each model's validation loss uses a different corpus, so the two loss values
cannot rank the models.

## Implications for Lab 16

The next experiment should diagnose why history never controls the generated
action. Do not regenerate Dataset B yet.

Lab 16 should:

1. Build a failure table for every fixed-state and gameplay call with the raw
   generation, parsed word, lexicon membership, repeated-output status,
   feedback consistency, and violated constraint.
2. Count repeated parsed outputs independently of lexicon membership.
3. Compare each model's output for the same word history under training and
   deployment prompts.
4. Build paired, reachable states that differ by one valid feedback branch.
   Do not invent arbitrary feedback strings that may describe impossible
   games.
5. Report state perturbation sensitivity as the fraction of paired states
   whose generated action changes:

   ```text
   changed generated actions / valid paired state perturbations
   ```

6. Cross action sensitivity with consistency in the perturbed state:

   | Action changes | New action is consistent | Diagnosis |
   | --- | --- | --- |
   | no | no | state insensitivity |
   | yes | no | state misinterpretation |
   | yes | yes | state-conditioned behavior |
   | no | yes in both states | shared valid action; not necessarily a failure |

7. Group failures by candidate bucket, history depth, repeated letters, and
   feedback pattern.
8. Inspect Model B's four gameplay attractors and trace which Dataset B targets
   or prompt fragments are nearest to them.
9. Separate generally informative guesses from state-conditioned valid
   guesses when reporting reduction.

The strongest next hypothesis is:

> Dataset B taught a small vocabulary of generally Wordle-like policy outputs,
> but Qwen3-0.6B still does not bind those outputs to the supplied feedback
> history.

Lab 16 should distinguish two ways that hypothesis could be wrong. The model
may be insensitive to state changes, or it may change its action while
misinterpreting the constraints. The first result points toward a more explicit
state representation. The second points toward targeted constraint learning.
Make that distinction before Lab 17 changes the representation.
