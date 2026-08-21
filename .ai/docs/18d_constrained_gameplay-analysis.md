# Lab 18d analysis: constrained full-game evaluation

## Main conclusion

Answer-constrained decoding materially improves complete gameplay on the fixed
19-answer battery. Every B-structured seed solved 10/19 games (52.6%) under
answer-constrained ranking, compared with 5/19, 3/19, and 3/19 under free
generation. Across the 57 paired seed-answer runs, constrained decoding gained
21 solves and lost 2. The mean within-seed improvement was 33.3 percentage
points.

This is deployed capability, not merely a better state-level metric. The
constrained models reduced the Turn 2 candidate set from a mean of 49.4 words to
5.2-10.1 words and ended games with only 1.2-1.3 candidates on average. The
state-conditioned ranking uncovered in Labs 18b and 18c therefore compounds
into substantially more solved games.

The learned policy is still strategically weak. None of the 57 constrained
Turn 2 actions matched the candidate-only entropy teacher. Mean open-action
regret was 1.05-1.45 bits, and only one action was within 0.25 bits of the best
answer-list action. The model also failed to close known solutions: when one
candidate remained, it selected that candidate on only 18/65 calls (27.7%).
Eighteen of the 27 constrained game failures ended with exactly one candidate
remaining.

Lab 19 should therefore distill relative action value rather than exact teacher
identity, but it should address two regimes: information-seeking choices in
broad early states and reliable candidate selection in sharp late states.

## Findings

### Constrained ranking more than doubles mean solve rate

| Seed | Free | Free-continue | Answer-constrained | Constrained minus free |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 5/19 (26.3%) | 5/19 (26.3%) | 10/19 (52.6%) | +26.3 points |
| 45 | 3/19 (15.8%) | 3/19 (15.8%) | 10/19 (52.6%) | +36.8 points |
| 47 | 3/19 (15.8%) | 3/19 (15.8%) | 10/19 (52.6%) | +36.8 points |

The constrained decoder gained six answers and lost one for seed 42, gained
eight and lost one for seed 45, and gained seven without a loss for seed 47.
Six answers were solved by all three constrained seeds, five were missed by all
three, and eight had seed-dependent outcomes.

Observed gains in every seed -> the replicated answer ranking survives
trajectory rollout -> Labs 18b and 18c exposed behavior that matters in games,
not only on isolated states -> answer-constrained gameplay should replace free
generation as the primary deployed evaluation interface.

The identical 10/19 aggregate result does not imply identical policies. The
seeds solved different answer sets: eight of 19 answers had mixed constrained
outcomes. As in Lab 18c, aggregate capability is more stable than individual
decisions.

### The gain is not a solved strategic policy

Turn 2 is the only state comparison fully paired across decoders. Under
answer-constrained ranking:

| Seed | Current-candidate action | Candidate-teacher match | Tier 2 teacher match | Mean open regret | Mean realized reduction | Mean candidates after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 21.1% | 0.0% | 0.0% | 1.16 bits | 2.81 bits | 7.6 |
| 45 | 36.8% | 0.0% | 0.0% | 1.05 bits | 3.32 bits | 5.2 |
| 47 | 21.1% | 0.0% | 5.3% | 1.45 bits | 2.37 bits | 10.1 |

The distinction between candidate membership and action quality matters.
Out-of-candidate guesses are legal exploratory actions, so the 21-37%
candidate-action rate is not itself a failure. Open-action regret evaluates
those guesses against the best action in the same 2,315-answer space. No Turn 2
action had zero regret, none was within 0.1 bits of optimal, and only 1/57 was
within 0.25 bits.

Observed solve gains despite zero exact Turn 2 teacher matches -> exact teacher
identity is not required for useful play -> one-hot imitation would discard
useful alternative actions -> Lab 19 should transfer relative or soft action
values.

Observed mean regret above one bit and almost no near-optimal actions -> low
teacher match is not explained by harmless ties or strategically equivalent
alternatives -> the model still needs stronger information-seeking ranking ->
Lab 19 should emphasize broad early states and score multiple actions per
state.

### The model knows where candidates are without choosing the best one

On Turn 2, current candidates had mean full-list rank percentile
0.083-0.087. The best current candidate had median rank 2, 2, and 4 across the
three seeds, and candidate probability mass was 0.27-0.29, a 28-35x lift over
uniform mass. In contrast, the candidate-only teacher's median full-list rank
was 210-254.

Observed candidate concentration and a top-four best candidate -> training has
learned a strong state-conditioned candidate signal -> the answer constraint
can expose a useful legal action near the top -> efficient constrained decoding
is justified as a real inference improvement.

Observed teacher ranks in the hundreds and candidate-restricted teacher match
near zero -> the model does not order plausible actions by information value ->
candidate grounding alone cannot supply the strategic policy -> retain
teacher-value supervision as the central Lab 19 intervention.

The Turn 2 candidate-rank percentile is weaker than the approximately 0.029
aggregate value in Lab 18c. That is consistent with Lab 18c's late-state-heavy
battery: broad deployed states are harder than the aggregate state set.

### Constrained actions produce useful information despite high regret

The constrained decoder reduced the Turn 2 candidate count from a mean of 49.4
to 7.6, 5.2, and 10.1 across the seeds. Mean realized reduction was 2.37-3.32
bits. Free generation left 21.0-39.7 candidates after Turn 2 when invalid
outputs were counted as zero-reduction actions.

Observed large deployed candidate reduction -> many non-optimal constrained
actions are still informative -> imperfect ranking can compound into game
success -> future evaluation should report action value and realized reduction,
not teacher exact-match alone.

Observed open regret of 1.05-1.45 bits alongside those reductions -> the policy
is useful but leaves substantial information on the table -> better relative
ranking should shorten trajectories and create more opportunities to close the
game.

The canonical candidate-teacher entropy gap was never negative. Although an
exploratory answer could theoretically outperform the best current candidate,
none of the selected constrained actions did so on this battery.

### Late-game closure is a second, distinct bottleneck

The constrained games ended with a mean of only 1.2-1.3 candidates, yet each
seed failed nine games. Of the 27 failures, 18 ended with exactly one candidate
and 24 ended with at most two.

Across all constrained calls made with one candidate remaining, the model chose
that candidate only 18/65 times (27.7%). It repeated an earlier guess on 21/65
calls (32.3%). Examples include repeatedly choosing `THONG` when `GHOST` was the
sole candidate, `ROOMY` when `ROUND` was known, and `DUMMY` when `DOUBT` was
known.

Observed uncertainty collapse without solution emission -> many remaining
failures are no longer search failures -> answer-list grounding does not ensure
history-conditioned completion -> Lab 19 also needs strong late-state
candidate preference, or the deployed decoder needs a separately justified
endgame rule.

This changes the simple diagnosis from "Turn 2 strategy is the only remaining
problem." Broad-state ranking is weak, but the model also fails to exploit
states that its own guesses have already resolved.

### Free-continue does not provide the intended causal decomposition

Free-continue produced exactly the same solved games as free generation for all
three seeds. It prevented formal invalid termination and increased the number
of model calls, but it did not recover a single game.

This equality is mechanically encouraged by the harness. An invalid action
leaves the history unchanged, and greedy decoding is deterministic. The next
turn therefore presents the same prompt and ordinarily produces the same
invalid action again. Free-continue measures whether unchanged-state greedy
generation spontaneously recovers; it is not a strong control that isolates
lexical validity from action quality.

Observed zero free-continue gains -> merely spending additional turns on an
unchanged prompt cannot rescue greedy generation -> the answer-constrained
intervention is necessary for these games -> do not attribute the full
constrained-minus-free-continue gain specifically to superior strategic
ranking.

The primary free-versus-constrained result remains valid. The narrower
preregistered claim that constrained versus free-continue cleanly isolates
action-quality gain is not supported by this deterministic counterfactual.

## Surprises

The largest surprise is the size and stability of the gameplay gain. The
constrained solve count was exactly 10/19 for all three seeds even though free
generation solved only three to five games and the individual constrained
answer outcomes differed. The approximately 30% state-level consistency
capability from Labs 18b and 18c was enough to more than double mean solve rate.

The second surprise is that this gain coexists with uniformly poor Turn 2
teacher behavior. Constrained play never selected the candidate-only teacher
on Turn 2, yet it reduced the candidate set by roughly 2.4-3.3 bits and solved
more games. This rejects exact teacher match as a sufficient definition of
action quality, but the high open regret also rejects the opposite story that
the chosen actions were mostly equivalent alternatives.

The strongest new failure mode is late-game closure. Most constrained failures
ended with one or two candidates, and the model selected a sole remaining
candidate on fewer than three in ten opportunities. The experiment began as a
test of early strategic ranking but uncovered a separate inability to convert
resolved state information into the final answer.

## Limitations

- The 19 answers are a fixed diagnostic battery. The 52.6% solve rate is not a
  precise estimate for the full answer distribution.
- There are only three training seeds. Their identical aggregate constrained
  solve count is encouraging but does not precisely estimate seed variance.
- Only Turn 2 is paired across decoder trajectories. Later-turn comparisons
  describe deployed behavior conditional on each decoder's earlier actions.
- Free-continue is degenerate under deterministic greedy decoding because an
  invalid action leaves the next prompt unchanged. It does not cleanly separate
  validity from strategic action quality.
- The constrained action space contains the 2,315 possible answers, not the
  larger valid-guess lexicon. It excludes some potentially optimal exploratory
  guesses.
- The open teacher optimizes one-step expected entropy under a uniform
  candidate prior. It is a useful strategic reference, not a proof of optimal
  whole-game play.
- Exact teacher match is sensitive to ties and near-ties. Open regret is the
  stronger value-based diagnostic here.
- Candidate membership is not synonymous with a good move in broad states;
  exploratory non-candidate actions can be strategically valid.
- The implementation is a diagnostic scorer, not a production decoder.
  Scoring all 2,315 answers required 17.2 minutes for this small battery,
  including the 40-state memory soak.

## Implications

1. Treat answer-constrained gameplay as the primary deployed evaluation. The
   named hypothesis from Labs 18b and 18c is confirmed: hidden state-conditioned
   ranking materially improves full games.
2. Keep exact teacher identity secondary. Lab 19 should distill soft,
   pairwise, or listwise action values so near-optimal alternatives receive
   appropriate credit.
3. Split Lab 19 coverage by uncertainty regime. Broad Turn 2 states need
   information-gain ranking; one- and two-candidate states need reliable
   completion of the inferred answer.
4. Use open-action entropy regret and realized candidate reduction as primary
   strategic metrics. Teacher exact-match alone would label all 57 Turn 2
   actions wrong while missing their substantial deployed benefit.
5. Preserve all three seeds in later gameplay evaluations. Equal solve totals
   concealed different answer-level policies.
6. Before attributing future gains to strategy rather than interface, replace
   the degenerate free-continue control with a counterfactual that changes the
   invalid-action state or samples a different valid action under an explicitly
   defined policy.
