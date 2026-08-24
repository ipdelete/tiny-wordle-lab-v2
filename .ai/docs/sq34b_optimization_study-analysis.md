# SQ34b optimization study analysis

## Main conclusion

`lower-step` advances. The other two arms do not.

Cutting the learning rate from `5e-5` to `1e-5`, with KL still at `0.02`,
moved this LoRA through six rounds without an adverse candidate-mass path.
Median mass rose from 0.177 to 0.203. Stochastic trie play finished 53/152,
inside the 8-solve veto. The original recipe reprinted SQ34 and hit the mass
floor again. Raising KL to `0.10` at the original step size made the collapse
faster, not slower.

SQ34c may use `lower-step` as its parent. A replicated Lab 34 is still not
justified. Clip was 0.0 on every arm. This study did not test GRPO clipping.

## Findings

### The contemporaneous control reprinted SQ34

`baseline-recipe` used the same knobs, seeds, and warmup as SQ34. It produced
the same 43 updates, the same LoRA Δ of 0.454 (2.49%), the same final-round KL
of 0.706, the same mass path, the same 10/19 / 5/19 / 53/152, and the same
final adapter digest `96af220d…`.

Observed exact reprint -> run-to-run variance is not the story -> the SQ34
failure is a property of `5e-5 / 0.02`, not of one noisy sample.

Training time was 11.5 minutes. The reset soak peaked at 6.82 GiB and
plateaued. The watched process kept at least 401 GiB available and used no
swap.

### `lower-step` moved, and mass went up

38 mixed-group updates. Relative LoRA Δ was 0.479%, just under the 0.5% bar.
Final-round mean KL was 0.151, which is how the arm cleared the exposure
rule. That is a real update, not a no-op.

Candidate mass by round: 0.180, 0.191, 0.200, 0.202, 0.203, 0.203. The hard
floor is 0.124. The advance floor is 0.150. Nothing here is a delayed copy of
the SQ34 decline.

Stochastic play was 53/152, the same count as the failed recipe and 3 below
the 56/152 baseline. The veto is 48. Greedy trie stayed 7/19. Deterministic
fell to 9/19. Nineteen games still cannot pick a winner, and they did not.

Observed rising mass plus nonzero KL -> a smaller step can keep the full-list
ranking alive through the horizon SQ34 failed -> this recipe is the parent
for a clipping study, not proof that play improved.

### Stronger KL did not save the original step

`stronger-kl` matched the control for two rounds, then fell through the floor
at update 25. Final mass was 0.110. SQ34 needed 43 updates to get to 0.122.
Five times the KL coefficient, same `5e-5` step, worse ranking, earlier stop.

Deterministic 11/19 and greedy 8/19 are recorded. They do not advance the
arm. The mass guard fired. Shopping that 11/19 would repeat the SQ34 round-3
mistake.

Observed earlier collapse under stronger KL -> the step size, not a weak
reference term, was the thing that wrecked mass -> do not spend another arm
on KL at `5e-5`.

### Clip still never fired

Mean clipped fraction was 0.0 on every arm, including the one that advanced.
One optimizer step per fresh group still leaves the word-level ratio at 1.
A safe `lower-step` parent still has not tested the clip term. That is what
SQ34c is for.

## Surprises

The control bit-matched SQ34, including the adapter digest. I expected MPS
noise. There was none that reached these metrics.

`lower-step` mass rose. I was looking for a slower decline. I did not expect
the ranking to improve.

KL on `lower-step` stayed tiny through round 4 (0.029) and jumped to 0.151 in
round 5. The mass path was already flat by then. The late KL spike is how the
arm cleared exposure. It is not how mass was saved.

`stronger-kl` looking like the control, then dying sooner, kills the idea that
SQ34 failed because the reference term was too weak.

## Limitations

Six rounds at `1e-5` is still a short horizon. Mass is stable here. It has
not been shown stable at 43 updates of this size, or at a second seed.

Relative Δ of 0.479% is small. The arm is exposed because of KL, not because
it moved like SQ34. A later study that needs a larger LoRA change may not
keep this mass path.

Stochastic 53/152 is compatible with baseline. It is also identical to the
failed recipe. This advance is a safety result, not a play result.

The graded decoder moved 10/19 -> 9/19 on the advancing arm. That is noise,
and it is not an improvement.

No arm multi-passed a frozen group. Clip remains untested.

## Implications

Next experiment is SQ34c, parent `lower-step` (`1e-5 / 0.02`). Give clipping
a chance to act by scoring a group more than once under a moving policy.
Keep the same answers, anchors, dual-decoder split, and the 0.85 mass bar.

Do not replicate Lab 34. Do not retune KL at `5e-5`. Do not select on
deterministic 11/19.

If SQ34c keeps mass and still cannot improve the trained policy, change the
objective. That question is not answered yet.
