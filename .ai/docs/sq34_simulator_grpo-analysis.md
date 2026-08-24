# SQ34 simulator GRPO analysis

## Main conclusion

SQ34 did not pass. The run proved that sparse simulator reward can move this
LoRA within the 512-episode cap, but the movement did not improve either policy
we care about. It also pushed the Lab 20 anchor candidate mass through the
preregistered safety floor, which stopped training after round 6.

The graded deterministic decoder finished at 10/19, unchanged from the seed 45
baseline. The stochastic trie policy finished at 53/152 after starting at
56/152, and greedy trie play fell from 7/19 to 5/19. This was not an
under-training null. The adapter changed by an L2 norm of 0.454, or 2.49% of its
initial norm, after 43 effective updates, and its final-round mean KL from the
reference was 0.706.

A replicated Lab 34 is not worth running with this recipe. The next useful
experiment is a smaller optimization study that holds the traces and evaluation
fixed while reducing policy movement or strengthening the reference constraint.
The question to answer first is whether candidate mass can be kept above the
anchor floor without making the update too small to measure.

## Findings

### The implementation gates passed

The two seed 45 adapters produced identical scores before training. The
teacher-forced scorer matched the sequential scorer exactly, with a worst
per-token difference of 0.0. The initial sampled-group ratio check was exact,
and the worst ratio-identity gap during training was 7.82e-5, below the
preregistered 1e-4 tolerance.

These checks rule out adapter switching, scorer mismatch, stale behavior
metadata, prompt rendering, and dropout as explanations for the outcome.
The result reflects the intended update, not a broken ratio calculation.

All four memory soaks plateaued:

| operation | peak driver memory | late creep | final range |
| --- | ---: | ---: | ---: |
| sampling | 2.67 GiB | 0.001 GiB | 0.016 GiB |
| reference scoring | 2.65 GiB | 0.000 GiB | 0.000 GiB |
| retained-gradient scoring | 6.31 GiB | 0.000 GiB | 0.000 GiB |
| full training step | 6.93 GiB | 0.000 GiB | 0.000 GiB |

The complete watched process kept at least 353 GiB of system memory available
and used no swap. Memory is not a limiting factor for the next experiment.

### Training produced a real and growing update

The run sampled 384 episodes in 96 groups before the stop. Forty-three groups,
44.8%, had mixed outcomes and produced updates. The LoRA delta grew each round:
0.089, 0.191, 0.298, 0.356, 0.404, then 0.454. Mean KL from the reference rose
from 0.015 in round 0 to 0.844 in round 4 before ending at 0.706.

Observed movement -> the run had enough optimization pressure to test the
hypothesis -> the unchanged held-out result cannot be dismissed as "the model
did not move" -> another run needs a different update regime, not simply more
episodes with the same settings.

The realized mixed-group fraction ranged from 0.313 to 0.563. That was enough
to generate 43 updates, close to the preregistered expectation. Sparse reward
did provide signal.

### The GRPO clip never engaged

The clipped fraction was 0.0 in every round. Each group was scored while the
current policy still matched the behavior policy, then received one optimizer
step. The ratio therefore started at one, and no second pass over that behavior
batch exposed the update to the clipping bounds.

Observed zero clipping -> the PPO-style clip placed no effective limit on these
updates -> the 5e-5 learning rate and 0.02 KL coefficient controlled movement
almost entirely -> a follow-up should treat learning rate and KL strength as
the main safety controls rather than expecting clipping to stop drift.

### Candidate mass deteriorated until the safety guard fired

Median anchor candidate mass began at 0.1766. It moved through 0.1911, 0.1585,
0.1409, 0.1258, 0.1349, and finally 0.1219. The preregistered floor was 0.1236,
so round 5 crossed it and stopped the run.

The final value is only 0.0017 below the floor, but the direction is not a
last-round accident. Four of the last five checkpoints were below the baseline,
and the overall decline was 31.0%. Median teacher rank also worsened from 6.5
to 10.0, while singleton median rank moved from 1 to 2. Those ranking changes
stayed within their broader stop limits, so candidate mass was the first guard
to catch the degradation.

Observed falling candidate mass -> the model assigned less probability to
feedback-consistent answers -> the decoder had less usable state information
even while optimizing game reward -> the next optimization study should plot
candidate mass against KL and parameter delta at every short checkpoint and
stop before running another full curriculum replication.

### Neither trained nor graded play improved

The deterministic decoder stayed at 10/19. It gained ROUND and SHEEP but lost
DOUBT and BANAL. With two gains and two losses, McNemar's exact p-value was 1.0.
The evaluation could not call any change significant unless at least six
discordant answers all moved in one direction.

The stochastic trie policy moved from 56/152 to 53/152. Its Wilson interval
moved from 0.296-0.448 to 0.278-0.427, with heavy overlap. Greedy trie play
fell from 7/19 to 5/19, with two gains and four losses and an exact p-value of
0.6875.

Observed no deterministic gain and small trie regressions -> the update did not
improve the policy it trained or the decoder used for grading -> a replicated
Lab 34 would currently spend more compute confirming a negative result -> first
change the optimization constraint and rerun this bounded SQ34 test.

### The round-3 12/19 result was transient

The mid-run deterministic evaluation reached 12/19 after round 3, while the
stochastic trie result was already down to 49/152 from 56/152. By the final
checkpoint the deterministic count returned to 10/19.

Observed temporary graded improvement alongside a worse trained policy -> the
12/19 checkpoint is more consistent with decoder sensitivity and a small
19-answer sample than with broad policy improvement -> it must not be selected
after the fact -> a future study should keep the same fixed checkpoint schedule
and require movement in both the trained-policy and graded-policy measures.

The decoder agreement on the 20 anchor states rose from 0.15 to 0.40. This does
not rescue the result. The trained policy regressed, and the anchor state suite
is not the Lab 18d state suite on which SQ31 measured 0.4545, so those agreement
rates cannot be compared across notebooks.

## Surprises

The biggest surprise is that the policy moved substantially without the
clipping statistic ever leaving zero. The nominal GRPO clip was not the active
safety mechanism in this one-step-per-group design.

The deterministic decoder briefly reached 12/19 even as stochastic trie play
fell. That weakens the hypothesis that improving one decoder will naturally
transfer to the other. It also shows how easily a 19-answer checkpoint can look
promising by chance.

The anchor guard fired before the diversity guards. Mean action surprisal fell
from 3.089 to 2.673, only 13.5%, and repeat rate stayed between 2.6% and 6.8%,
well below its 25% stop threshold. The policy did not collapse into repetitive
play. It lost candidate mass first.

## Limitations

Only one optimizer seed, one learning rate, and one KL coefficient were tested.
This result rejects this recipe, not simulator GRPO in general.

The final checkpoint is the safety-stop checkpoint. The experiment correctly
did not select the transient 12/19 round-3 checkpoint, but the available data
cannot estimate how often a fixed round-3 evaluation would reproduce that
count.

The 19-answer deterministic evaluation is underpowered for modest effects.
The 152 stochastic games provide more trials, but eight games on each of 19
answers still mix answer difficulty with sampling variance.

The per-round solve rates use different training answers each round. Their
0.234-0.422 range is not a learning curve and should not be read as one.

## Implications

Do not launch a replicated Lab 34 from the round-3 or round-5 checkpoint.
Neither checkpoint has evidence of improvement in the trained policy, and the
run crossed its safety guard.

Run a short SQ34 optimization study next. Keep the answer order, seeds,
reference adapter, trace format, anchors, and dual-decoder evaluation fixed.
Compare lower policy movement against a stronger reference constraint, and
measure candidate mass, KL, parameter delta, stochastic trie performance, and
the deterministic decoder at the same fixed checkpoints. Advance to another
512-episode run only if the candidate-mass trajectory stays inside the guard
while the LoRA delta remains measurable.

The saved traces and their verified sidecars make the present run reproducible.
They also give the next study exact episode lengths, rewards, and behavior
checkpoints for diagnosing which groups produced the largest KL and
candidate-mass changes.
