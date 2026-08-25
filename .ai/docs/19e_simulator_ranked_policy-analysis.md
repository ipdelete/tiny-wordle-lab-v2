# Lab 19e analysis: simulator-ranked policy

## Main conclusion

Do not advance this recipe.

The entropy arm hit the candidate-mass stop at the first checkpoint, after 32
updates. Mass fell in every Lab 20 regime. Dev entropy regret got worse than
both the incumbent and the matched preservation control. The treatment did
solve 17 of 19 games with singleton closure, but that number hides damaged raw
ranking.

The control stayed close to the incumbent. That matters. Continued training,
weight decay, online support refresh, and truncated support KL did not cause
the collapse by themselves. The pairwise entropy loss did.

## Findings

### Gate B passed

The differentiable scorer matched indexed full-list scores to `9.92e-5`. The
full-list path reproduced the frozen Lab 18d vector exactly.

The 40-iteration full-list soak plateaued at 54.77 GiB. The alternating
refresh and training soak plateaued at 54.89 GiB. Both had 0.0 GiB late creep.
Memguard kept at least 400 GiB available during Gate B.

Observed scorer agreement and flat memory means the Gate C result comes from
the declared objective. It is not a decoder mismatch or an allocator failure.

### The matched control stayed intact

Both arms began from the same adapter digest. Each ran the same 32 states with
the same optimizer, schedule, support construction, online refresh cadence,
and truncated support KL.

The preservation control moved by 0.334%. Its overall candidate-mass ratio was
0.996. Per-regime ratios were 0.851, 1.115, 1.147, and 0.991. No rule tripped.
Dev entropy regret improved slightly, from 0.784 bits for the incumbent to
0.774 bits. Gameplay stayed at 15/19.

Observed stability in the control means the experiment isolated the damaging
term. The entropy arm did not fail because AdamW touched the adapter.

### Pairwise entropy training destroyed candidate ranking

The entropy arm moved by 1.030% in 32 updates. Its candidate-mass ratios at the
first checkpoint were:

| Regime | Mass ratio |
| --- | ---: |
| singleton | 0.094 |
| 2 candidates | 0.373 |
| 3 to 10 candidates | 0.186 |
| 11 or more candidates | 0.451 |

The floor was 0.85 in every regime. All four stops fired.

The raw singleton answer moved from median rank 1 on the anchor suite to rank
8. In gameplay states, the median raw singleton rank moved from 3 to 12 and
the top-1 rate fell from 45.5% to 28.6%.

Observed broad mass loss and worse singleton rank means this is another
full-list redistribution, not a narrow trade between exploration and solving.
The hard stop did its job.

### The treatment did not learn the held-out teacher

Mean dev entropy regret was:

| Policy | Mean regret |
| --- | ---: |
| preservation control | 0.774 bits |
| incumbent | 0.784 bits |
| entropy ranking | 0.836 bits |

The treatment was 0.052 bits worse than the incumbent and 0.062 bits worse
than the control. The required improvement was 0.10 bits in the other
direction.

Training loss did fall. Pairwise loss ended at 2.41 after reaching as high as
14.15, while truncated support KL rose to 0.737 nats. The model changed the
supported comparisons, but those updates did not transfer to unseen dev
states.

Observed optimization progress with worse dev regret means "the model did not
train" is not an explanation. The local pairwise objective did not generalize.

### The 17/19 game score is not a pass

With deterministic singleton closure, the incumbent and control each solved
15/19. The entropy arm solved 17/19.

That looks encouraging until the raw scores are inspected. The treatment
reached 14 singleton closures, then code supplied the answer. At those same
states, the model ranked the answer at median position 12. The environment
rescued behavior the model had lost.

Observed more solved games beside worse dev regret, collapsed candidate mass,
and worse raw singleton rank means the 17/19 is path luck plus symbolic
closure. It cannot override the preregistered guards.

## Surprises

The symbolic teacher is excellent. Lab 19f solved all 2,315 answers with it.
The model still got worse when trained on its pairwise preferences.

The collapse was fast. SQ34 needed 43 updates to cross its overall mass floor.
This arm failed all four regime floors after 32.

The preservation control improved dev regret a little without changing play.
That is useful evidence. The support refresh and truncated KL are not inert,
and they are not the source of the treatment collapse.

## Limitations

The hard stop ended both arms after 32 of 256 planned updates. This is a valid
rejection of the frozen recipe, not a comparison at the intended horizon.

Only seed 45 ran. A second seed would not repair a four-regime mass collapse
and is not justified for this recipe.

The pairwise loss treated every unequal entropy pair equally. It did not scale
updates by utility gap or distinguish actions that were both poor. That is one
possible cause, not an established diagnosis.

The truncated KL covered at most 64 actions per state. It protected the
control but did not restrain the stronger treatment update. It was never a
full-list preservation loss.

The 19-game battery is small. Lab 19f's symbolic universe result does not give
the learned model a population guarantee.

## Implications

Do not rerun this recipe longer. Do not weaken the mass stops to keep it alive.

The next experiment should reduce the entropy update itself before changing
the teacher. The smallest useful comparison is a treatment with much lower
pairwise-loss weight against the same preservation control, with the first
drift check earlier than 32 updates. Another option is to train only on strict
open-versus-candidate disagreements, the 111 state types where Lab 19f found a
real entropy advantage, instead of all unequal pairs.

The question is now narrower:

> Can the model learn a small number of consequential exploratory preferences
> without moving the rest of the answer ranking?

The current full-support pairwise recipe says no.
