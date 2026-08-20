# Lab 18b analysis: constrained ranking probe

## Main conclusion

Lab 18b found a real decoding problem and a real learned policy underneath it.
Imposing the answer lexicon raised usable actions from 14.5% to 30.3% for
B-structured and from 16.0% to 30.0% for G-structured. The base model reached
only 0.16% through the same interface, so the lexicon alone did not create the
gain. The adapters learned to move state-consistent words near the top of the
answer ranking.

This does not rescue Dataset G. B and G are effectively tied at Tier 1, while B
is better at choosing the teacher's preferred word from the consistent
candidates. The evidence supports using constrained ranking as the primary
interface for the seed replication in Lab 18c. It does not support another data
distribution experiment before measuring seed variance.

## Findings

### Lexicon grounding hid about half of the usable policy

B-structured rose from 90/620 usable actions under free generation to 188/620
under answer-list ranking, a gain of 15.8 percentage points with a paired 95% CI
of 12.6 to 19.0 points. There were 108 states where Tier 1 recovered a usable
action that Tier 0 missed, against 10 losses, with exact p = 6.5e-22.

G-structured rose from 99/620 to 186/620, a gain of 14.0 points with a paired
95% CI of 10.8 to 17.3 points. It gained 101 states and lost 14, with exact
p = 2.0e-17.

Observed gain -> free generation discards substantial learned signal -> Labs
15 through 18 understated policy quality -> Lab 18c should report constrained
ranking as its primary battery readout and retain free generation as a
secondary continuity metric.

The constraint is not a complete fix. Both adapters remain unusable on about
70% of states at Tier 1. Grounding was one bottleneck, not the only bottleneck.

### Training, not the answer list, produced the state-conditioned ranking

The base model produced a usable Tier 1 action on 1/620 states. B-structured
produced 188 and G-structured produced 186. Relative to base, the paired gains
were 30.2 points for B, 95% CI 26.6 to 33.9, and 29.8 points for G, 95% CI 26.3
to 33.4.

The full rankings tell the same story. The mean percentile of consistent
candidates was 0.433 for base, close to the 0.5 state-blind reference, versus
0.028 for B and 0.032 for G. The best consistent candidate had median full-list
rank 390.5 for base, 3 for B, and 4 for G.

Observed separation from base -> the adapters use the supplied feedback to
promote legal words -> the constrained result is not a lexical prior artifact
-> seed replication must test the trained adapters, not only the decoding
constraint.

The argmax diversity check rejects the state-blind explanation as well. Base
used only 12 distinct Tier 1 words and chose `GREEN` on 39.7% of states. B used
342 distinct words with a 1.9% most-common share; G used 348 with a 1.8% share.

### The entropy policy exists, but it is much weaker than consistency filtering

The teacher word's median rank among all 2,315 answers was 9 for B and 10 for G,
compared with 861 for base. B placed the teacher in its top 10 on 53.4% of
states and top 50 on 79.2%; G reached 50.2% and 78.5%.

When ranking only consistent candidates, B matched the teacher on 356/620
states, or 57.4%, against a per-state chance expectation of 52.2%. Its
5.2-point gain over base had a paired 95% CI of 1.5 to 8.9 points and exact
p = 0.007. G reached 338/620, or 54.5%; its 2.3-point gain over base had a
95% CI of -1.3 to 5.8 and p = 0.231.

Observed strong candidate promotion but small Tier 2 lift -> the adapters
learned consistency filtering much more strongly than the teacher's entropy
preference -> changing state allocation alone is unlikely to solve the
remaining policy error -> after seed replication, any next intervention should
target ranking among legal candidates directly.

This Tier 2 result needs care because the battery is late-state heavy. There
are 288 states with only one or two candidates, which raises the aggregate
chance expectation to 52.2%. In the larger 3-10 candidate bucket, B reached
32.7% teacher match against 24.4% chance, while G reached 26.5%.

### Dataset G does not improve the constrained policy

B and G are tied on Tier 1 usability: 30.3% versus 30.0%. A paired comparison
derived from the saved state-level results gives G minus B = -0.3 points,
95% bootstrap CI -3.4 to +2.7, with 50 B-only and 48 G-only successes.

At Tier 2, B leads G by 2.9 points. The paired flips are 40 B-only against 22
G-only, exact p = 0.030. B also has the better mean candidate rank percentile,
0.0283 versus 0.0318. The difference is small, but every main ranking readout is
either tied or favors B.

Observed null or negative G differences -> whole-game-derived visitation did
not improve the policy even after removing free-generation noise -> Lab 18's
null was not merely an evaluation artifact -> use B-structured as the arm for
the first Lab 18c seed replication.

### Token-length bias does not explain the result

Switching from summed string log-likelihood to length-normalized likelihood
moved Tier 1 usability from 30.3% to 31.1% for B and from 30.0% to 30.5% for G.
Teacher match moved from 16.8% to 17.1% for B and from 16.1% to 16.5% for G.

Observed changes under one percentage point -> EOS and token-count effects do
not drive the headline result -> keep summed string likelihood as the
preregistered primary rule.

## Surprises

The strongest surprise is that the earlier free-generation scores were both too
pessimistic and directionally useful. Constrained ranking doubles usable
actions, but B and G remain tied. The decoder hid policy signal without hiding a
Dataset G win.

The base control also resolves an important ambiguity. Restricting output to
real words does almost nothing for the untrained model. The roughly 30% Tier 1
rate comes from adapter training, not from making every possible action
well-formed.

Tier 2 weakens the simple story that the model only needed a legal decoder. B
has measurable teacher-policy signal, but G is statistically indistinguishable
from the base model once the candidate set is supplied. The remaining problem
is choosing well among legal actions.

## Limitations

- This is one training seed per adapter. Lab 18c is still required before
  treating the B-over-G Tier 2 difference as stable.
- Tier 1 ranks only the 2,315 answer words, not the larger valid-guess lexicon.
  It therefore evaluates answer-constrained play rather than every strategic
  Wordle action.
- The state battery does not establish solve rate. A constrained decoder must
  be evaluated through complete games before making gameplay claims.
- The battery contains many one- and two-candidate states. Aggregate Tier 2
  accuracy overstates performance on broad early-game decisions.
- The B-versus-G paired intervals in this analysis were derived after execution
  from the persisted `tier-results.csv`; they were not preregistered as the
  primary Lab 18b comparison.
- Full answer-list scoring took about 35 minutes per model on this hardware.
  It is a diagnostic reference implementation, not yet an efficient decoder.

## Implications

1. Lab 18c should train two additional B-structured seeds and evaluate all
   three seeds with Tier 1 constrained ranking as the primary metric. Free
   generation remains secondary so earlier labs stay comparable.
2. The seed study should include candidate rank percentile and Tier 2 teacher
   match. Tier 1 alone would miss variance in the entropy policy.
3. Do not spend more training budget on Dataset G before replication. Its
   constrained ranking is no better than B and its Tier 2 result is worse.
4. If B's constrained gains replicate, build an efficient answer-constrained
   decoder and rerun full gameplay under the same termination rules for every
   model.
5. If Tier 2 remains only slightly above chance across seeds, the next training
   experiment should supervise relative ranking among consistent candidates
   rather than change the visitation distribution again.
