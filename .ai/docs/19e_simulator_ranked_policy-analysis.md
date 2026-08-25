# Lab 19e analysis: no-model validation

## Main conclusion

Gate A now preregisters the matched `preservation-control` and
`entropy-ranking` experiment without loading a model. The treatment objective
is support-invariant pairwise ranking over unequal one-ply entropy utilities.
The control task term is exactly zero. Both arms otherwise share the frozen
training, drift, checkpoint, and evaluation contract.

This execution provides no evidence that either arm improves model behavior.
Gates B and C were disabled.

## Findings

The selection pass excluded Lab 20 anchors before sampling. All 20 anchor keys
are disjoint from both selected pools. Seven eligible dev states were anchors
and were removed; no eligible train state matched an anchor. The resulting
sample has 128 train states with 712 current candidates and 64 dev states with
367. The 3 to 16 candidate bound still retains 90.6% of broad train states and
90.2% of broad dev states.

The minimum Gate A support is the deduplicated union of current candidates and
the entropy teacher's top 16 actions. It spans 18 to 31 actions in train and 16
to 27 in dev. Train supports average 98.04 valid unequal-utility pairs and
120.26 tied pairs. Dev supports average 93.42 valid pairs and 121.23 tied
pairs. Mean nonzero utility gaps are 0.553 bits in train and 0.526 bits in dev.

Sixteen train states and seven dev states have no valid pair after
nine-decimal utility rounding. Every pair is tied in those supports. The
treatment therefore has a zero task term on those updates, while
`truncated_support_kl` still applies. This is preferable to fabricating
ordering information where the simulator supplies none.

The teacher remains diverse after anchor exclusion. Its state-hash tie order
produces 121 distinct top actions and 1,217 distinct top-16 words across train,
plus 61 top actions and 759 top-16 words across dev. The teacher's first action
is a current candidate in only 3 of 128 train states and 4 of 64 dev states.
The top-16 lists average 15.64 noncandidates in train and 15.36 in dev, so the
open-teacher category still contributes actions that candidate-only support
would miss.

The preregistration SHA-256 is
`0b5686401897aeed6d0bc81c7c5abad8bcdb98e9886cd2e297fe1fe84b103b05`.
The target hashes are
`6fded49de16058843a0e1772a74f5f97ddc7888bed938eb0233265db3837c5d7`
for train and
`0d6a95eb0caba58ac8c4f43f7a3be1b9dfefd121e36d2ffd4bf6fa8ccfefb8ef`
for dev.

## Surprises

Removing the seven leaked dev anchors changed the selected dev pool and its
target hash, as it should. The train pool did not change because none of the
Lab 20 anchors survived the train source and reserved-answer filters.

The zero-valid-pair states are the useful adversarial case in this audit. A
pairwise objective removes support normalization, but it cannot manufacture a
preference when all frozen utilities tie. Recording zero valid pairs makes
that limitation visible in training history.

## Limitations

Gate A cannot audit the incumbent-top-16 or current-top-16 categories because
those require model scores. It therefore cannot establish final support sizes
or scorer equivalence.

No adapter, optimizer, full-list scorer, drift checkpoint, memory soak, dev
model evaluation, or gameplay loop ran. The matched-arm isolation assertions,
per-regime drift stops, raw singleton ranks, causal dev contrast, and solve
criteria remain untested until the guarded model run.

## Implications

Any model run must use the new preregistration hash. Gate B must first compare
`score_encoded_actions` directly with indexed `score_all_words` results for
multiple prompts, every action-token-length bucket, and the full 64-action
support. Both 40-iteration soaks must pass under the 128 GiB cap and memguard
before Gate C runs.

If Gate C completes, the primary result is treatment regret minus control
regret, which is the treatment-minus-control change from incumbent. A negative
value favors treatment. `entropy-ranking` advances only if it improves
by at least 0.10 bits against the incumbent and by at least 0.10 bits against
the control, passes treatment drift and solve guards, and completes without a
treatment hard stop. The control is diagnostic and cannot be selected as the
trained recipe.
