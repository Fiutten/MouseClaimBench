# Prospective DANDI contrast profile-v2.1 freeze

The protocol uses only published metadata for DANDI:000039 version
`0.230223.1216`: 32 mouse subjects, 100 NWB assets, declared optical physiology
and behavior, and 22.6 GB total size. No asset catalog, NWB response value, or
source-paper numerical result was inspected before this freeze.

One behavior-plus-ophysiology asset per subject will be selected by a fixed
lexicographic path rule. The trial-level population-response endpoint, feature
map, ridge model, temporal split, comparator, mouse-level bootstrap, thresholds,
and bounded claim interpretation are fixed in
`configs/benchmarks/dandi_contrast_profile_v2_1.yaml`.

This is an evaluation of prospective artifact authorization. Contrast tuning is
already the subject of the source resource and is not presented as a new
neuroscience discovery.
