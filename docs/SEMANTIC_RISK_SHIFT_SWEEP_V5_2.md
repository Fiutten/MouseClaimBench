# Semantic-risk shift degradation v5.2

The fixed topology-specific authorizer passed its in-population v5.1 risk lock
and final split but failed the declared OOD stress. Version 5.2 characterizes
that boundary with six prespecified severity levels.

Each level contains 100 seed bundles and uses the same seed identifiers to
support paired descriptive comparisons. Pair and adapter-control selection also
uses one shared namespace across levels. Within a level, seed bundles are the
independent units. The six risk statements use Bonferroni-adjusted one-sided
confidence. Dependence across paired levels therefore does not invalidate the
simultaneous error allocation.

The threshold remains fixed. A shift warning cannot alter a decision, restore a
certificate, or select a new threshold. The primary comparison asks whether the
warning occurs no later than the first certificate failure. One sweep cannot
validate a general-purpose detector, and monotonic behavior is descriptive.
