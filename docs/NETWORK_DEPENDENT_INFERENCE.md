# Network-dependent inference for MICRONS

The fixed MICRONS endpoint uses almost one million directed candidate pairs per
cohort. Those rows are not independent because each neuron occurs in many
pairs. Treating them as ordinary independent observations can substantially
understate uncertainty.

The primary covariance follows the directed-dyad extension of the sandwich
estimator by Aronow, Samii, and Assenova (2015, DOI
`10.1093/pan/mpv018`). It permits arbitrary covariance between observations
whose directed pairs share either member. The implementation constructs the
score meat directly and is tested against an explicit pairwise sum. Its working
independence assumption applies only to pairs with no shared neuron.

The corroborating analysis follows the Freedman--Lane MRQAP logic evaluated by
Dekker, Krackhardt, and Snijders (2007, DOI
`10.1007/s11336-007-9016-1`). The outcome is first regressed on the fixed
controls. A common random relabeling is then applied to both row and column
indices of the residual array. This preserves the directed matrix structure
better than shuffling individual pair rows. The test is still conditional on
exchangeability of the reduced-model residual array. Cell-type and spatial
heterogeneity can violate that assumption, so this p-value is corroborating
rather than a replacement for scientific judgment.

Both analyses estimate an association conditional on log distance, squared log
distance, pre- and post-synaptic degree, and coarse cell-type agreement. They do
not identify a causal synaptic effect. A positive outcome remains local to the
co-registered MICRONS windows and is not an independent biological replication.

