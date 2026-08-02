# CausalRivers v5 transport protocol

CausalRivers provides real discharge time series and directed river-network
references. The topology is useful for testing transport of a fixed
topology-specific claim gate. It is not interventional causal ground truth.

The adapter fixes 40 graph edges and 40 nonedges per region using identifiers
and SHA-256 ordering only. Historical series are reduced to 10,000 uniformly
spaced rows. Failed or sparse pairs are recorded and are not replaced after
their values are inspected.

The Elbe analysis evaluates one fixed pair set in the long historical eastern
Germany series and in the 2024 flood series. This creates a matched shift audit
at the station level. Shared stations and network paths induce dependence, so
pair-level confidence intervals are prohibited.

There are only three top-level geographical clusters: Bavaria, eastern Germany,
and the Elbe flood subset. The result is therefore descriptive external
transport evidence. It cannot establish the exact seed-population certificate
used for TimeGraph and cannot be used to restore the failed synthetic OOD test.
