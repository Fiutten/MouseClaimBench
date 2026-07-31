# Q1 Execution Status

## Current Bottom Line

MouseBrainBench now has a Q1-candidate empirical package: the positive MICRONS
stratified signal is internally reproduced across a discovery cohort and two
non-overlapping CAVE hold-out subsets. It is still not a whole-brain or
causal-mechanism claim.

The most important result is now the expanded MICrONS pilot plus its
stratified follow-up:

- 2991 total co-registered digital-twin/EM units across three non-overlapping
  CAVE windows;
- 6575 total CAVE synapses;
- 5943 total unique directed connected pairs;
- Q1-scale pilot gate: passed;
- global positive structure-function result: failed;
- stratified structure-function result: positive after distance/degree/FDR
  controls, dominated by `readout_location`;
- first hold-out stratified result: positive after distance/degree/FDR controls
  on `992` additional units and `2161` synapses;
- second hold-out stratified result: positive after distance/degree/FDR controls
  on `999` additional units and `2147` synapses.

In the global aggregate test, connected pairs are more functionally similar than a random null, and more
similar than a degree-matched null, but the effect does not survive the
distance-matched null. Therefore, the aggregate result remains a stress test,
while the stratified result provides the current Q1-candidate signal.

The two hold-out results change the Q1 situation from "candidate requiring
internal reproduction" to "candidate ready for manuscript planning". The
strongest effect remains local and correlational rather than causal or
interventional.

## Step Status

| Step | Status | Result |
|---|---|---|
| Expand MICrONS with CAVE | Done | `digital_twin_properties_bcm_coreg_v4`, v1507, 1000 units |
| Add structure-function model | Done | Synapse count/size, random, distance, degree controls |
| Add stronger nulls | Done | Distance and degree matched permutation tests |
| Scale Sensorium on Mac GPU | Done | MPS works; bounded official model improves over mean in 4/5 mice but remains far below local temporal/MLP baselines |
| Publication decision table | Done | Q1 candidate supported by internally reproduced MICRONS stratification; causal claims remain blocked |

## Key Results

| Evidence block | Scale | Result | Q1 use |
|---|---:|---|---|
| MICrONS static micro-pilot | 172 units, 82 synapses | Negative/inconclusive | Stress test only |
| MICrONS expanded pilot | 1000 units, 2267 synapses | Positive vs random/degree, not distance | Not enough for positive Q1 claim |
| MICrONS stratified expanded pilot | 1000 units, 2267 synapses | 28 tests positive after FDR, mostly readout-location | Discovery cohort |
| MICrONS stratified hold-out offset1000 | 992 units, 2161 synapses | 30 tests positive after FDR, readout-location internally reproduced | Q1 candidate evidence |
| MICrONS stratified hold-out offset2000 | 999 units, 2147 synapses | 25 tests positive after FDR, readout-location internally reproduced | Q1 candidate evidence strengthened |
| Sensorium official bounded | 5 mice | Runs on MPS, positive vs mean in 4/5, weak absolute correlation | Integration evidence, not Q1/SOTA |
| Dynamic Sensorium MLP | 5 mice | Stronger prediction | Predictive baseline, not mechanistic |

## Next Q1-Capable Moves

1. Convert the internally reproduced MICRONS result into a manuscript-grade benchmark
   section with claims limited to local structure-function association.
2. Keep the global MICrONS result negative unless the all-pairs distance-matched
   effect turns positive.
3. Treat non-readout functional positives as exploratory unless they reproduce in
   fixed hold-outs.
4. Obtain or train a stronger official Sensorium baseline. The current local
   official models are integration controls, not competitive baselines.
5. Frame the paper around the central methodological claim: MouseBrainBench
   prevents predictive success from being over-interpreted as mechanistic
   identifiability.

## Critical Interpretation

The project is not failing; it is doing what a serious benchmark should do. It
has already rejected weak mechanistic claims that would be tempting to overstate.
For Q1, the remaining technical work is no longer to find a positive signal,
but to harden the positive signal:

- report exact hold-out protocol and CAVE provenance: done;
- add effect-size confidence intervals or bootstrap stability: done;
- keep causal, whole-brain, and Sensorium-SOTA claims explicitly blocked: done.

The current next step is manuscript production plus one external-facing
external replication or benchmark comparison if time permits, not open-ended
exploratory signal search.
