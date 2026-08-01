# External Artifact Provenance

The second-paper repository stores only lightweight summary artifacts required
to reproduce the manuscript tables. Raw MICRONS data remain excluded.

| Artifact | Source repository | Recorded analysis revision | SHA-256 |
|---|---|---|---|
| `microns_primary_robustness/summary.json` | `Fiutten/Mouse-brain` | `b3c970d` series, recorded inside the artifact | `6ed032013fd059f3084b89cda3bc60534b750c16f19d510369563c75443697b5` |
| `microns_q1_package/summary.json` | `Fiutten/Mouse-brain` | `7f529` series, recorded inside the artifact | `c4a6561f1574ec2fc361e3d358e180555fe23dec50eaf836de9bb79ccc11283b` |

The summaries are copied byte-for-byte. Numerical fields must not be edited in
this repository. Regeneration requires the source repository, its documented
CAVE access, and the corresponding MICRONS resource version.
