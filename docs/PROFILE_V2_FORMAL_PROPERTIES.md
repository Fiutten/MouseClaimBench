# Profile v2 formal properties

For a declared claim `c`, let `Q(c)` be its required evidence blocks and let
`s(e)` be the effective state of block `e` after profile-schema validation. The
authorization rule is

`A(c) = 1` if and only if `s(e) = passed` for every `e` in `Q(c)`.

The deficit trace is the exact complement

`D(c) = {e in Q(c) : s(e) != passed}`.

These definitions give five properties relative to the versioned profile:

1. Soundness: authorization implies that every required block passed.
2. Completeness: if every required block passed, the claim is authorized.
3. Trace identity: the returned deficits equal `D(c)`.
4. Degradation monotonicity: replacing a passed requirement by a non-passed
   state cannot create authorization.
5. Irrelevance and order invariance: reordering facts or adding a fact outside
   `Q(c)` cannot alter the decision or its deficits.

The first four follow directly from the authorization and deficit definitions.
The executable property benchmark checks that the Python implementation
preserves them over deterministic generated packages. This is not a proof that
`Q(c)` is a complete or scientifically valid description of the claim.
