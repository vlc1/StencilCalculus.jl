# Plan: `AbstractScalar` + unified `Constant` term

## Context

`StencilCalculus` is the middle package of a trilogy (`StencilCore` →
`StencilCalculus` → `StencilAssembly`). Today **`Scalar` and `Const` are
subtypes of `AbstractTerm`**, which is conceptually wrong: an `AbstractTerm{T}`
is a *dimension-/size-less array-like collection* whose cells hold values of
type `T`, whereas a `Scalar`/`Const` is a *single* value. Treating a single
value as an array-like term is what produced the bug fixed in `7e1b3ca`
(`derivative(-,…) = Const(-1)` — a data value — instead of the array identity
`-One{eltype}()`).

The fix introduces an `AbstractScalar{T}` hierarchy that is **sibling to, not a
subtype of**, `AbstractTerm{T}`. Per the user's refinement, the literal-constant
leaf `Const` and the identities `One`/`Zero` are **unified into a single
`Constant{T,V} <: AbstractTerm{T}`** node — "a `Ones` broadcast of either a
literal value (`V === T`) or a `Scalar` (`V <: Scalar`)". `Scalar` is retained
(now an `AbstractScalar`); `Const`, `One`, `Zero` are removed.

The payoff: scalars are wrapped in `Constant` at the operator boundary, so they
never enter `Term.args` and `simplify` stays a pure `AbstractTerm → AbstractTerm`
rewriter. There is no "scalar inside a term" to thread through the CAS.

## New type model

**`StencilCore/src/scalars.jl`** (new file; included after `term.jl`):

```julia
abstract type AbstractScalar{T} end
Base.eltype(::Type{<:AbstractScalar{T}}) where {T} = T
Base.eltype(s::AbstractScalar) = eltype(typeof(s))

struct Scalar{S, T} <: AbstractScalar{T}          # named broadcast parameter
    Scalar{S,T}() where {S,T} = (_assert_concrete(:Scalar, T); new{S,T}())
end
Scalar{S}() where {S} = Scalar{S, Float64}()

struct Constant{T, V} <: AbstractTerm{T}          # broadcast-to-grid term
    val::V
end
Constant(val::T) where {T} = (_assert_concrete(:Constant, T); Constant{T,T}(val))
Constant(sc::Scalar{S,T}) where {S,T} = Constant{T, typeof(sc)}(sc)

# Shift-invariance (position-independent): StaticShift lives in Core.
Base.getindex(s::AbstractScalar) = s
Base.getindex(s::AbstractScalar, ::StaticShift) = s
Base.getindex(c::Constant) = c
Base.getindex(c::Constant, ::StaticShift) = c     # subsumes the user's Task-1 One/Zero[shift]

# Literal-vs-symbolic identity predicates (dispatch, not value-probing a Scalar):
_const_iszero(c::Constant{T,T}) where {T} = iszero(c.val)
_const_iszero(::Constant)   = false               # V<:Scalar ⇒ symbolic, never identity
_const_iszero(::AbstractTerm) = false
_const_isone(c::Constant{T,T}) where {T} = isone(c.val)
_const_isone(::Constant)   = false
_const_isone(::AbstractTerm) = false
```

Move the existing `_assert_concrete` helper from `StencilCalculus/src/terms.jl`
to `StencilCore` (it is shared by `Scalar`/`Constant` here and `Slot`/`Term` in
Calculus). Export `AbstractScalar, Scalar, Constant` (and `_assert_concrete`,
`_const_iszero`, `_const_isone` as internal imports) from `StencilCore`.

**Placement note:** `Constant` is an `AbstractTerm`, and Core's `term.jl`
currently states "concrete subtypes live in StencilCalculus". I am placing
`Constant` in Core anyway, beside `Scalar`, because it is the scalar↔term bridge
and belongs with the scalar family the user asked to move to Core. Update that
comment. (Alternative: keep `Constant` in Calculus — flag if preferred.)

**`StencilCalculus/src/terms.jl`** keeps **`Slot`, `Term`, `Shifted`** only
(plus the `@slot` macro). `Term.args` stays `Tuple{Vararg{AbstractTerm}}` — **no
union widening needed**. Add a dispatch-only alias used by operators:

```julia
const TermLike = Union{AbstractTerm, AbstractScalar}     # operator-boundary only
asterm(t::AbstractTerm)   = t
asterm(s::AbstractScalar) = Constant(s)
asterm(x::Number)         = Constant(x)
Base.convert(::Type{<:AbstractTerm}, x::Number) = Constant(x)   # now type-correct
```

`@scalar` / `@const` macros: `@scalar` builds a `Scalar` (Core type); `@const`
now builds a `Constant`. Keep both in Calculus (re-exported), referencing Core
types.

## File-by-file changes (StencilCalculus)

- **`StencilCalculus.jl`**: import `AbstractScalar, Scalar, Constant,
  _assert_concrete, _const_iszero, _const_isone` from `StencilCore`; export
  `AbstractScalar, Scalar, Constant` (drop `Const, Zero, One` exports).
- **`operators.jl`**: collapse the binary/unary overloads onto `TermLike` so
  scalars wrap via `asterm`:
  ```julia
  op(a::TermLike, b::TermLike) = Term(op, (asterm(a), asterm(b)))
  op(a::TermLike, b::Number)   = Term(op, (asterm(a), Constant(b)))
  op(a::Number,   b::TermLike) = Term(op, (Constant(a), asterm(b)))
  op(a::TermLike)              = Term(op, (asterm(a),))         # unary
  SVector(args::TermLike...)   = Term(SVector, map(asterm, args))
  ```
  Keep `getindex` shift sugar for `Slot`/`Shifted` here; the `Scalar`/`Constant`
  shift-invariance moved to Core. Remove the old `Scalar` getindex methods.
  `Diff` field bound widens to `Diff{V<:TermLike}` so `∂(τ)` (bare `Scalar`)
  works.
- **`simplify.jl`**: stays `AbstractTerm → AbstractTerm`.
  - `rule_shift_const`: `Shifted{…,U<:Constant}` → `t.term` (Constant is
    position-independent). `Shifted` can no longer wrap a scalar (scalars aren't
    `AbstractTerm`), so the old `Union{Const,Zero,One,Scalar}` collapses to
    `Constant`.
  - `rule_shift_pushdown`: unchanged — args are all `AbstractTerm`; a `Constant`
    arg simply gets a `Shifted` wrapper that `rule_shift_const` then strips.
  - `rule_identity`: replace `isa Zero`/`isa One` type checks with
    `_const_iszero`/`_const_isone`, and return `Constant(zero(eltype(t)))` where
    it returned `Zero{…}()`.
  - `rule_fold`: fold all-`Constant{T,T}` (literal) args → `Constant(fn(vals…))`.
- **`differentiate.jl`**:
  - Derivative table: `One{P}()` → `Constant(one(P))`, `-One{…}()` →
    `Constant(-one(…))`, `Const(2)` → `Constant(2)`, etc.
  - `_diff`: replace the `Const/Zero/One` and `Scalar` leaf rules with rules on
    `Constant`:
    ```julia
    _diff(c::Constant{T,V}, ::Scalar{S}) where {T,S,V<:Scalar{S}} =
        _Contrib[ô => Constant(one(T))]      # ∂Constant(τ)/∂τ
    _diff(::Constant, ::AbstractScalar) = _Contrib[]
    _diff(::Constant, ::Slot)           = _Contrib[]
    ```
    Keep `_diff(::Slot{S2},::Slot{S})`, `_diff(::Shifted,::Slot)`,
    and the cross-inert `Slot↔Scalar` cases.
  - `_Contrib` value type and the `coefs` vector stay `AbstractTerm` (coeffs are
    always terms now — no bare scalars escape). The "drop identically-zero"
    filter becomes `findall(c -> !_const_iszero(c), coefs)`.
  - `differentiate(t, ::Scalar{S})` still returns an `AbstractTerm` (the
    coefficient term); update the docstring (no scalar leaks).
- **`materialize.jl`**: replace `_body_expr(::Scalar)`, `_body_expr(::Const)`,
  `_body_expr(::Zero/One)` with one method on `Constant`:
  ```julia
  _body_expr(c::Constant{T,T}, idx) = c.val                     # literal
  _body_expr(c::Constant{T,V}, idx) where {V<:Scalar{S}} where {S} =
      Expr(:., :args, QuoteNode(S))                             # args.S
  ```
  `_collect_acc!(a, ::Constant) = a` (broadcast values define no grid box;
  same as the old Scalar/Const handling).
- **`show.jl`**: `Base.show(io, s::AbstractScalar)` → its symbol;
  `_show(c::Constant{T,T})` → `show(io, c.val)`; `_show(c::Constant{<:Any,
  <:Scalar{S}})` → `print(io, S)`. **Glyph note (ramification):** the old
  type-agnostic `0`/`1` rendering of `Zero`/`One` is lost — `Constant(0.0)`
  shows `0.0`. Optionally special-case `_const_iszero`/`_const_isone` to keep
  `0`/`1` glyphs; **recommend dropping the glyph special-case** (show the value).
- **`trees.jl`**: replace the `Scalar`/`Const`/`Zero`/`One` `nodevalue`/`children`
  methods with `Constant` (`nodevalue(c::Constant{T,T}) = c.val`;
  `nodevalue(c::Constant{<:Any,<:Scalar{S}}) = S`; `children(::Constant) = ()`).
  Add `Scalar` node methods if bare scalars are ever walked.
- **`bridge.jl`**: `densify` pads with `Constant(zero(T))` instead of
  `Zero{T}()`. `_to_column`'s `Shifted(-s, g)` and `_interlace(::NTuple{
  AbstractTerm})` are unaffected — coefficients are `Constant`/`Term`, all
  `AbstractTerm`.

## Key ramifications (call out for review)

1. **Loss of type-level identity dispatch.** `Zero`/`One` were distinguished by
   *type*; the codebase deliberately avoided `iszero`/`isone` probes. Unifying
   into `Constant` forces **value-based** identity detection (`_const_iszero`/
   `_const_isone` on literal `Constant{T,T}`). Symbolic `Constant{T,<:Scalar}`
   is never an identity (guarded by dispatch on `V`).
2. **`strict, no auto-fold` semantics change.** Previously `f * Const(0.0)` did
   **not** annihilate (only the `Zero` *type* did). Now `f * Constant(0.0)`
   annihilates to `Constant(0.0)` via `_const_iszero`. This is mathematically
   correct but is a deliberate behavior change; the corresponding "strict" tests
   (`!(… isa Zero)`) are removed.
3. **The user's original Task 1** (`getindex(::One/::Zero, ::SShift) = self`) is
   subsumed by `Constant` shift-invariance (`getindex(::Constant, ::StaticShift)
   = c`).
4. **No `AbstractExpr`/union threading** through `Term`/`simplify`/`Stencil`:
   scalars are wrapped in `Constant` at the boundary, so the CAS core is
   untouched structurally. The only `TermLike` union is dispatch sugar in
   `operators.jl`.
5. **StencilAssembly**: no references to `Scalar`/`Const`/`One`/`Zero` — no
   changes needed.

## Tests to update (both packages)

`StencilCalculus/test/runtests.jl`: replace `Const`/`Zero`/`One` constructions
with `Constant`/`Scalar`; drop `Const`-as-term and `isa Zero` assertions; adjust
`@const α 1` ⇒ `α === Constant(1)`; `repr` expectations for `0`/`1`/`τ`;
`simplify(Shifted(ê₁, Const(2.0)))` ⇒ a `Constant` shift-invariance test;
differentiation `sst.terms` expectations (`One{…}` ⇒ `Constant(one(…))`).
`StencilCore/test/runtests.jl`: add `AbstractScalar`/`Scalar`/`Constant`
construction + `eltype` + concrete-type-assertion + shift-invariance tests.

## Verification (success criteria)

- `cd ../StencilCore && julia --project -e 'using Pkg; Pkg.test()'` passes
  (new scalar/constant tests included).
- `cd StencilCalculus && julia --project -e 'using Pkg; Pkg.test()'` passes,
  exercising: operator wrapping (`τ * f` builds `Term(*,(Constant(τ),f))`),
  `simplify` identity/fold via `Constant`, `differentiate` w.r.t. both a `Slot`
  and a `Scalar`, `materialize`/`code_string`, and the `build_stencil` bridge.
- Spot-check the trilogy end-to-end: a `differentiate → build_stencil →
  StencilAssembly.build` path still produces a matching sparse operator.
- `julia> τ = Scalar{:τ}(); f = Slot{:f}(); ∂(τ)(τ*f)` returns `f` (term), and
  `∂(f)(τ*f)` returns the `Constant(τ)` coefficient stencil.
