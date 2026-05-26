# Record: `AbstractScalar` algebra in Core + `Fill` bridge in Calculus

> **Status:** shipped. StencilCore on `scalar-algebra-refactor` (5 commits
> ahead of `main`); StencilCalculus on `scalar-algebra-refactor` (2 commits
> ahead of `main`). Test status: Core 156/156, Calculus 166/166. Both
> deployed docs sites build.

## Context

`AbstractTerm{T}` is conceptually an *array-like collection of `T`-valued
scalars*, not a scalar itself. Pre-refactor, `Scalar` and `Const` were
subtypes of `AbstractTerm` — the source of the bug fixed in `7e1b3ca`
(`derivative(-,…) = Const(-1)` used a *value* where a type-level *array
identity* belonged). The refactor splits the type vocabulary into two sibling
algebras, bridged by one new term.

**Scalars and terms are not redundant.** A scalar tree contracts to *one*
materialized value per call; a term tree contracts to a *per-cell array* sized
by `Slot` access and `Shifted` offsets. `Fill` says "this term has no spatial
variation — materialize the scalar once per cell." The cost is some structural
duplication in `simplify` rules and `AbstractTrees` plumbing; that is
acceptable.

## Two parallel hierarchies (as shipped)

| role                    | `AbstractScalar` (Core)                          | `AbstractTerm` (Calculus)                    |
|-------------------------|--------------------------------------------------|----------------------------------------------|
| named leaf              | `Symbolic{S, T}`                                 | `Slot{S, T}`                                 |
| literal leaf            | `Const{T}` with `val::T`                         | — (literals enter terms via `Fill(Const(…))`)|
| tree node               | `Scalar{F, A<:Tuple{Vararg{AbstractScalar}}, T}` | `Term{F, A<:Tuple{Vararg{AbstractTerm}}, T}` |
| additive identity       | `Null{T}`                                        | `Zero{T}` (retained)                         |
| multiplicative identity | `Unity{T}`                                       | `One{T}` (retained)                          |
| spatial shift node      | —                                                | `Shifted{Sh,T,U}`                            |
| broadcast bridge        | —                                                | `Fill{T}` with `val::T`                      |

**Difference from the original plan.** The plan recommended *removing* `One{T}`
in favour of `Fill(Const(one(T)))`. What actually shipped: `One{T}` is
**kept** as the type-level term-side multiplicative identity, and a new
`Unity{T} <: AbstractScalar{T}` is added for symmetry. So *both* sides have a
type-dispatched mul identity (and the row `Unity ↔ One` joins `Null ↔ Zero`
in the parallel table). User decision documented in this thread.

**Naming consequences.** Old `Scalar` ⇒ new `Symbolic`. Old `Const` (was
`AbstractTerm`) ⇒ new `Const` (now `AbstractScalar`); literal terms are
written `Fill(Const(v))`. `@scalar` macro renamed to `@symbolic`. `@const`
and `@slot` keep their names. `Const`'s field renamed `value` → `val`.

## `Fill`: the bridge

```julia
struct Fill{T} <: AbstractTerm{T}
    val::T
    Fill{T}(val) where {T} = (_assert_concrete(:Fill, T); new{T}(val))
end
Fill(v::AbstractScalar) = Fill{typeof(v)}(v)
Fill(v) = Fill{typeof(v)}(v)

# Recursive eltype: Fill{Symbolic{:τ,Float64}} has eltype Float64 (not the
# Symbolic type), so Slot{:f,Float64}() + Fill(τ) promotes through `promote_op`
# against the underlying numeric type.
Base.eltype(::Type{Fill{T}}) where {T<:AbstractScalar} = eltype(T)

Base.getindex(f::Fill) = f
Base.getindex(f::Fill, ::StaticShift) = f             # shift-invariant
```

`Fill` lives in StencilCalculus (it is an `AbstractTerm`).

## Scalar precedence rule (the answer to "what is `2 * Fill(τ)`?")

**Statement.** Inside a `Term`, a contiguous all-`Fill` sub-expression is
*not* a normal form: it is folded into a single `Fill` wrapping a `Scalar`
tree of the equivalent scalar expression.

**Mechanism (simplify rule).** Operators always build a `Term`. A new rule
rewrites:

```julia
function rule_fill_collapse(t::Term)
    all(a -> a isa Fill, t.args) || return nothing
    Fill(simplify(Scalar(t.fn, map(a -> a.val, t.args))))
end
```

so `Term(*, (Fill(Const(2)), Fill(τ)))` → `Fill(Scalar(*, (Const(2), τ)))`.
A `Number` operand entering a `Term` arrives as `Fill(Const(n))` (numeric
literals canonicalise to `Const` first); an `AbstractScalar` operand `s`
arrives as `Fill(s)`. When the other operand is a *true* term (a `Slot`, a
`Shifted`, a `Term` with non-Fill leaves), the rule does not fire and the
result stays a `Term` with a `Fill` leaf — which is correct: there is
spatial structure to broadcast over.

## What shipped — StencilCore

Five commits on `scalar-algebra-refactor`:

- `d2fb123` **add AbstractScalar algebra (scalar-side hierarchy)**:
  `src/scalars.jl` (types, ctors, ops, getindex shift-invariance, `asscalar`,
  `convert`, `@symbolic` / `@const` macros, show), `src/scalar_trees.jl`
  (AbstractTrees plumbing), AbstractTrees added to `Project.toml`,
  `_assert_concrete` moved to `term.jl` for sharing.
- `8aee1d0` **add scalar simplify**: `src/scalar_simplify.jl` with
  `rule_identity_scalar` (Null/Unity type-dispatch) and `rule_fold_scalar`
  (all-`Const` foldable ops). Internal helpers prefixed `_scalar_*`.
- `db285dc` **add scalar materialize and differentiate**:
  `src/scalar_materialize.jl` (with `_scalar_body_expr` for Calculus codegen)
  and `src/scalar_differentiate.jl` (frule table, chain rule short-circuits
  Null subderivatives). `differentiate(::AbstractScalar, ::Symbolic)`
  returns `Null{eltype(s)}()` for the no-dependence case (does not throw).
- `3bb015b` **tighten scalar derivative dispatch; export shared generics**:
  every `derivative(…)` method typed on `Vararg{AbstractScalar}` /
  `::AbstractScalar` so Calculus's `Vararg{AbstractTerm}` table is disjoint
  on the same generic; export `simplify`, `materialize`, `differentiate`,
  `derivative`.
- `052affc` **document the AbstractScalar algebra**: `docs/src/index.md`,
  `docs/src/guide.md` updated; `warnonly=[:cross_references]` added to
  `docs/make.jl`.

## What shipped — StencilCalculus

Two commits on `scalar-algebra-refactor`:

- `e53a11e` **cut over to AbstractScalar + Fill bridge**:
  - `terms.jl`: keep `Slot`, `Zero`, `One`, `Term`, `Shifted`; add `Fill`;
    drop old `Scalar`/`Const`/`_assert_concrete` (now imported); new
    `asterm` / `convert` route via `Fill(Const(…))` / `Fill(s)`.
  - `operators.jl`: `TermLike = Union{AbstractTerm, AbstractScalar}` for the
    operator boundary (dispatch only — not threaded through `Term.args` or
    simplify). Numeric literals → `Fill(Const(n))`; bare scalars → `Fill(s)`.
    `getindex` shift-invariance for `Fill`, `Zero`, `One`. `Diff{V<:TermLike}`.
  - `simplify.jl`: helpers `_is_term_zero` / `_is_term_one` — type-dispatched
    on `Zero`/`One`/`Fill{<:Null}`/`Fill{<:Unity}`, value-dispatched on
    `Fill{<:Const}`. New `rule_fill_collapse`. `_rebuild(Fill{<:AbstractScalar})`
    defers to Core's scalar simplify so inner scalars reach scalar-side
    normal form.
  - `differentiate.jl`: term-side `derivative` table on `Vararg{AbstractTerm}`.
    New `_diff_scalar` walker for `differentiate(::AbstractTerm, ::Symbolic)`
    that delegates `Fill{<:AbstractScalar}` leaves to Core's scalar
    `differentiate`. Slot path's `_diff(::Union{Fill,Zero,One}, ::Slot) = []`.
    Zero-filter uses `_is_term_zero`.
  - `materialize.jl`: `_body_expr(::Fill{<:AbstractScalar})` calls
    `_scalar_body_expr`; `_body_expr(::Fill)` emits the literal.
    `_collect_acc!(::Fill) = a` (no spatial access).
  - `show.jl`, `trees.jl`: drop old `Scalar`/`Const` methods, add `Fill`
    (renders as its wrapped value via the scalar-side `show`).
  - `bridge.jl` unchanged (uses `Zero{T}()`, still valid).
  - Tests rewritten: `Scalar{:τ,T}()` → `Symbolic{:τ,T}()`,
    `Const(v)` in term contexts → `Fill(Const(v))`, `@scalar` → `@symbolic`.
    Added tests for `Fill` eltype recursion, value-vs-type identity
    predicates, and `rule_fill_collapse`.
- `0b91b48` **document the cutover**: `docs/src/{index,guide,differentiation,
  api}.md` refreshed for the new design.

## Key ramifications (what landed)

1. **`One{T}` retained**; `Unity{T}` added — the parallel table now has *both*
   identities on both sides (Null↔Zero, Unity↔One). The plan's removal of
   `One{T}` was rejected in favour of symmetric type-dispatch on both
   identities.
2. **Identity predicates are hybrid.** `_is_term_zero` / `_is_term_one`
   type-dispatch on `Zero`/`One` and `Fill{<:Null}`/`Fill{<:Unity}`, and
   value-dispatch on `Fill{<:Const}` (`iszero`/`isone` on the wrapped value).
   A `Fill{<:Symbolic}` or `Fill{<:Scalar}` is never an identity (runtime
   value unknown).
3. **"Strict, no auto-fold" semantics changed deliberately.** `f *
   Fill(Const(0.0))` now annihilates to `Zero{Float64}()` via the value-
   dispatched predicate. This walks back the prior stance and is
   mathematically correct.
4. **`getindex(::Zero|::One|::Fill, ::SShift) = self`** — the plan's original
   Task 1 — landed in `operators.jl`.
5. **`@scalar` → `@symbolic` rename** (DSL break, accepted).
6. **`Const`'s field renamed `value` → `val`** (test API break, mechanical).
7. **One generic per CAS op**: `simplify`, `materialize`, `differentiate`,
   `derivative` are exported from Core; Calculus imports and extends with
   disjoint methods on `Vararg{AbstractTerm}` / `::AbstractTerm`.
8. **`Term.args` stays `Tuple{Vararg{AbstractTerm}}`** — no `AbstractExpr`
   union threaded anywhere in the CAS. The `TermLike` union is dispatch sugar
   at the operator boundary only.
9. **StencilAssembly untouched.**

## Verification

- `cd ../StencilCore && julia --project -e 'using Pkg; Pkg.test()'` ⇒
  **156/156**.
- `cd ../StencilCalculus && julia --project -e 'using Pkg; Pkg.test()'` ⇒
  **166/166**.
- `cd docs && julia --project make.jl` builds both doc sites (with the
  pre-existing cross-ref warnings for term↔scalar analogue links;
  `warnonly = [:cross_references]` on both).
- Spot-check: `τ = Symbolic{:τ}(); f = Slot{:f}()`. Then
  `∂(τ)(τ*f) === f` (term coefficient), `∂(f)(τ*f) isa Stencil`
  with single offset `ô` and coefficient `Fill(τ)`.
- `simplify(Fill(Const(2.0)) + Fill(Const(3.0))) === Fill(Const(5.0))` via
  `rule_fill_collapse` + Core's `rule_fold_scalar`.

## Out of scope (not addressed in this work)

- Root-level design records `docs/core.md` (StencilCore) and `docs/cas.md`
  (StencilCalculus) still describe the pre-refactor design. Not part of the
  deployed docs site; left for a follow-up.
- Branches not pushed; no PRs opened.
- Untracked scratch files in the Calculus worktree (`.agent-handoff.md`,
  `AGENTS.md`, `CLAUDE.md`, `docs/constant.md`, `docs/lands.md`) untouched.
