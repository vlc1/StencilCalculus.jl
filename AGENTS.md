# Copilot Instructions for StencilCalculus.jl

## Build, Test, and Lint

**Julia version**: 1.11 (minimum requirement; CI also tests 1.12 and nightly)

**Run all tests**:
```bash
julia --color=yes --project=. -e 'using Pkg; Pkg.test()'
```

**Run a specific test set** (e.g., "leaf construction + eltype"):
```bash
julia --project=. -e 'include("test/runtests.jl")'  # then modify runtests.jl to focus on one @testset
```

**Interactive REPL** (for development):
```bash
julia --project=.
```

**CI environment**: Tests run on Julia 1.11, 1.12, and pre-release with unregistered dependencies. The CI workflow checks out `StencilCore` and `StencilAssembly` as sibling directories (via `[sources]` paths in Project.toml).

## High-Level Architecture

**Three-package stack**:
1. **StencilCore** (dependency) — provides `AbstractTerm{T}`, `StaticShift` offset vocabulary, and `AbstractStencil` base types
2. **StencilCalculus** (this package) — a small CAS with concrete term types, rewriting, differentiation, and codegen
3. **StencilAssembly** (extension) — CSC sparse matrix assembly from stencils (test-only dependency)

**Design philosophy**: *Type parameters encode structure (shifts, operations); runtime fields hold data (coefficients, values).*

**Core pipeline** (from README):
```
@slot f, @var ψ             ← Define named placeholders (DSL sugar)
expr = ψ .* δ₊{1}(f)        ← Build symbolic expression tree (broadcast syntax)
simplify(expr)              ← Reduce via dispatch-driven rules (shifts, identities, const folding)
materialize(expr, data)     ← Compile to LazyArray + code_string (codegen via RuntimeGeneratedFunctions)
differentiate(expr, f)      ← Symbolic differentiation → Stencil{RowAccess}
build_stencil(…)            ← Bridge + narrow to LinearStencil/StarStencil (RowAccess → ColumnAccess)
```

**Main modules** (each 50–150 lines):
- **pointwise.jl** — `Slot{S,T}`, `IdentityStencil{T,U}`, `DiagonalStencil{T,A}`, `Pointwise`, `Shifted`, `Fill`, and the `Zero{T} = Fill{Null{T}}` alias; constructor macro `@slot`
- **operators.jl** — Broadcast plumbing (`PointwiseStyle` / `ScalarStyle`), `SVector` interception, indexing sugar (`f[-2ê₁]`), DSL functors (`δ₊`, `δ₋`, `σ₊`, `σ₋`)
- **simplify.jl** — Hand-rolled rule rewriter (`POINTWISE_DEFAULT_RULES`: shift composition/pushdown, identity/annihilator collapse, double-negation, scalar precedence); post-walk to fixed point
- **differentiate.jl** — Symbolic differentiation table (function-keyed rules, extensible via `@pointwise_rule`); row-anchored Jacobian coefficients
- **materialize.jl** — Codegen to `Expr` → compiled kernel via `RuntimeGeneratedFunctions`; `LazyArray` wrapper
- **bridge.jl** — Convert `Stencil{RowAccess} → ColumnAccess`, narrow to `LinearStencil`/`StarStencil`
- **apply.jl** — `*(::AbstractStencil, ::AbstractPointwise)` operator surface. The two diagonal-stencil methods (`IdentityStencil`, `DiagonalStencil`) have working bodies; the three `NeighborhoodStencil` shells (`Stencil`, `LinearStencil`, `StarStencil`) still throw `"not yet implemented"`. See "Stencil application" below.
- **trees.jl** — `AbstractTrees.jl` interface (for traversal)
- **show.jl** — Component-form display (normal-form canonical printing)

## Key Conventions

### Type System & Concrete Types
- **`T` must be concrete**: All `AbstractPointwise{T}` subtypes require concrete element types (e.g., `Float64`, not `Number`). Enforced by `_assert_concrete()` at construction; throws `ArgumentError` if violated.
- **Element type inference at construction**: `T = Base.promote_op(fn, eltype.(args)...)` for `Pointwise` nodes. `SVector{2, Float64}` from `SVector(a::Slot{_, Float64}, b::Slot{_, Float64})`.
- **Defaults to `Float64`**: `Slot{:f}()` ≡ `Slot{:f, Float64}()`.
- **Two CAS layers**: Scalar-land uses `Var{S,T}` / `Constant` / `Null{T}` / `Unity{T}` / `Scalar(fn, args)` (StencilCore). Pointwise-land uses `Slot{S,T}` / `Fill` / `Zero{T}` / `IdentityStencil{T,U}` / `DiagonalStencil{T,A}` / `Pointwise(fn, args)` (StencilCalculus). The `Fill` bridge wraps a scalar or literal into a spatially-invariant `AbstractPointwise`. `Zero` is a type alias: `const Zero{T} = Fill{Null{T}}` — it shares `Null`'s bool-shape discipline, and `eltype(Zero(Float64)) === Bool` (the Float64 is the *input* the bool-shape ctor consumes; promotion in surrounding arithmetic recovers it). `IdentityStencil` mirrors scalar-side `Unity`: bool-shape eltype `T` (`Bool`, `SMatrix{N,N,Bool}`), value-space `U` recovered at materialize-time via `one(U)`. `DiagonalStencil(t)` wraps an `AbstractPointwise{T}` (with `one(T)` defined) and applies as elementwise multiplication.
- **Stencil hierarchy**: `AbstractStencil{T}` is the single-parameter abstract root (StencilCore). Its two abstract children are `AbstractPointwise{T} <: AbstractStencil{T}` (diagonal stencils — every pointwise term IS a diagonal stencil) and `NeighborhoodStencil{T, S<:AccessStyle} <: AbstractStencil{T}` (the offset-bearing `Stencil`/`LinearStencil`/`StarStencil`, which carry an [`AccessStyle`](@ref) for column-/row-anchored assembly).
- **Construction syntax — broadcast vs scalar**: dotted operators (`f .* g`, `sin.(f)`) build `Pointwise` nodes; un-dotted operators (`τ * ψ`, `sin(τ)`) build `Scalar` nodes in scalar-land. Un-dotted operators on `AbstractPointwise` raise `MethodError` (no overload); broadcast with no `AbstractPointwise` operand raises `ArgumentError` (the `ScalarStyle` materializer rejects it). The `SVector(_, _)` constructor, `f[ê₁]` shift sugar, and DSL functors (`δ₊{D}(_)`, `∂(_)(_)`) are not Base operators and are unchanged.

### Dispatch-Driven Simplification
- **Rules are functions**: `(::AbstractPointwise) -> Union{Nothing, AbstractPointwise}`. Return `nothing` if inapplicable; return rewritten term otherwise.
- **Post-walk fixed-point**: `simplify` processes children first, applies the first matching rule per node, and repeats passes until `===` (structural equality).
- **Six default rules** (in `simplify.jl`, exported as `POINTWISE_DEFAULT_RULES`):
  1. Shift composition: `Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁ + s₂, t)`
  2. Shift pushdown: `Shifted(s, f(a…)) → f(Shifted(s, a)…)`
  3. Shift over constants: `Shifted(s, Fill|IdentityStencil) → Fill|IdentityStencil` (Zero is a `Fill{<:Null}`)
  4. Identity/annihilator: `Zero .+ x → x`, `x .* Zero → Zero`, `IdentityStencil .* x → x` (dispatch on types, never `iszero`/`isone` probes)
  5. Double negation: `.-(.-x) → x`
  6. Scalar precedence: all-`Fill` `Pointwise` → `Fill(Scalar(fn, …))`

### Shifts are Type-Level Structure
- **`StaticShift` / `StaticPair{D, O}`**: Offsets entered via DSL functors (`FwdDiff{D}`, `BwdDiff{D}`, etc.); axis `D` and offset `O` are type parameters.
- **Shift algebra**: `+`/`-`/`*Int` compositions evaluated at the type level; simplification reduces nested shifts via dispatch.
- **Normal form**: `Shifted` appears only at the top level (directly wrapping a `Slot`); `simplify` pushes shifts down over `Pointwise` nodes.

### Differentiation & Row-Anchored Stencils
- **Function-keyed rule table**: Derivative coefficients stored as a table of `(fn, (arg_indices)) -> derivative_expr` rules (ChainRules `frule` convention). Extend with `@pointwise_rule`.
- **Row-anchored**: Jacobian coefficient at offset `δ` is `∂F/∂(shifted_arg)` evaluated at the **row index**—no shifts injected into the coefficient.
- **Output**: `Stencil{…, T, RowAccess}` with per-offset coefficient terms, where `T = _unity_space(eltype(slot))` (passed explicitly into the `Stencil(T, S, …)` ctor so all-wildcard derivatives like `differentiate(f, f)` pin T). The ctor enforces strict uniformity against the supplied T (`Fill{<:Null}` / `Fill{<:Unity}` / `IdentityStencil` and any `Pointwise`/`Shifted` reaching only such leaves are *wildcards* and pass the check unconditionally). Bridge converts to `ColumnAccess` (flips shifts: `Shifted(-δ, …)`) before narrowing. Mixed-eltype literals in the input (e.g. `3 .* f` on a `Slot{:f, Float64}`) will error at `Stencil` construction — promote literals to match (`3.0`).

### Stencil application: `*(stencil, pointwise)`
- **Diagonal stencils**: `*(::IdentityStencil, p::AbstractPointwise) === p` (neutral); `*(d::DiagonalStencil, p::AbstractPointwise) === Pointwise(*, (d.term, p))` (broadcast multiply). Both enforce the eltype-match rule below before applying.
- **Neighborhood-stencil shells**: `Base.*(::Stencil, ::AbstractPointwise)`, `Base.*(::LinearStencil, ::AbstractPointwise)`, `Base.*(::StarStencil, ::AbstractPointwise)` still throw `ErrorException` with `"not yet implemented"`. Implementation lands in a follow-up.
- **Eltype-match rule**: a stencil `<: AbstractStencil{T}` may multiply an `AbstractPointwise{U}` only when `T === _unity_space(U)` (for DiagonalStencil / NeighborhoodStencil) or `T === _to_bool_shape(_unity_space(U))` (for IdentityStencil, whose `T` is bool-shape). Scalar-on-scalar for `U <: Number`; `SMatrix{N, N, F}`-on-`SVector{N, F}` for vector-valued fields.
- **`*(::AbstractPointwise, ::AbstractPointwise)` remains unsupported as a generic** by design — even though `AbstractPointwise <: AbstractStencil` now, the `*` methods are defined on each *concrete* stencil subtype (`IdentityStencil`, `DiagonalStencil`, plus the three shells). Two raw `Slot`s, two `Pointwise`s, or any other pair of AbstractPointwise operands that aren't reified as diagonal stencils raises `MethodError` (regression-guarded by tests).

### Codegen & Materialization
- **`materialize` → compiled kernel**: Builds a Julia `Expr` from the term tree; executes via `RuntimeGeneratedFunctions.jl`.
- **Index arithmetic from types**: Shift offsets (from `StaticShift` type parameters) become literal `+`/`−` operations in the index.
- **`LazyArray` wrapper**: Holds the compiled function, substitution `NamedTuple`, and axes; acts as an `AbstractArray{T,N}`.
- **Inspectable source**: `code_string(expr; name=:kernel)` dumps the same `Expr` as formatted Julia source.

### Exports & Public API
- **Concrete pointwise terms**: `Slot`, `IdentityStencil`, `DiagonalStencil`, `Pointwise`, `Shifted`, `Fill`, `AbstractPointwise` (from StencilCore); `Zero` is exported as the alias `Fill{Null{T}}`
- **Re-exported scalar terms** (from StencilCore): `AbstractScalar`, `Var`, `Constant`, `Null`, `Unity`, `Scalar`
- **Macros**: `@slot`, `@var` (re-exported from StencilCore), `@pointwise_rule`
- **DSL operators**: `FwdDiff`, `BwdDiff`, `FwdSum`, `BwdSum`, `δ₊`, `δ₋`, `σ₊`, `σ₋`
- **Rewriting & analysis**: `simplify`, `POINTWISE_DEFAULT_RULES`, `differentiate`, `derivative`, `Diff`, `∂`
- **Materialization**: `materialize`, `code_string`, `LazyArray`
- **Assembly bridge**: `build_stencil`, `densify`
- **Stencil application**: `*(stencil, pointwise)` (reserved, shells; bodies TBD)
- **Shift vocabulary** (from StencilCore): `StaticShift`, `SShift`, `StaticPair`, `SPair`, basis vectors `ê₁`–`ê₉`, zero shift `ô`

## Testing Patterns

- **Organized by concern**: Test sets cover leaf construction, macros, operators, eltypes, getindex sugar, display, simplification rules, differentiation, materialization, and bridge.
- **Immutable struct equality**: Use `===` for structural identity checks (every node is immutable, no object identity needed).
- **Example from `test/runtests.jl`**:
  ```julia
  @testset "constructor macros" begin
      @slot a
      @test a === Slot{:a, Float64}()
  end
  ```