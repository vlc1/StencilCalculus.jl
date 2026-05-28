# Guide

An end-to-end tour: from a symbolic grid expression to an assembled sparse
matrix.

## Building expressions

Discrete fields are [`Slot`](@ref)s; cell-level parameters are
[`Symbolic`](@ref)s (named, runtime-substituted) and [`Const`](@ref)s
(literal); both come from StencilCore. Pointwise operators combine them, and
the non-local functors [`δ₊`/`δ₋`/`σ₊`/`σ₋`](@ref FwdDiff) apply
forward/backward differences and sums along a chosen axis. Indexing a slot
shifts it by a lattice offset:

```julia
using StencilCalculus

@slot f Float64
g = f[-2ê₁] .- 4 .* f[-ê₁] .+ 3 .* f[]      # f[i-2] - 4 f[i-1] + 3 f[i]
@slot ψ Float64
adv = ψ .* δ₊{1}(f)                # ψ[i] * (f[i+1] - f[i])
```

The `@slot` / `@symbolic` / `@const` macros bind a variable to a leaf named
after it (`@slot f` ≡ `f = Slot{:f, Float64}()`); the type argument defaults
to `Float64` and must be **concrete** (it is the type the term materializes
into). The element type is computed at construction, so an ill-typed
expression is rejected early.

A bare `Symbolic` or `Const` is an [`AbstractScalar`](@ref) — *not* an
`AbstractTerm`. The operators lift it into term-land via a
[`Fill`](@ref) leaf the moment it interacts with a real term:

```julia
@symbolic τ Float64                   # τ isa Symbolic <: AbstractScalar
τ .* f                                # Term(*, (Fill(τ), f)) — Fill wraps τ
@const α 2
α .+ f                                # Term(+, (Fill(α), f))
2 .* f                                # Term(*, (Fill(Const(2)), f))
```

Un-dotted operators (`*`, `+`, `sin`) on `AbstractPointwise` raise `MethodError`
— they are reserved for scalar-land. Broadcast with no `AbstractPointwise`
operand (e.g. `τ .* ψ` with both scalars) raises `ArgumentError`.

`Symbolic`s materialize to a single broadcast value (e.g. a timestep), unlike
`Slot`s which materialize to per-cell arrays.

## Simplifying

[`simplify`](@ref) rewrites to a normal form — shifts pushed onto the leaves,
nested shifts merged, identities collapsed, and **all-`Fill` sub-expressions
folded into one `Fill(Scalar(…))`** (the scalar-precedence rule: scalar
arithmetic done once at compile time, not broadcast cell-by-cell):

```julia
@slot g Float64
simplify(δ₊{1}(f .+ g))               # (f[ê₁] + g[ê₁]) - (f + g)

# All-Fill collapse + scalar folding in one pass:
simplify(Fill(Const(2.0)) .+ Fill(Const(3.0)))   # Fill(Const(5.0))
simplify(2 .* (f .+ 3 .* f))                     # (2*1 + 2*3 = 8) ⇒ collapses inside Fill
```

The identity rules detect a structural `Zero`/`IdentityStencil` *by type*, a
`Fill{<:Null}`/`Fill{<:Unity}` *by type* (matching the scalar-side dispatch),
and a literal `Fill{<:Const}` *by value* (`iszero`/`isone` on the wrapped
literal). The last is a deliberate departure from a stricter no-auto-fold
stance — `f .* Fill(Const(0.0))` annihilates to `Zero(Float64)`, which is
mathematically correct.

## Differentiating into a stencil

[`differentiate`](@ref) with respect to a slot yields a row-anchored `Stencil`
whose per-offset coefficients are the partial derivatives:

```julia
differentiate(δ₊{1}(f), f)           # offsets (ô, ê₁), coefficients (-1, 1)

# variable coefficient — ∂(ψ·δ₊{1}(f))/∂f
@slot ψ Float64
differentiate(ψ .* δ₊{1}(f), f)

# nonlinear — ∂(f*f)/∂f = f + f
differentiate(f .* f, f)
```

A Laplacian-shaped expression differentiates to the five-point star:

```julia
lap = δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f))  # f[i±1] + f[j±1] - 4 f
differentiate(lap, f)                     # a Stencil that narrows to a star
```

Differentiating with respect to a `Symbolic` collapses the per-offset
structure to a single broadcast coefficient (an `AbstractTerm`, not a
`Stencil`):

```julia
@symbolic τ Float64
differentiate(τ .* f, τ)             # === f       (a term)
differentiate(τ .* f, f)             # a Stencil   (offset ô, coefficient Fill(τ))
```

The scalar pieces of the expression — anything inside a `Fill{<:AbstractScalar}`
— are differentiated by StencilCore's scalar `differentiate`; the result is
re-wrapped in a `Fill` on the way back into term-land.

## Building and assembling

[`build_stencil`](@ref) converts the row-anchored result to a column-anchored,
assemblable stencil (narrowing to a `LinearStencil`/`StarStencil` and
materializing the coefficients). Pass `pad = true` to fill single-axis offset
gaps, and the mesh `size` for a constant coefficient. Then assemble with
[StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/):

```julia
using StencilCalculus, StencilAssembly

@slot f Float64
sst = differentiate(lap, f)
st  = build_stencil(sst; size = (5, 4))     # → StarStencil{1, 2, 5, …}
A   = build(st, (1:5, 1:4), (1:5, 1:4))     # SparseMatrixCSC
```

For a variable-coefficient operator, pass the substituted arrays instead of
`size`:

```julia
ψv  = collect(1.0:8.0)
@slot ψ Float64
sst = differentiate(ψ .* δ₊{1}(f), f)
st  = build_stencil(sst, (ψ = ψv,))         # coefficients read from ψv
```

## Inspecting: materialize and code_string

[`materialize`](@ref) compiles an expression into a read-only
[`LazyArray`](@ref); [`code_string`](@ref) renders the same per-cell kernel as
source you can drop into a file:

```julia
fv = rand(16); ψv = rand(16)
la = materialize(adv, (f = fv, ψ = ψv))     # axes (1:15,); la[i] = ψv[i]*(fv[i+1]-fv[i])

print(code_string(adv; name = :advect))
```

A `Fill` materializes its wrapped value (recursively, for an
`AbstractScalar`) once per cell — so `materialize(τ .* f, (f = fv, τ = 0.5))`
gives a kernel that reads `args.τ * args.f[i]`, not `args.τ[i] * args.f[i]`.
