# Guide

An end-to-end tour: from a symbolic grid expression to an assembled sparse
matrix.

## Building expressions

Discrete fields are [`Slot`](@ref)s; parameters are [`Scalar`](@ref)s. Pointwise
operators combine them, and the non-local functors
[`δ₊`/`δ₋`/`σ₊`/`σ₋`](@ref FwdDiff) apply forward/backward differences and sums
along a chosen axis. Indexing a slot shifts it by a lattice offset:

```julia
using StencilCalculus

f = Slot{:f, Float64}()
g = f[-2ê₁] - 4f[-ê₁] + 3f[]      # f[i-2] - 4 f[i-1] + 3 f[i]
adv = Slot{:ψ, Float64}() * δ₊{1}(f)   # ψ[i] * (f[i+1] - f[i])
```

The element type is computed at construction, so an ill-typed expression is
rejected early. `Scalar`s materialize to a single broadcast value (e.g. a
timestep), unlike `Slot`s which materialize to per-cell arrays.

## Simplifying

[`simplify`](@ref) rewrites to a normal form — shifts pushed onto the leaves,
nested shifts merged, identities collapsed:

```julia
simplify(δ₊{1}(f + Slot{:g, Float64}()))   # (f[ê₁] + g[ê₁]) - (f + g)
```

## Differentiating into a stencil

[`differentiate`](@ref) with respect to a slot yields a row-anchored `Stencil`
whose per-offset coefficients are the partial derivatives:

```julia
differentiate(δ₊{1}(f), f)               # offsets (ô, ê₁), coefficients (-1, 1)

# variable coefficient — ∂(ψ·δ₊{1}(f))/∂f
differentiate(Slot{:ψ,Float64}() * δ₊{1}(f), f)

# nonlinear — ∂(f*f)/∂f = f + f
differentiate(f * f, f)
```

A Laplacian-shaped expression differentiates to the five-point star:

```julia
lap = δ₋{1}(δ₊{1}(f)) + δ₋{2}(δ₊{2}(f))   # f[i±1] + f[j±1] - 4 f
differentiate(lap, f)                     # a Stencil that narrows to a star
```

## Building and assembling

[`build_stencil`](@ref) converts the row-anchored result to a column-anchored,
assemblable stencil (narrowing to a `LinearStencil`/`StarStencil` and
materializing the coefficients). Pass `pad = true` to fill single-axis offset
gaps, and the mesh `size` for a constant coefficient. Then assemble with
[StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/):

```julia
using StencilCalculus, StencilAssembly

f = Slot{:f, Float64}()
sst = differentiate(lap, f)
st  = build_stencil(sst; size = (5, 4))     # → StarStencil{1, 2, 5, …}
A   = build(st, (1:5, 1:4), (1:5, 1:4))     # SparseMatrixCSC
```

For a variable-coefficient operator, pass the substituted arrays instead of
`size`:

```julia
ψv  = collect(1.0:8.0)
sst = differentiate(Slot{:ψ,Float64}() * δ₊{1}(f), f)
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
