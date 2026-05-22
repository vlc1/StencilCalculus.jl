# GridAlgebra.jl

A small Computer Algebra System for expressions on N-D Cartesian meshes, whose
purpose is to **build `AbstractStencil`s by differentiation**. Part of a
three-package stack: it depends on [StencilCore](../StencilCore) for the term /
stencil types, and pairs with [CartesianOperators](../CartesianOperators) for
CSC assembly.

Define discrete fields as **slots**, compose component-wise and non-local
(shift) operators into a symbolic tree, then `simplify`, `differentiate`, and
lower the result to a compiled lazy array (`materialize`) or an assemblable
stencil (`build_stencil`).

## Pipeline at a glance

```julia
using GridAlgebra, CartesianOperators
using StaticArrays: SVector

f = Slot{:f, Float64}()
ψ = Slot{:ψ, Float64}()

# A symbolic grid expression (upwind advection):  out[i] = ψ[i] · (f[i+1] − f[i])
expr = ψ * δ₊{1}(f)

# Inspect the per-cell kernel that materialize compiles:
print(code_string(expr; name = :advect))
#   function advect(args, i)
#       args.ψ[i] * (args.f[i + 1] - args.f[i])
#   end

# Evaluate it lazily over substituted arrays:
fv = rand(16); ψv = rand(16)
lazy = materialize(expr, (f = fv, ψ = ψv))     # LazyArray, axes (1:15,)

# Differentiate w.r.t. f → a row-anchored Stencil, then build a CSC matrix:
sst = differentiate(expr, f)                   # StencilCore.Stencil{RowAccess}
st  = build_stencil(sst, (ψ = ψv,))            # → LinearStencil{ColumnAccess}
A   = build(st, (1:15,), (1:15,))              # SparseMatrixCSC
```

## What it provides

- **Term types** `Slot{S,T}` (per-cell field), `Scalar{S,T}` (broadcast
  parameter), `Const`, the type-level identities `Zero`/`One`, `Term`, and
  `Shifted`; element type `T` is computed at construction.
- **DSL**: component-wise operator overloads, `SVector` interception, indexing
  sugar `f[-2ê₁]` (shift by a `StaticShift`), and the difference / sum functors
  `δ₊`/`δ₋`/`σ₊`/`σ₋` (`FwdDiff`/`BwdDiff`/`FwdSum`/`BwdSum`).
- **`simplify`** — a rule rewriter (shift composition/pushdown, identities on
  `Zero`/`One`, constant folding).
- **`differentiate(t, ::Slot)`** — row-anchored symbolic differentiation onto a
  `Stencil{RowAccess}`.
- **`materialize`** / **`code_string`** — codegen to a `LazyArray` (via
  RuntimeGeneratedFunctions) and the same kernel as inspectable source.
- **`build_stencil`** — convert (Row→Column), `materialize`, and narrow
  (`as_linear`/`as_star`, with optional `densify` padding) to an assemblable
  `LinearStencil`/`StarStencil`.

## Install

Unregistered; clone [StencilCore](../StencilCore),
[CartesianOperators](../CartesianOperators), and this repo **side by side** (the
relative `[sources]` paths resolve them), then `]dev` this package.

## Design

The full design rationale lives in
[`CartesianOperators/docs/cas.md`](../CartesianOperators/docs/cas.md) (with the
package-split plan in `docs/core.md`).
