# StencilCalculus.jl

[![Build Status](https://github.com/vlc1/StencilCalculus.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/StencilCalculus.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vlc1.github.io/StencilCalculus.jl/dev/)

A small Computer Algebra System for expressions on N-D Cartesian meshes, whose
purpose is to **build `AbstractStencil`s by differentiation**. Part of a
three-package stack: it depends on [StencilCore](../StencilCore) for the term /
stencil types, and pairs with [StencilAssembly](../StencilAssembly) for
CSC assembly.

Define discrete fields as **slots**, compose component-wise and non-local
(shift) operators into a symbolic tree, then `simplify`, `differentiate`, and
lower the result to a compiled lazy array (`materialize`) or an assemblable
stencil (`build_stencil`).

## Pipeline at a glance

```julia
using StencilCalculus, StencilAssembly
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

## Design principle: type parameters are *structure*, values are *data*

The leitmotiv behind the data structures. **Type parameters encode structure**
— what drives dispatch, narrowing, and code generation; **runtime fields hold
data** — values computed or substituted later.

- **Shifts are structure.** They enter only through the DSL operators
  (`δ₊{D}`, …) whose axis `D` is known at compile time, and they fix the offset
  pattern, the codegen index arithmetic, and the `as_linear`/`as_star`
  narrowing. So `StaticPair{D,O}` / `StaticShift` live entirely at the type
  level, with a `+`/`-`/`*` algebra evaluated by the compiler.
- **General coefficients are data.** Arbitrary (possibly non-`isbits`,
  position-dependent) values would explode the type domain and wreck inference,
  so they live in `Const`'s runtime field and in the substituted slot arrays.
- **`0` and `1` are the exception — because they *are* structure.** The
  additive/multiplicative identities are the neutral and annihilating elements
  that make differentiation collapse and let `simplify` rewrite *by dispatch*
  (`Zero + x → x`, `One * x → x`, `_ * Zero → Zero`) with no runtime
  `iszero`/`isone` probing. That is why — and *only* why — they are promoted to
  the type level as `Zero{T}` / `One{T}`. A user-written `Const(0)` is left
  alone (the input is assumed reasonably simplified).

This split is what lets the same expression be reasoned about symbolically
(type-level shifts, dispatch-driven simplification) yet carry arbitrary runtime
coefficients into the materialized kernel.

## Install

Unregistered; clone [StencilCore](../StencilCore),
[StencilAssembly](../StencilAssembly), and this repo **side by side** (the
relative `[sources]` paths resolve them), then `]dev` this package.

## Design

The full design rationale lives in [`docs/cas.md`](docs/cas.md) (with the
package-split plan in [`StencilCore/docs/core.md`](../StencilCore/docs/core.md)).
