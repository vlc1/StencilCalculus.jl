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

@slot f Float64
@slot ψ Float64

# A symbolic grid expression (upwind advection):  out[i] = ψ[i] · (f[i+1] − f[i])
expr = ψ .* δ₊{1}(f)

# Inspect the per-cell kernel that materialize compiles:
print(code_string(expr; name = :advect))
#   function advect(args, i)
#       args.ψ[i] * (args.f[i + 1] - args.f[i])
#   end

# Evaluate it lazily over substituted arrays:
fv = rand(16); ψv = rand(16)
lazy = materialize(expr, (f = fv, ψ = ψv))     # LazyArray, axes (1:15,)

# Differentiate w.r.t. f → a row-anchored Stencil, then build a CSC matrix:
sst = differentiate(expr, f)                   # StencilCore.Stencil{RowAccess, Float64}
st  = build_stencil(sst, (ψ = ψv,))            # → LinearStencil{…, ColumnAccess}
A   = build(st, (1:15,), (1:15,))              # SparseMatrixCSC
```

## What it provides

- **Term types** `Slot{S,T}` (per-cell field), `Fill{T}` (scalar broadcast),
  the multiplicative identity `One`, the additive identity alias
  `Zero{T} = Fill{Null{T}}`, `Pointwise`, and `Shifted`; the (concrete)
  element type `T` is computed at construction. The `@slot` macro binds a
  variable to a leaf named after it
  (`@slot f` ≡ `f = Slot{:f, Float64}()`; the type defaults to `Float64`).
- **DSL**: pointwise expressions are built with broadcast syntax
  (`f .* g`, `sin.(f)`) — un-dotted Base operators are reserved for scalar-land
  (`τ * ψ`). `SVector` interception, indexing sugar `f[-2ê₁]` (shift by a
  `StaticShift`), and the difference / sum functors `δ₊`/`δ₋`/`σ₊`/`σ₋`
  (`FwdDiff`/`BwdDiff`/`FwdSum`/`BwdSum`) are not Base operators and stay as
  function-call syntax.
- **`simplify`** — a rule rewriter (shift composition/pushdown, identities on
  `Zero`/`One`, constant folding via `POINTWISE_DEFAULT_RULES`).
- **`differentiate(t, ::Slot)`** — row-anchored symbolic differentiation onto a
  `Stencil{RowAccess}`.
- **`materialize`** / **`code_string`** — codegen to a `LazyArray` (via
  RuntimeGeneratedFunctions) and the same kernel as inspectable source.
- **`build_stencil`** — convert (Row→Column), `materialize`, and narrow
  (`as_linear`/`as_star`, with optional `densify` padding) to an assemblable
  `LinearStencil`/`StarStencil`.

## Two CAS layers

StencilCalculus is built on top of a scalar CAS in StencilCore. Each layer has
its own type hierarchy; `Fill` is the one-way bridge that promotes a
position-independent scalar into the pointwise world.

| Concept | `AbstractScalar` (StencilCore) | `AbstractPointwise` (StencilCalculus) |
|---|---|---|
| Abstract supertype | `AbstractScalar{T}` | `AbstractPointwise{T}` |
| Named symbolic leaf | `Var{S,T}` | `Slot{S,T}` |
| Literal / constant leaf | `Constant{T}(val)` | `Fill(Constant(val))` |
| Interior (composite) node | `Scalar{F,A,T}` | `Pointwise{F,A,T}` |
| Additive identity (structural zero) | `Null{T}` | `Zero{T} = Fill{Null{T}}` |
| Multiplicative identity (structural one) | `Unity{T}` | `One{T}` |
| Construction syntax (binary) | `τ * ψ`, `τ + ψ` | `f .* g`, `f .+ g` |
| Construction syntax (unary) | `sin(τ)`, `-τ` | `sin.(f)`, `.-f` |
| Spatial shift node | — | `Shifted{Sh,T,U}` |
| Scalar-to-pointwise bridge | — (source) | `Fill{T}` (wraps `AbstractScalar`) |
| Constructor macro | `@var` | `@slot` |
| Default simplify rule set | `SCALAR_DEFAULT_RULES` | `POINTWISE_DEFAULT_RULES` |
| Register a derivative rule | `@scalar_rule` | `@pointwise_rule` |
| `differentiate` return type | `AbstractScalar` | `Stencil{RowAccess, T}` (T = common coef eltype) |
| Multiplication of two operands | `τ * ψ` builds `Scalar(*, …)` | `u * v` raises `MethodError` — `*` is reserved for `stencil * pointwise` (currently a stubbed shell) |

`Zero` is a thin alias for `Fill{Null{T}}` — it reuses scalar-land's `Null` as
the structural zero and lifts it into pointwise-land via the existing `Fill`
bridge. Like `Null`, its `T` is bool-shaped (the outer ctors map any concrete
value-space type through `_to_bool_shape`), and `eltype(Zero(Float64)) === Bool`
— promotion in surrounding arithmetic recovers the cell type. `One` and `Unity`
play the parallel multiplicative role: type-level identities that let
`simplify` and `differentiate` collapse algebraically by dispatch.

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
  so they live in `Constant`'s runtime field and in the substituted slot arrays.
- **`0` and `1` are the exception — because they *are* structure.** The
  additive/multiplicative identities are the neutral and annihilating elements
  that make differentiation collapse and let `simplify` rewrite *by dispatch*
  (`Zero .+ x → x`, `One .* x → x`, `_ .* Zero → Zero`) with no runtime
  `iszero`/`isone` probing. That is why — and *only* why — they are promoted to
  the type level as `Zero{T}` (`Fill{Null{T}}`) / `One{T}`. A user-written
  `Fill(Constant(0.0))` is now folded too (the simplify rule treats a literal
  `Fill(Constant(0))` as a real zero).

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
