# StencilCalculus.jl

StencilCalculus is a small Computer Algebra System for expressions on N-D
Cartesian grids, with one purpose: **build sparse stencils by differentiating
them**. It is the symbolic front of a small stack — it depends on
[StencilCore](https://vlc1.github.io/StencilCore.jl/dev/) for the term and
stencil types, and hands its results to
[StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/) for assembly
into a `SparseMatrixCSC`.

## Why differentiate a grid expression?

On a structured mesh, the operations of interest often define a field `ψ`
*implicitly* as the root of a (possibly nonlinear) system

```math
F(\psi, \phi) = 0,
```

for instance a compact finite-difference scheme. Solving it with a Newton-type
method needs the **Jacobians** of `F`: the *implicit* Jacobian `∂F/∂ψ` and the
*explicit* Jacobian `−∂F/∂φ`. Because the same formula is applied at every node,
these Jacobians are sparse with a repeating pattern — a **stencil**.

Writing those stencils by hand is tedious and error-prone, especially with
variable coefficients. Instead, you write `F` as a symbolic grid expression and
let the package **differentiate** it into a stencil.

!!! note "“Jacobian”, you said?"
    Strictly, on a mesh of dimension greater than one these objects are not
    Jacobians until the unknowns are flattened to a linear numbering — which is
    exactly what assembling into a `SparseMatrixCSC` does. We use the word
    loosely; the package makes the flattening precise.

## How it fits together

The data structures follow one rule, inherited from StencilCore — **type
parameters are structure; values are data**: lattice offsets live at the type
level, coefficients are ordinary (or lazy) arrays.

1. **Build** an expression from [`Slot`](@ref)s (discrete fields),
   [`Scalar`](@ref)s (parameters), pointwise operators, and the non-local
   difference/sum functors [`δ₊`/`δ₋`/`σ₊`/`σ₋`](@ref FwdDiff). Index sugar
   `f[-ê₁]` shifts a field.
2. [`simplify`](@ref) the expression to normal form (shifts pushed onto the
   leaves; identities collapsed).
3. [`differentiate`](@ref) with respect to a `Slot` → a row-anchored
   `Stencil`.
4. [`build_stencil`](@ref) converts it to a column-anchored, assemblable
   `LinearStencil`/`StarStencil`, materializing the coefficients.
5. Assemble the result with
   [StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/).

For inspection, [`materialize`](@ref) compiles an expression into a read-only
lazy array, and [`code_string`](@ref) renders the same per-cell kernel as Julia
source.

## Quickstart

```julia
using StencilCalculus, StencilAssembly
using StaticArrays

@slot f Float64
@slot ψ Float64

# Upwind advection:  out[i] = ψ[i] * (f[i+1] - f[i])
expr = ψ * δ₊{1}(f)

print(code_string(expr; name = :advect))
#   function advect(args, i)
#       args.ψ[i] * (args.f[i + 1] - args.f[i])
#   end

# Differentiate w.r.t. f, then build and assemble the Jacobian.
sst = differentiate(expr, f)              # a row-anchored Stencil
st  = build_stencil(sst, (ψ = rand(16),)) # → an assemblable LinearStencil
A   = build(st, (1:15,), (1:15,))         # SparseMatrixCSC
```

See the [Guide](@ref) for the Laplacian and the full end-to-end loop, and the
[API reference](@ref).
