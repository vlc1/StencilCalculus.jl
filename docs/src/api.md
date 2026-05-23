# API reference

## Terms, operators, and operations

```@autodocs
Modules = [StencilCalculus]
Private = false
```

## Re-exported from StencilCore

The offset vocabulary, the `AbstractTerm` supertype, and the **scalar algebra**
(`AbstractScalar`, `Symbolic`, `Const`, `Null`, `Unity`, `Scalar`, plus the
`@symbolic` / `@const` macros) are re-exported from
[StencilCore](https://vlc1.github.io/StencilCore.jl/dev/) (full docs there).
The `Stencil` produced by [`differentiate`](@ref) and the narrowing functions
live there too.

```@docs
AbstractTerm
AbstractScalar
Symbolic
Const
Null
Unity
Scalar
StaticPair
SPair
StaticShift
SShift
ô
Stencil
as_linear
as_star
```
