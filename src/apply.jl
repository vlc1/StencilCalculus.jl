# `stencil * pointwise` — reserved operator surface for applying a discrete
# linear operator (an `AbstractStencil{S, T}`) to a discrete field (an
# `AbstractPointwise{U}`). The eltype-match rule (to be enforced when the
# bodies land) is `T === _unity_space(U)`: scalar-on-scalar for `U <: Number`,
# `SMatrix{N, N, F}`-on-`SVector{N, F}` for vector-valued fields. See
# StencilCore's `_unity_space` / `_jacobian_type`.
#
# **Shells only**: the three concrete-subtype methods below claim the dispatch
# surface and the module ownership (StencilCalculus, not StencilCore), but
# their bodies are stubs — calling any of them throws
# `"not yet implemented"`. The eltype-match check and the eager expansion to
# a `Pointwise`/`Shifted`/`Fill` tree land in a follow-up step.

using StencilCore: Stencil, LinearStencil, StarStencil

function Base.:*(::Stencil, ::AbstractPointwise)
    error("StencilCalculus: `*(::Stencil, ::AbstractPointwise)` is reserved " *
          "for stencil application but is not yet implemented")
end

function Base.:*(::LinearStencil, ::AbstractPointwise)
    error("StencilCalculus: `*(::LinearStencil, ::AbstractPointwise)` is " *
          "reserved for stencil application but is not yet implemented")
end

function Base.:*(::StarStencil, ::AbstractPointwise)
    error("StencilCalculus: `*(::StarStencil, ::AbstractPointwise)` is " *
          "reserved for stencil application but is not yet implemented")
end
