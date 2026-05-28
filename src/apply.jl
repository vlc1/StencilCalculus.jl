# `stencil * pointwise` — applying a discrete linear operator (an
# `AbstractStencil{T}`) to a discrete field (an `AbstractPointwise{U}`). The
# eltype-match rule is `T === _unity_space(U)`: scalar-on-scalar for
# `U <: Number`, `SMatrix{N, N, F}`-on-`SVector{N, F}` for vector-valued
# fields. See StencilCore's `_unity_space` / `_jacobian_type`.
#
# Diagonal stencils (the new `IdentityStencil` and `DiagonalStencil`, both
# `<: AbstractPointwise`) have working bodies; the three NeighborhoodStencil
# subtypes (`Stencil`, `LinearStencil`, `StarStencil`) remain shells whose
# bodies land in a follow-up step.

using StencilCore: Stencil, LinearStencil, StarStencil

# IdentityStencil application: the pointwise operand is returned unchanged.
# The Bool-shape eltype `T_id` of the IdentityStencil must match the bool-
# shape of `_unity_space(U)` for `U = eltype(p)`.
function Base.:*(::IdentityStencil{T_id}, p::AbstractPointwise{U}) where {T_id, U}
    T_id === _to_bool_shape(_unity_space(U)) || throw(ArgumentError(
        "IdentityStencil{$T_id} cannot apply to AbstractPointwise{$U}: " *
        "need T_id === _to_bool_shape(_unity_space(U)) = " *
        "$(_to_bool_shape(_unity_space(U)))"))
    p
end

# DiagonalStencil application: elementwise multiplication with the wrapped
# term. Eltype-match: `T === _unity_space(U)`.
function Base.:*(d::DiagonalStencil{T}, p::AbstractPointwise{U}) where {T, U}
    T === _unity_space(U) || throw(ArgumentError(
        "DiagonalStencil{$T} cannot apply to AbstractPointwise{$U}: " *
        "need T === _unity_space(U) = $(_unity_space(U))"))
    Pointwise(*, (d.term, p))
end

# NeighborhoodStencil shells — bodies TBD.
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
