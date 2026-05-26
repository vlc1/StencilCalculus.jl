# Broadcast plumbing, SVector interception, indexing-as-shift sugar, and the
# non-local difference/sum DSL functors.
#
# Construction syntax: pointwise expressions are built from Julia's native
# broadcast (`.*`, `sin.(_)`, `(f .+ g) .* h`, …). The un-dotted Base operators
# (`*`, `+`, `sin`, …) are **not** overloaded on `AbstractPointwise`: any such
# call raises `MethodError`, signalling that the user should use the broadcast
# form. Scalar-land overloads in StencilCore keep working under un-dotted
# operators on `AbstractScalar`/`Number` operands.

# Dispatch-only union at the SVector / DSL boundary. Not threaded through
# `Pointwise.args` (which stays `Tuple{Vararg{AbstractPointwise}}`).
const PointwiseLike = Union{AbstractPointwise, AbstractScalar}

# --- Broadcast styles -------------------------------------------------------
#
# Two styles encode the rule "dotted ⇒ pointwise":
#   PointwiseStyle absorbs anything it mixes with and materializes to a
#                  Pointwise tree.
#   ScalarStyle    catches broadcast with no AbstractPointwise operand (pure
#                  AbstractScalar / Number) and errors at materialize — broadcast
#                  with no pointwise operand is forbidden.

struct PointwiseStyle <: Base.Broadcast.BroadcastStyle end
struct ScalarStyle    <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:AbstractPointwise}) = PointwiseStyle()
Base.BroadcastStyle(::Type{<:AbstractScalar})    = ScalarStyle()

# PointwiseStyle absorbs everything we mix with.
Base.BroadcastStyle(::PointwiseStyle, ::PointwiseStyle)                      = PointwiseStyle()
Base.BroadcastStyle(::PointwiseStyle, ::ScalarStyle)                         = PointwiseStyle()
Base.BroadcastStyle(::PointwiseStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = PointwiseStyle()

# ScalarStyle absorbs Numbers (DefaultArrayStyle{0}) but always errors at copy.
Base.BroadcastStyle(::ScalarStyle, ::ScalarStyle)                            = ScalarStyle()
Base.BroadcastStyle(::ScalarStyle, ::Base.Broadcast.DefaultArrayStyle{0})    = ScalarStyle()

# Do not Ref-wrap our types in the broadcast machinery — broadcastable returns
# the object itself so the style routing above is respected.
Base.broadcastable(x::AbstractPointwise) = x
Base.broadcastable(x::AbstractScalar)    = x

# Bypass `combine_axes` for our styles: our operands are symbolic (not arrays),
# so there are no axes to align — the default `instantiate` would error on
# `axes(::Slot)`. Returning `bc` unchanged short-circuits straight to `copy`.
Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{PointwiseStyle}) = bc
Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{ScalarStyle})    = bc

# Materialize a Broadcasted{PointwiseStyle} into a Pointwise tree. Nested
# Broadcasted args are recursively materialized; we intentionally do NOT
# `Broadcast.flatten` — each `.op` produces one Pointwise node so the symbolic
# tree reflects the user's syntax (and the simplify rules see it that way).
function Base.copy(bc::Base.Broadcast.Broadcasted{PointwiseStyle})
    args = map(_pointwise_arg, bc.args)
    Pointwise(bc.f, args)
end
_pointwise_arg(b::Base.Broadcast.Broadcasted) = copy(b)
_pointwise_arg(x)                              = asterm(x)

Base.copy(::Base.Broadcast.Broadcasted{ScalarStyle}) = throw(ArgumentError(
    "broadcast on AbstractScalar requires at least one AbstractPointwise " *
    "operand; use plain `*`/`+`/`sin`/… for scalar-land arithmetic"))

# --- SVector interception ---------------------------------------------------
# `SVector` is a constructor, not a broadcast op, so it's untouched by the
# broadcast machinery; intercept it directly to build a component-wise
# `Pointwise(SVector, args)`. Scalars/numbers wrap via `asterm`.
SVector(args::PointwiseLike...) = Pointwise(SVector, map(asterm, args))

# --- Indexing-as-shift sugar ------------------------------------------------
# `AbstractPointwise` is not `<: AbstractArray`, so `getindex` is free to mean
# "shift by a StaticShift" with no clash. `Fill` covers the `Zero` alias too
# (`Zero` is `Fill{<:Null}`), so no `t::Zero` overload is needed.
Base.getindex(t::Slot)                    = t
Base.getindex(t::Slot, s::StaticShift)    = Shifted(t, s)
Base.getindex(t::Shifted)                 = t
Base.getindex(t::Shifted, s::StaticShift) = Shifted(t.term, t.shift + s)
Base.getindex(t::Fill)                = t
Base.getindex(t::Fill, ::StaticShift) = t
Base.getindex(t::One)                 = t
Base.getindex(t::One, ::StaticShift)  = t
# AbstractScalars are position-independent (same value everywhere).
Base.getindex(s::AbstractScalar)                = s
Base.getindex(s::AbstractScalar, ::StaticShift) = s

# --- DSL functors -----------------------------------------------------------

"""
    FwdDiff{D} / BwdDiff{D} / FwdSum{D} / BwdSum{D}   (aliases δ₊, δ₋, σ₊, σ₋)

Non-local first-order operators along mesh axis `D`, applied by **calling the
type** on a term:

| operator      | `op{D}(t)`                  | meaning            |
|---------------|-----------------------------|--------------------|
| `δ₊` `FwdDiff`| `t[i+1] - t[i]` along `D`   | forward difference |
| `δ₋` `BwdDiff`| `t[i]   - t[i-1]`           | backward difference|
| `σ₊` `FwdSum` | `t[i+1] + t[i]`             | forward sum        |
| `σ₋` `BwdSum` | `t[i]   + t[i-1]`           | backward sum       |

Each builds a `Pointwise` over a `Shifted` leaf (e.g. `δ₊{1}(f) == f[ê₁] - f`); on a
`Number` they collapse to the local closed form (`0`, `0`, `2x`, `2x`).
Dimension-polymorphic — only the axis `D` is fixed.
"""
struct FwdDiff{D} <: Function end
struct BwdDiff{D} <: Function end
struct FwdSum{D}  <: Function end
struct BwdSum{D}  <: Function end

const δ₊ = FwdDiff
const δ₋ = BwdDiff
const σ₊ = FwdSum
const σ₋ = BwdSum

# Called directly on the type — `δ₊{1}(t)` ≡ `FwdDiff{1}(t)`.
(::Type{FwdDiff{D}})(t::AbstractPointwise) where {D} = Pointwise(-, (Shifted(t, SShift((SPair{D,  1}(),))), t))
(::Type{BwdDiff{D}})(t::AbstractPointwise) where {D} = Pointwise(-, (t, Shifted(t, SShift((SPair{D, -1}(),)))))
(::Type{FwdSum{D}})(t::AbstractPointwise)  where {D} = Pointwise(+, (Shifted(t, SShift((SPair{D,  1}(),))), t))
(::Type{BwdSum{D}})(t::AbstractPointwise)  where {D} = Pointwise(+, (t, Shifted(t, SShift((SPair{D, -1}(),)))))

# Scalar (Number) fall-throughs: the local closed-form values.
(::Type{<:FwdDiff})(x::Number) = zero(x)
(::Type{<:BwdDiff})(x::Number) = zero(x)
(::Type{<:FwdSum})(x::Number)  = 2x
(::Type{<:BwdSum})(x::Number)  = 2x

"""
    Diff(v)   (alias ∂)

"With respect to" functor: `∂(v)(e) == differentiate(e, v)`. `v` may be a
[`Slot`](@ref) (the result is a spatial `Stencil`) or a [`Var`](@ref)
(the result is an `AbstractPointwise` coefficient — a per-cell broadcast).
"""
struct Diff{V<:PointwiseLike}
    term::V
end

const ∂ = Diff

(d::Diff)(e::AbstractPointwise) = differentiate(e, d.term)
