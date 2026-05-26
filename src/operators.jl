# Component-wise operator overloads, SVector interception, indexing-as-shift
# sugar, and the non-local difference/sum DSL functors. All build `Pointwise` /
# `Shifted` nodes; numeric literals are wrapped via `Fill(Const(…))` and bare
# AbstractScalars via `Fill(s)`. Operations that involve no AbstractPointwise at
# all (Number↔AbstractScalar, AbstractScalar↔AbstractScalar, unary
# AbstractScalar) live in StencilCore and build `Scalar` nodes; the overloads
# below cover only the combinations that yield an `AbstractPointwise`.

# Dispatch-only union for the operator boundary. Not threaded through
# `Pointwise.args` (which stays `Tuple{Vararg{AbstractPointwise}}`) or through simplify.
const PointwiseLike = Union{AbstractPointwise, AbstractScalar}

# Binary operators — every combination that includes at least one AbstractPointwise.
for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractPointwise,   b::AbstractPointwise)   = Pointwise($op, (a, b))
    @eval Base.$op(a::AbstractPointwise,   b::AbstractScalar) = Pointwise($op, (a, Fill(b)))
    @eval Base.$op(a::AbstractScalar, b::AbstractPointwise)   = Pointwise($op, (Fill(a), b))
    @eval Base.$op(a::AbstractPointwise,   b::Number)         = Pointwise($op, (a, Fill(Constant(b))))
    @eval Base.$op(a::Number,         b::AbstractPointwise)   = Pointwise($op, (Fill(Constant(a)), b))
end

# Unary operators on AbstractPointwise. The AbstractScalar unary overloads live in
# StencilCore and produce `Scalar` nodes.
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractPointwise) = Pointwise($op, (a,))
end

# SVector interception: a vector field built component-wise. Scalars/numbers
# get wrapped via `asterm` so the args are uniformly AbstractPointwise.
SVector(args::PointwiseLike...) = Pointwise(SVector, map(asterm, args))

# Indexing-as-shift sugar. `AbstractPointwise` is not `<: AbstractArray`, so
# `getindex` is free to mean "shift by a StaticShift" with no clash.
Base.getindex(t::Slot)                    = t
Base.getindex(t::Slot, s::StaticShift)    = Shifted(t, s)
Base.getindex(t::Shifted)                 = t
Base.getindex(t::Shifted, s::StaticShift) = Shifted(t.term, t.shift + s)
# Position-independent term leaves: any shift is the identity.
Base.getindex(t::Fill)                = t
Base.getindex(t::Fill, ::StaticShift) = t
# AbstractScalars are position-independent (same value everywhere).
Base.getindex(s::AbstractScalar)              = s
Base.getindex(s::AbstractScalar, ::StaticShift) = s
Base.getindex(t::Zero)                = t
Base.getindex(t::Zero, ::StaticShift) = t
Base.getindex(t::One)                 = t
Base.getindex(t::One, ::StaticShift)  = t

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
