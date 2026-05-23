# Component-wise operator overloads, SVector interception, indexing-as-shift
# sugar, and the non-local difference/sum DSL functors. All build `Term` /
# `Shifted` nodes; numeric literals are wrapped via `Fill(Const(…))` and bare
# AbstractScalars via `Fill(s)`. Operations that involve no AbstractTerm at
# all (Number↔AbstractScalar, AbstractScalar↔AbstractScalar, unary
# AbstractScalar) live in StencilCore and build `Scalar` nodes; the overloads
# below cover only the combinations that yield an `AbstractTerm`.

# Dispatch-only union for the operator boundary. Not threaded through
# `Term.args` (which stays `Tuple{Vararg{AbstractTerm}}`) or through simplify.
const TermLike = Union{AbstractTerm, AbstractScalar}

# Binary operators — every combination that includes at least one AbstractTerm.
for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractTerm,   b::AbstractTerm)   = Term($op, (a, b))
    @eval Base.$op(a::AbstractTerm,   b::AbstractScalar) = Term($op, (a, Fill(b)))
    @eval Base.$op(a::AbstractScalar, b::AbstractTerm)   = Term($op, (Fill(a), b))
    @eval Base.$op(a::AbstractTerm,   b::Number)         = Term($op, (a, Fill(Const(b))))
    @eval Base.$op(a::Number,         b::AbstractTerm)   = Term($op, (Fill(Const(a)), b))
end

# Unary operators on AbstractTerm. The AbstractScalar unary overloads live in
# StencilCore and produce `Scalar` nodes.
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractTerm) = Term($op, (a,))
end

# SVector interception: a vector field built component-wise. Scalars/numbers
# get wrapped via `asterm` so the args are uniformly AbstractTerm.
SVector(args::TermLike...) = Term(SVector, map(asterm, args))

# Indexing-as-shift sugar. `AbstractTerm` is not `<: AbstractArray`, so
# `getindex` is free to mean "shift by a StaticShift" with no clash.
Base.getindex(t::Slot)                    = t
Base.getindex(t::Slot, s::StaticShift)    = Shifted(t, s)
Base.getindex(t::Shifted)                 = t
Base.getindex(t::Shifted, s::StaticShift) = Shifted(t.term, t.shift + s)
# Position-independent term leaves: any shift is the identity.
Base.getindex(t::Fill)                = t
Base.getindex(t::Fill, ::StaticShift) = t
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

Each builds a `Term` over a `Shifted` leaf (e.g. `δ₊{1}(f) == f[ê₁] - f`); on a
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
(::Type{FwdDiff{D}})(t::AbstractTerm) where {D} = Term(-, (Shifted(t, SShift((SPair{D,  1}(),))), t))
(::Type{BwdDiff{D}})(t::AbstractTerm) where {D} = Term(-, (t, Shifted(t, SShift((SPair{D, -1}(),)))))
(::Type{FwdSum{D}})(t::AbstractTerm)  where {D} = Term(+, (Shifted(t, SShift((SPair{D,  1}(),))), t))
(::Type{BwdSum{D}})(t::AbstractTerm)  where {D} = Term(+, (t, Shifted(t, SShift((SPair{D, -1}(),)))))

# Scalar (Number) fall-throughs: the local closed-form values.
(::Type{<:FwdDiff})(x::Number) = zero(x)
(::Type{<:BwdDiff})(x::Number) = zero(x)
(::Type{<:FwdSum})(x::Number)  = 2x
(::Type{<:BwdSum})(x::Number)  = 2x

"""
    Diff(v)   (alias ∂)

"With respect to" functor: `∂(v)(e) == differentiate(e, v)`. `v` may be a
[`Slot`](@ref) (the result is a spatial `Stencil`) or a [`Symbolic`](@ref)
(the result is an `AbstractTerm` coefficient — a per-cell broadcast).
"""
struct Diff{V<:TermLike}
    term::V
end

const ∂ = Diff

(d::Diff)(e::AbstractTerm) = differentiate(e, d.term)
