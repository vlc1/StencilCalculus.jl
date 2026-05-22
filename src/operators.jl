# Component-wise operator overloads, SVector interception, indexing-as-shift
# sugar, and the non-local difference/sum DSL functors. All build `Term` /
# `Shifted` nodes; numeric literals are wrapped via `asterm`.

# Binary operators (term⊗term, term⊗number, number⊗term).
for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractTerm, b::AbstractTerm) = Term($op, (a, b))
    @eval Base.$op(a::AbstractTerm, b::Number)       = Term($op, (a, asterm(b)))
    @eval Base.$op(a::Number,       b::AbstractTerm) = Term($op, (asterm(a), b))
end

# Unary operators.
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractTerm) = Term($op, (a,))
end

# SVector interception: a vector field built component-wise from terms.
SVector(args::AbstractTerm...) = Term(SVector, args)

# Indexing-as-shift sugar. `AbstractTerm` is not `<: AbstractArray`, so
# `getindex` is free to mean "shift by a StaticShift" with no clash.
Base.getindex(t::Slot)                    = t
Base.getindex(t::Slot, s::StaticShift)    = Shifted(t, s)
Base.getindex(t::Shifted)                 = t
Base.getindex(t::Shifted, s::StaticShift) = Shifted(t.term, t.shift + s)
# A Scalar is position-independent, so shifting it is the identity (consistent
# with simplify's rule_shift_const).
Base.getindex(t::Scalar)                  = t
Base.getindex(t::Scalar, ::StaticShift)   = t

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
    Diff(v::AbstractTerm)   (alias ∂)

"With respect to" functor: `∂(v)(e) == differentiate(e, v)`. With a `Slot`
variable the result is a `Stencil` (it has spatial offsets); with a `Scalar`
it is an `AbstractTerm` (a scalar has no offsets). For example,
`∂(τ)(τ * f) == f`.
"""
struct Diff{T<:AbstractTerm}
    term::T
end

const ∂ = Diff

(d::Diff)(e::AbstractTerm) = differentiate(e, d.term)
