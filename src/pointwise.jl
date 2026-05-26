# Concrete `AbstractPointwise{T}` subtypes — the leaves and interior nodes of a
# grid expression tree. `AbstractPointwise{T}` and the `_assert_concrete` guard live
# in StencilCore; `T` is the materialized element type (concrete; default
# `Float64`).
#
# Scalars (`AbstractScalar`) live in scalar-land and enter term expressions
# only via [`Fill`](@ref), so `Pointwise.args` stays `Tuple{Vararg{AbstractPointwise}}`.

"""
    Slot{S, T}()

Placeholder for a discrete field named `S` (a `Symbol`) whose cells hold
values of the concrete type `T` (default `Float64`). Substituted with an
`AbstractArray` at `materialize` and indexed per cell. Pointwise-side analogue of
[`Var`](@ref).
"""
struct Slot{S, T} <: AbstractPointwise{T}
    Slot{S, T}() where {S, T} = (_assert_concrete(:Slot, T); new{S, T}())
end
Slot{S}() where {S} = Slot{S, Float64}()

"""
    Zero{T}() / One{T}()

Type-level additive / multiplicative identities (structure, not data): they
let differentiation collapse and `simplify` rewrite by dispatch. Lower to
`zero(T)` / `one(T)`. Scalar-side analogues: [`Null`](@ref) / [`Unity`](@ref).
"""
struct Zero{T} <: AbstractPointwise{T}
    Zero{T}() where {T} = (_assert_concrete(:Zero, T); new{T}())
end
struct One{T} <: AbstractPointwise{T}
    One{T}() where {T} = (_assert_concrete(:One, T); new{T}())
end

"""
    Pointwise(fn, args::Tuple{Vararg{AbstractPointwise}})

Internal node applying `fn` to `args` component-wise. The element type
`T = Base.promote_op(fn, eltype.(args)...)` is computed **at construction**;
a `Union{}` result (e.g. genuine `SVector` inhomogeneity) is an
unconstructable term and throws.
"""
struct Pointwise{F, A<:Tuple{Vararg{AbstractPointwise}}, T} <: AbstractPointwise{T}
    fn::F
    args::A
    Pointwise{F, A, T}(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractPointwise}}, T} =
        new{F, A, T}(fn, args)
end

function Pointwise(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractPointwise}}}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && throw(ArgumentError(
        "unconstructable Pointwise: $(fn) over eltypes $(map(eltype, args)) has no " *
        "result type (Base.promote_op returned Union{})"))
    Pointwise{F, A, T}(fn, args)
end

"""
    Shifted(shift::StaticShift, term) / Shifted(term, shift::StaticShift)

A `term` read at the lattice offset `shift`. The element type is unchanged
(`eltype(term)`); the zero shift `ô` is the identity (returns `term`).
"""
struct Shifted{Sh<:StaticShift, T, U<:AbstractPointwise{T}} <: AbstractPointwise{T}
    shift::Sh
    term::U
    Shifted{Sh, T, U}(shift::Sh, term::U) where {Sh, T, U<:AbstractPointwise{T}} =
        new{Sh, T, U}(shift, term)
end

# Zero shift = identity (most specific).
Shifted(::StaticShift{Tuple{}}, term::AbstractPointwise) = term
Shifted(term::AbstractPointwise, ::StaticShift{Tuple{}}) = term
# Canonical (shift, term) and the (term, shift) sugar order.
Shifted(shift::StaticShift, term::AbstractPointwise{T}) where {T} =
    Shifted{typeof(shift), T, typeof(term)}(shift, term)
Shifted(term::AbstractPointwise, shift::StaticShift) = Shifted(shift, term)

"""
    Fill{T} <: AbstractPointwise{T}

Broadcast-to-grid bridge from scalar-land to term-land: wraps a single value
(a literal or an [`AbstractScalar`](@ref)) and presents it as a spatially-
invariant `AbstractPointwise`. The element type follows the wrapped value: for a
literal `T` it is `T`; for an `AbstractScalar` it is `eltype(scalar)`
(recursive), so e.g. `Fill(Symbolic{:τ, Float64}())` has eltype `Float64`
and arithmetic with a `Slot{:f, Float64}` promotes cleanly.
"""
struct Fill{T} <: AbstractPointwise{T}
    val::T
    Fill{T}(val) where {T} = (_assert_concrete(:Fill, T); new{T}(val))
end
Fill(v::AbstractScalar) = Fill{typeof(v)}(v)
Fill(v) = Fill{typeof(v)}(v)

# Specialize `eltype` recursively for Fills wrapping an AbstractScalar so that
# `Base.promote_op` against the scalar's underlying numeric type works.
Base.eltype(::Type{Fill{T}}) where {T<:AbstractScalar} = eltype(T)

# Promote a numeric literal / bare scalar to a term: literals canonicalise in
# scalar-land first (`Fill(Const(x))`); bare scalars wrap (`Fill(s)`); terms
# pass through.
asterm(t::AbstractPointwise)   = t
asterm(s::AbstractScalar) = Fill(s)
asterm(x::Number)         = Fill(Constant(x))

Base.convert(::Type{<:AbstractPointwise}, x::Number)         = Fill(Constant(x))
Base.convert(::Type{<:AbstractPointwise}, s::AbstractScalar) = Fill(s)

# --- Constructor macros: bind a variable to a leaf named after it -----------

"""
    @slot name [T = Float64]

Bind `name` to `Slot{:name, T}()`, taking the variable name as the field
symbol. `@slot f` ≡ `f = Slot{:f, Float64}()`; `@slot f Float32` ≡
`f = Slot{:f, Float32}()`. Scalar-side analogue: [`@var`](@ref).
"""
macro slot(name, T = :Float64)
    name isa Symbol || throw(ArgumentError("@slot expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Slot{$(QuoteNode(name)), $(esc(T))}())
end
