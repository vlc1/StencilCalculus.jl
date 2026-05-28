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
    IdentityStencil{T, U}()
    IdentityStencil(U::Type) / IdentityStencil(x)

Type-level multiplicative identity for [`AbstractPointwise`](@ref): the
pointwise-side analogue of scalar-side [`Unity`](@ref), now reified as a
*stencil* (so it can serve as the neutral element of `*(stencil, pointwise)`
without breaking the `AbstractPointwise` × `AbstractPointwise` discipline).

The parameter `T` is **bool-shaped** (`Bool` or `AbstractArray{Bool}`),
mirroring `Unity`'s discipline so that promotion in surrounding arithmetic
recovers the value type without pinning a Stencil's coefficient eltype. The
second parameter `U` is the *value space* (e.g. `Float64`, `SMatrix{N,N,F}`)
recovered at materialize time via `one(U)`.

The outer constructors map any concrete value-space type `U` to its bool-
shape `T = _to_bool_shape(_unity_space(U))`, so
`IdentityStencil(Float64) === IdentityStencil{Bool, Float64}()` and
`eltype(IdentityStencil(Float64)) === Bool`. Materializes to `one(U)` —
e.g. `1.0` for `U = Float64`.

See also [`Zero`](@ref) for the additive identity.
"""
struct IdentityStencil{T, U} <: AbstractPointwise{T}
    function IdentityStencil{T, U}() where {T, U}
        _assert_bool_shape(:IdentityStencil, T)
        applicable(one, U) || throw(ArgumentError(
            "IdentityStencil{T, U} requires `one(U)` to be defined (a " *
            "square-scalar shape); got U=$U"))
        new{T, U}()
    end
end
IdentityStencil(::Type{U}) where {U} = IdentityStencil{_to_bool_shape(_unity_space(U)), _unity_space(U)}()
IdentityStencil(::U)       where {U} = IdentityStencil{_to_bool_shape(_unity_space(U)), _unity_space(U)}()

# Value-space accessor for codegen / display.
_value_space(::Type{<:IdentityStencil{T, U}}) where {T, U} = U
_value_space(s::IdentityStencil) = _value_space(typeof(s))

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

"""
    Zero{T} = Fill{Null{T}}
    Zero(T::Type) / Zero(x)

Type-level additive identity for [`AbstractPointwise`](@ref), defined as the
`Fill` of a scalar-side [`Null`](@ref). The parameter `T` is bool-shaped
(`Bool` or `AbstractArray{Bool}`), mirroring `Null`'s discipline; the outer
constructors map any concrete value-space type to its bool shape, so
`Zero(Float64) === Fill(Null{Bool}())` and `eltype(Zero(Float64)) === Float64`
via the `Fill{<:AbstractScalar}` eltype specialization above.

Materializes to a broadcast of `zero(T)` (Bool false, etc.) — promotion in
surrounding arithmetic recovers the cell type, exactly as for [`Null`](@ref)
in scalar-land.
"""
const Zero{T} = Fill{Null{T}}

Zero(T::Type)       = Fill(Null(T))
Zero(::T) where {T} = Fill(Null(T))

"""
    DiagonalStencil(t::AbstractPointwise{T}) -> DiagonalStencil{T, typeof(t)}

A *diagonal* stencil that, when applied to another `AbstractPointwise{U}`
via `*(d, p)`, broadcasts the wrapped term elementwise: `d * p ==
Pointwise(*, (d.term, p))`. Pointwise-side counterpart of a Number
coefficient pinned to a column. The eltype `T = eltype(t)` is the value
space of the diagonal entries — Number (scalar-on-scalar) or a square
`SMatrix{N, N, F}` (matrix-on-vector). Requires `one(T)` to be defined.

Pure pointwise terms (e.g. `Slot`, `Pointwise(*, …)`) cannot be used as a
left operand of `*` directly — by design, multiplication on pointwise terms
is reserved for stencil application. Wrap in `DiagonalStencil` to opt in.
"""
struct DiagonalStencil{T, A<:AbstractPointwise} <: AbstractPointwise{T}
    term::A
    # We deliberately do NOT tie A's type parameter to T (`A<:AbstractPointwise{T}`):
    # for `Fill(τ::Var{:τ, Float64})`, `A = Fill{Var{:τ, Float64}}` carries the
    # *payload* type at the type-parameter level while `eltype` reports `Float64`
    # via the `Fill{<:AbstractScalar}` specialization. We honor `eltype(t)` as the
    # source of truth and verify it inside the inner ctor.
    function DiagonalStencil{T, A}(t::A) where {T, A<:AbstractPointwise}
        eltype(t) === T || throw(ArgumentError(
            "DiagonalStencil{T} requires eltype(t) === T; got eltype = " *
            "$(eltype(t)), T = $T"))
        applicable(one, T) || throw(ArgumentError(
            "DiagonalStencil{T} requires T to be a square-scalar shape " *
            "(one(T) defined); got T=$T"))
        new{T, A}(t)
    end
end
DiagonalStencil(t::AbstractPointwise) = DiagonalStencil{eltype(t), typeof(t)}(t)

# Bool-shape structural markers: a `Zero` (== `Fill{<:Null}`), a
# `Fill{<:Unity}`, or an `IdentityStencil` — all materialize to
# `zero(T)`/`one(T)` of any surrounding T via promotion. Pointwise-/Shifted-
# trees over wildcards-only are themselves wildcards (chain-rule expressions
# like `Pointwise(-, (IdentityStencil(T),))` produced by `derivative(-, …)`
# carry the same Bool-shape discipline through the Stencil-eltype check).
_is_eltype_wildcard(::Fill{<:Union{Null, Unity}}) = true
_is_eltype_wildcard(::IdentityStencil)            = true
_is_eltype_wildcard(p::Pointwise)                 = all(_is_eltype_wildcard, p.args)
_is_eltype_wildcard(s::Shifted)                   = _is_eltype_wildcard(s.term)

# Promote a value to a term: AbstractPointwise passes through; AbstractScalar
# wraps in Fill; anything else (Number, SMatrix, …) wraps as Fill(Constant(x))
# so that the scalar algebra stays consistent and rule_fill_collapse can fold.
asterm(t::AbstractPointwise)   = t
asterm(s::AbstractScalar) = Fill(s)
asterm(x)                 = Fill(Constant(x))

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
