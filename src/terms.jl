# Concrete `AbstractTerm{T}` subtypes — the leaves and nodes of a grid
# expression tree. `AbstractTerm{T}` and `eltype(::Type{<:AbstractTerm{T}}) = T`
# live in StencilCore; `T` is the materialized element type.

"""
    Slot{S, T}()

Placeholder for a discrete field named `S` (a `Symbol`) whose cells hold
values of type `T` (default `Number`). Substituted with an `AbstractArray`
at `materialize` and indexed per cell.
"""
struct Slot{S, T} <: AbstractTerm{T} end
Slot{S}() where {S} = Slot{S, Number}()

"""
    Scalar{S, T}()

Named, runtime-substituted **broadcast** parameter `S` (e.g. a timestep) of
type `T` (default `Number`). Unlike [`Slot`](@ref) it materializes to a single
un-indexed value; like a constant it has zero derivative.
"""
struct Scalar{S, T} <: AbstractTerm{T} end
Scalar{S}() where {S} = Scalar{S, Number}()

"""
    Const(value)

A literal constant carrying its `value` in a runtime field (general data).
"""
struct Const{T} <: AbstractTerm{T}
    value::T
end

"""
    Zero{T}() / One{T}()

Type-level additive / multiplicative identities (structure, not data): they
let differentiation collapse and `simplify` rewrite by dispatch. Lower to
`zero(T)` / `one(T)`.
"""
struct Zero{T} <: AbstractTerm{T} end
struct One{T}  <: AbstractTerm{T} end

# Promote a numeric literal to a term.
Base.convert(::Type{<:AbstractTerm}, x::Number) = Const(x)
asterm(t::AbstractTerm) = t
asterm(x::Number) = Const(x)

"""
    Term(fn, args::Tuple{Vararg{AbstractTerm}})

Internal node applying `fn` to `args` component-wise. The element type
`T = Base.promote_op(fn, eltype.(args)...)` is computed **at construction**;
a `Union{}` result (e.g. genuine `SVector` inhomogeneity) is an
unconstructable term and throws.
"""
struct Term{F, A<:Tuple{Vararg{AbstractTerm}}, T} <: AbstractTerm{T}
    fn::F
    args::A
    Term{F, A, T}(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractTerm}}, T} =
        new{F, A, T}(fn, args)
end

function Term(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractTerm}}}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && throw(ArgumentError(
        "unconstructable Term: $(fn) over eltypes $(map(eltype, args)) has no " *
        "result type (Base.promote_op returned Union{})"))
    Term{F, A, T}(fn, args)
end

"""
    Shifted(shift::StaticShift, term) / Shifted(term, shift::StaticShift)

A `term` read at the lattice offset `shift`. The element type is unchanged
(`eltype(term)`); the zero shift `ô` is the identity (returns `term`).
"""
struct Shifted{Sh<:StaticShift, T, U<:AbstractTerm{T}} <: AbstractTerm{T}
    shift::Sh
    term::U
    Shifted{Sh, T, U}(shift::Sh, term::U) where {Sh, T, U<:AbstractTerm{T}} =
        new{Sh, T, U}(shift, term)
end

# Zero shift = identity (most specific).
Shifted(::StaticShift{Tuple{}}, term::AbstractTerm) = term
Shifted(term::AbstractTerm, ::StaticShift{Tuple{}}) = term
# Canonical (shift, term) and the (term, shift) sugar order.
Shifted(shift::StaticShift, term::AbstractTerm{T}) where {T} =
    Shifted{typeof(shift), T, typeof(term)}(shift, term)
Shifted(term::AbstractTerm, shift::StaticShift) = Shifted(shift, term)

# --- Constructor macros: bind a variable to a leaf named after it -----------

"""
    @slot name [T = Number]

Bind `name` to `Slot{:name, T}()`, taking the variable name as the field
symbol. `@slot f` ≡ `f = Slot{:f, Number}()`; `@slot f Float64` ≡
`f = Slot{:f, Float64}()`.
"""
macro slot(name, T = :Number)
    name isa Symbol || throw(ArgumentError("@slot expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Slot{$(QuoteNode(name)), $(esc(T))}())
end

"""
    @scalar name [T = Number]

Bind `name` to `Scalar{:name, T}()`. `@scalar τ` ≡ `τ = Scalar{:τ, Number}()`;
`@scalar τ Float64` ≡ `τ = Scalar{:τ, Float64}()`.
"""
macro scalar(name, T = :Number)
    name isa Symbol || throw(ArgumentError("@scalar expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Scalar{$(QuoteNode(name)), $(esc(T))}())
end

"""
    @const name value

Bind `name` to `Const(value)`. `@const α 1` ≡ `α = Const(1)`. Defined via the
`var"@const"` function form because `const` is a reserved word (so a plain
`macro const` would not parse).
"""
function var"@const"(__source__::LineNumberNode, __module__::Module, name, value)
    name isa Symbol || throw(ArgumentError("@const expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Const($(esc(value))))
end
