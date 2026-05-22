# Code generation: lower a (normal-form) term to a compiled per-cell kernel
# (`LazyArray`), and `code_string` to inspect that kernel as Julia source.
# Both share the `Expr` builder, so what runs is what you read.

# --- LazyArray -------------------------------------------------------------

"""
    LazyArray{T, N, F, Args<:NamedTuple} <: AbstractArray{T, N}

Read-only array view of a materialized term: `fn(args, I...)` computes the
element at `I`, where `fn` is a generated per-cell kernel and `args` is the
substitution `NamedTuple`. `axes` is the term's valid index box (each
substituted array's axes shrunk inward by the term's shift footprint).
"""
struct LazyArray{T, N, F, Args<:NamedTuple} <: AbstractArray{T, N}
    fn::F
    args::Args
    ax::NTuple{N, UnitRange{Int}}
end

Base.size(la::LazyArray)             = map(length, la.ax)
Base.axes(la::LazyArray)             = la.ax
Base.IndexStyle(::Type{<:LazyArray}) = IndexCartesian()

@inline function Base.getindex(la::LazyArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(la, I...)
    la.fn(la.args, I...)
end

# --- shared Expr builder ---------------------------------------------------

# Index variables: i, j, k for N ≤ 3, else i1, i2, …
_idxvars(N::Integer) = N <= 3 ? [:i, :j, :k][1:N] : [Symbol(:i, d) for d in 1:N]

_slotref(S, idxs)         = Expr(:ref, Expr(:., :args, QuoteNode(S)), idxs...)   # args.S[idxs...]
_body_expr(::Slot{S}, idx)   where {S} = _slotref(S, idx)
_body_expr(::Scalar{S}, idx) where {S} = Expr(:., :args, QuoteNode(S))           # args.S
_body_expr(c::Const, idx)              = c.value
_body_expr(::Zero{T}, idx)   where {T} = Expr(:call, :zero, T)
_body_expr(::One{T}, idx)    where {T} = Expr(:call, :one, T)
_body_expr(t::Term, idx)               = Expr(:call, nameof(t.fn), (_body_expr(a, idx) for a in t.args)...)
function _body_expr(t::Shifted, idx)
    t.term isa Slot || error("materialize/code_string expect normal-form input (shifts on slots)")
    _slotref(_slotsym(t.term),
             [_shifted_ix(idx[d], _axis_offset(t.shift, d)) for d in eachindex(idx)])
end
_shifted_ix(v, o) = o == 0 ? v : Expr(:call, :+, v, o)

# --- leaf / access collection ----------------------------------------------
_collect_acc!(a, s::Slot{S}) where {S} = (push!(a, (S, ô)); a)
_collect_acc!(a, ::Union{Scalar, Const, Zero, One}) = a
_collect_acc!(a, t::Shifted) = (push!(a, (_slotsym(t.term), t.shift)); a)
_collect_acc!(a, t::Term)    = (foreach(x -> _collect_acc!(a, x), t.args); a)
_accesses(t) = _collect_acc!(Tuple{Symbol, StaticShift}[], t)

# --- materialize -----------------------------------------------------------

"""
    materialize(term::AbstractTerm, pairs::NamedTuple) -> LazyArray

Substitute the slots/scalars named in `pairs` into `term` and lower it to a
compiled `LazyArray`. The grid rank `N` is the substituted arrays' `ndims`
(which must agree); the result's `axes` are their axes intersected and shrunk
inward by the term's shift footprint (no implicit broadcasting). Shifts become
index arithmetic in the generated kernel.
"""
function materialize(term::AbstractTerm, pairs::NamedTuple)
    t = simplify(term)
    acc = _accesses(t)
    isempty(acc) && throw(ArgumentError(
        "expression has no Slot to materialize against (only scalars/constants)"))
    slot_syms = unique(first.(acc))
    N = ndims(pairs[first(slot_syms)])
    for s in slot_syms
        ndims(pairs[s]) == N || throw(ArgumentError(
            "slots disagree on ndims: $(s) has $(ndims(pairs[s])), expected $(N)"))
    end

    # Valid index box: per axis, intersect (shifted) array axes.
    axs = ntuple(N) do d
        lo, hi = typemin(Int), typemax(Int)
        for (sym, shift) in acc
            o = _axis_offset(shift, d)
            r = axes(pairs[sym], d)
            lo = max(lo, first(r) - o)
            hi = min(hi, last(r) - o)
        end
        lo:hi
    end

    idx = _idxvars(N)
    fnexpr = Expr(:function, Expr(:tuple, :args, idx...), _body_expr(t, idx))
    fn = @RuntimeGeneratedFunction(fnexpr)

    Tinf = Base.infer_return_type(fn, Tuple{typeof(pairs), ntuple(_ -> Int, N)...})
    T = isconcretetype(Tinf) ? Tinf : eltype(t)

    LazyArray{T, N, typeof(fn), typeof(pairs)}(fn, pairs, axs)
end

# --- code_string -----------------------------------------------------------

function _ndims_hint(t)
    acc = _accesses(simplify(t))
    isempty(acc) ? 1 : max(1, maximum(a -> _shift_maxaxis(a[2]), acc))
end

"""
    code_string(term::AbstractTerm; name = :kernel, ndims = <hint>) -> String

Render the per-cell kernel for `term` as Julia source — the same `Expr`
`materialize` compiles, wrapped in a named function definition. `ndims`
defaults to the largest shifted axis (≥ 1). For inspection / dumping to a file.
"""
function code_string(term::AbstractTerm; name::Symbol = :kernel, ndims::Integer = _ndims_hint(term))
    t = simplify(term)
    idx = _idxvars(ndims)
    fnexpr = Expr(:function, Expr(:call, name, :args, idx...), _body_expr(t, idx))
    string(fnexpr)
end
