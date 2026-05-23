# Symbolic differentiation of a (normal-form) grid expression with respect to
# a Slot (→ `StencilCore.Stencil{RowAccess}`) or a Symbolic scalar parameter
# (→ `AbstractTerm`). Methods on the same `differentiate` / `derivative`
# generics that StencilCore's scalar-side machinery uses — disjoint dispatch
# on `Vararg{AbstractTerm}` vs `Vararg{AbstractScalar}`.
#
# The Slot path is **row-anchored**: the coefficient at offset `σ` is
# `∂F/∂(f[σ])` evaluated at the equation/row index — no shifts are injected
# (the Row→Column conversion lives in the assembly bridge). Neutral elements
# are the type-level `Zero`/`One`, so the chain rule collapses under
# `simplify`.

# --- Derivative table (frule-shape: ∂f/∂(arg i); term-side, Vararg{AbstractTerm}) ---

_pe(args) = mapreduce(eltype, promote_type, args)   # promoted element type

derivative(::typeof(+), ::Val,    args::Vararg{AbstractTerm}) = One{_pe(args)}()
derivative(::typeof(-), ::Val{1}, x::AbstractTerm)            = Term(-, (One{eltype(x)}(),))
derivative(::typeof(-), ::Val{1}, x::AbstractTerm, y::AbstractTerm) = One{_pe((x, y))}()
derivative(::typeof(-), ::Val{2}, x::AbstractTerm, y::AbstractTerm) = Term(-, (One{_pe((x, y))}(),))
derivative(::typeof(*), ::Val{1}, x::AbstractTerm, y::AbstractTerm) = y
derivative(::typeof(*), ::Val{2}, x::AbstractTerm, y::AbstractTerm) = x
derivative(::typeof(/), ::Val{1}, x::AbstractTerm, y::AbstractTerm) = Term(/, (One{_pe((x, y))}(), y))
derivative(::typeof(/), ::Val{2}, x::AbstractTerm, y::AbstractTerm) = Term(-, (Term(/, (x, Term(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x::AbstractTerm, n::AbstractTerm) =
    Term(*, (n, Term(^, (x, Term(-, (n, One{_pe((n,))}()))))))
derivative(::typeof(sin),  ::Val{1}, x::AbstractTerm) = Term(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x::AbstractTerm) = Term(-, (Term(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x::AbstractTerm) = Term(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x::AbstractTerm) = Term(/, (One{_pe((x,))}(), x))
derivative(::typeof(sqrt), ::Val{1}, x::AbstractTerm) =
    Term(/, (One{_pe((x,))}(), Term(*, (Fill(Const(2)), Term(sqrt, (x,))))))
derivative(f, ::Val, args::Vararg{AbstractTerm}) =
    throw(ArgumentError("no term derivative rule for $(f)"))

# --- Per-offset contribution collection (Slot path) ------------------------

const _Contrib = Pair{StaticShift, AbstractTerm}

_slotsym(::Slot{S}) where {S} = S

# Position-independent leaves (Fill, Zero, One) have no Slot dependence.
_diff(::Union{Fill, Zero, One}, ::Slot) = _Contrib[]

_diff(::Slot{S2, T}, ::Slot{S}) where {S2, T, S} =
    S2 === S ? _Contrib[ô => One{T}()] : _Contrib[]

function _diff(sh::Shifted, ::Slot{S}) where {S}
    sl = sh.term
    sl isa Slot || throw(ArgumentError(
        "differentiate expects normal-form input (shifts on slots); got a " *
        "Shifted over $(typeof(sl)). Call simplify first."))
    _slotsym(sl) === S ? _Contrib[sh.shift => One{eltype(sh)}()] : _Contrib[]
end

function _diff(t::Term, v::Slot)
    out = _Contrib[]
    for (i, arg) in enumerate(t.args)
        sub = _diff(arg, v)
        isempty(sub) && continue
        dfn = derivative(t.fn, Val(i), t.args...)
        for (sh, partial) in sub
            push!(out, sh => simplify(Term(*, (dfn, partial))))
        end
    end
    return out
end

# --- Differentiation w.r.t. a Symbolic (scalar parameter) ------------------
# A Symbolic appears in a term only inside a `Fill`, so the derivative
# collapses to a per-cell broadcast coefficient. Walking strategy: like the
# Slot path, but without spatial offsets — accumulate one AbstractTerm.

_diff_scalar(::Union{Slot, Shifted, Zero, One}, ::Symbolic) = nothing
function _diff_scalar(f::Fill{<:AbstractScalar}, v::Symbolic)
    d = differentiate(f.val, v)
    d isa Null ? nothing : Fill(d)
end
_diff_scalar(::Fill, ::Symbolic) = nothing                      # literal Fill: no dependence

function _diff_scalar(t::Term, v::Symbolic)
    out = nothing
    for (i, arg) in enumerate(t.args)
        sub = _diff_scalar(arg, v)
        sub === nothing && continue
        dfn     = derivative(t.fn, Val(i), t.args...)
        contrib = simplify(Term(*, (dfn, sub)))
        out = (out === nothing) ? contrib : simplify(Term(+, (out, contrib)))
    end
    return out
end

# --- Grouping, reverse-lex ordering, assembly (Slot path) ------------------

# Sum coefficients sharing an offset; preserve first-seen order.
function _group(contribs::Vector{_Contrib})
    shifts = StaticShift[]
    coefs  = AbstractTerm[]
    for (sh, c) in contribs
        idx = findfirst(==(sh), shifts)
        if idx === nothing
            push!(shifts, sh); push!(coefs, c)
        else
            coefs[idx] = simplify(Term(+, (coefs[idx], c)))
        end
    end
    return shifts, coefs
end

_shift_maxaxis(::StaticShift{Tuple{}}) = 0
_shift_maxaxis(s::StaticShift) = maximum(dim, s.pairs)

_axis_offset(::StaticShift{Tuple{}}, ::Int) = 0
function _axis_offset(s::StaticShift, d::Int)
    for p in s.pairs
        dim(p) == d && return offset(p)
    end
    return 0
end

"""
    differentiate(t::AbstractTerm, ::Slot{S}) -> StencilCore.Stencil{RowAccess}

Differentiate `t` with respect to the field named `S` (matched on the symbol
only). Returns a row-anchored `Stencil` in structure-of-arrays form: reverse-
lex-ordered offsets `shifts` and a parallel tuple `terms` whose `k`-th entry is
the coefficient `∂t/∂(f[σ_k])`. Throws if `t` does not depend on `S`
(identically-zero derivative).
"""
function differentiate(t::AbstractTerm, v::Slot{S}) where {S}
    shifts, coefs = _group(_diff(simplify(t), v))
    keep = findall(c -> !_is_term_zero(c), coefs)
    shifts, coefs = shifts[keep], coefs[keep]
    isempty(shifts) && throw(ArgumentError(
        "the expression does not depend on Slot :$(S); its derivative is " *
        "identically zero"))
    N = maximum(_shift_maxaxis, shifts)
    key(s) = ntuple(k -> _axis_offset(s, N - k + 1), N)   # axis N most significant
    perm = sortperm(shifts; by = key)
    shifts, coefs = shifts[perm], coefs[perm]
    Stencil(RowAccess, (shifts...,), (coefs...,))
end

"""
    differentiate(t::AbstractTerm, ::Symbolic{S}) -> AbstractTerm

Differentiate `t` with respect to the named scalar parameter `S`. A Symbolic
appears in `t` only inside a [`Fill`](@ref), so the derivative is a per-cell
broadcast coefficient (a single `AbstractTerm`, not a `Stencil`). Throws if
`t` does not depend on `S`.
"""
function differentiate(t::AbstractTerm, v::Symbolic{S}) where {S}
    out = _diff_scalar(simplify(t), v)
    out === nothing && throw(ArgumentError(
        "the expression does not depend on Symbolic :$(S); its derivative " *
        "is identically zero"))
    simplify(out)
end
