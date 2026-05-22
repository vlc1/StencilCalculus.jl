# Symbolic differentiation of a (normal-form) grid expression with respect to
# a Slot, producing a `StencilCore.Stencil{RowAccess}`.
#
# The result is **row-anchored**: the coefficient at offset `σ` is `∂F/∂(f[σ])`
# evaluated at the equation/row index — no shifts are injected (the Row→Column
# conversion lives in the assembly bridge). Neutral elements are the type-level
# `Zero`/`One`, so the chain rule collapses under `simplify`.

# --- Derivative table (frule-shape: ∂f/∂(arg i)) ---------------------------

"""
    derivative(f, ::Val{i}, args...) -> AbstractTerm

The symbolic partial derivative `∂f/∂(argᵢ)` of a primitive `f` applied to
`args` (in the ChainRules `frule` style). Returns a term; the `0`/`1` cases use
the type-level `Zero`/`One`. Extend it to teach `differentiate` a new primitive:

```julia
StencilCalculus.derivative(::typeof(myfun), ::Val{1}, x) = ...
```
"""
function derivative end

_pe(args) = mapreduce(eltype, promote_type, args)   # promoted element type

derivative(::typeof(+), ::Val,    args...) = One{_pe(args)}()
derivative(::typeof(-), ::Val{1}, x)       = Const(-1)
derivative(::typeof(-), ::Val{1}, x, y)    = One{_pe((x, y))}()
derivative(::typeof(-), ::Val{2}, x, y)    = Const(-1)
derivative(::typeof(*), ::Val{1}, x, y)    = y
derivative(::typeof(*), ::Val{2}, x, y)    = x
derivative(::typeof(/), ::Val{1}, x, y)    = Term(/, (One{_pe((x, y))}(), y))
derivative(::typeof(/), ::Val{2}, x, y)    = Term(-, (Term(/, (x, Term(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x, n)    = Term(*, (n, Term(^, (x, Term(-, (n, One{_pe((n,))}()))))))
derivative(::typeof(sin),  ::Val{1}, x)    = Term(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x)    = Term(-, (Term(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x)    = Term(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x)    = Term(/, (One{_pe((x,))}(), x))
derivative(::typeof(sqrt), ::Val{1}, x)    = Term(/, (One{_pe((x,))}(), Term(*, (Const(2), Term(sqrt, (x,))))))
derivative(f, ::Val, args...) = throw(ArgumentError("no derivative rule for $(f)"))

# --- Per-offset contribution collection ------------------------------------

const _Contrib = Pair{StaticShift, AbstractTerm}

_slotsym(::Slot{S}) where {S} = S

# The differentiation variable is passed as an instance (a Slot or Scalar) so
# that a Slot and a Scalar sharing a symbol do not collide: each matches only
# its own kind of the same symbol and is inert with respect to the other.
_diff(::Union{Const, Zero, One}, ::AbstractTerm) = _Contrib[]

_diff(::Slot{S2, T}, ::Slot{S}) where {S2, T, S} =
    S2 === S ? _Contrib[ô => One{T}()] : _Contrib[]
_diff(::Slot, ::Scalar) = _Contrib[]

_diff(::Scalar{S2, T}, ::Scalar{S}) where {S2, T, S} =
    S2 === S ? _Contrib[ô => One{T}()] : _Contrib[]
_diff(::Scalar, ::Slot) = _Contrib[]

function _diff(sh::Shifted, ::Slot{S}) where {S}
    sl = sh.term
    sl isa Slot || throw(ArgumentError(
        "differentiate expects normal-form input (shifts on slots); got a " *
        "Shifted over $(typeof(sl)). Call simplify first."))
    _slotsym(sl) === S ? _Contrib[sh.shift => One{eltype(sh)}()] : _Contrib[]
end
_diff(::Shifted, ::Scalar) = _Contrib[]   # a shifted slot is inert w.r.t. a scalar

function _diff(t::Term, v::AbstractTerm)
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

# --- Grouping, reverse-lex ordering, assembly ------------------------------

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
only). Returns a row-anchored `Stencil`: reverse-lex-ordered offsets and a
single `SVector`-valued coefficient term whose `k`-th entry is `∂t/∂(f[σ_k])`.
Throws if `t` does not depend on `S` (identically-zero derivative).
"""
function differentiate(t::AbstractTerm, v::Slot{S}) where {S}
    shifts, coefs = _group(_diff(simplify(t), v))
    keep = findall(c -> !(c isa Zero), coefs)
    shifts, coefs = shifts[keep], coefs[keep]
    isempty(shifts) && throw(ArgumentError(
        "the expression does not depend on Slot :$(S); its derivative is " *
        "identically zero"))
    N = maximum(_shift_maxaxis, shifts)
    key(s) = ntuple(k -> _axis_offset(s, N - k + 1), N)   # axis N most significant
    perm = sortperm(shifts; by = key)
    shifts, coefs = shifts[perm], coefs[perm]
    Stencil(RowAccess, (shifts...,), Term(SVector, (coefs...,)))
end

"""
    differentiate(t::AbstractTerm, ::Scalar{S}) -> AbstractTerm

Differentiate `t` with respect to the broadcast parameter named `S`. A scalar
carries no spatial offset, so the per-offset structure collapses: the result is
a single coefficient **term** (not a `Stencil`). Throws if `t` does not depend
on `S`.
"""
function differentiate(t::AbstractTerm, v::Scalar{S}) where {S}
    _, coefs = _group(_diff(simplify(t), v))   # all contributions sit at ô
    keep = findall(c -> !(c isa Zero), coefs)
    isempty(keep) && throw(ArgumentError(
        "the expression does not depend on Scalar :$(S); its derivative is " *
        "identically zero"))
    coefs = coefs[keep]
    length(coefs) == 1 ? coefs[1] : foldl((a, b) -> simplify(Term(+, (a, b))), coefs)
end
