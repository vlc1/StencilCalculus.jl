# Symbolic differentiation of a (normal-form) grid expression with respect to
# a Slot (→ `StencilCore.Stencil{RowAccess}`) or a Var scalar parameter
# (→ `AbstractPointwise`). Methods on the same `differentiate` / `derivative`
# generics that StencilCore's scalar-side machinery uses — disjoint dispatch
# on `Vararg{AbstractPointwise}` vs `Vararg{AbstractScalar}`.
#
# The Slot path is **row-anchored**: the coefficient at offset `σ` is
# `∂F/∂(f[σ])` evaluated at the equation/row index — no shifts are injected
# (the Row→Column conversion lives in the assembly bridge). Neutral elements
# are the type-level `Zero`/`One`, so the chain rule collapses under
# `simplify`.

# --- @pointwise_rule macro ---------------------------------------------------

"""
    @pointwise_rule f(x) = expr
    @pointwise_rule f(x, y) = (expr1, expr2)

Concise syntax for registering a symbolic derivative rule for a primitive `f`
on the *pointwise* side, mirroring `StencilCore.@scalar_rule` (which targets
[`AbstractScalar`](@ref) arguments).

The single-argument form defines `∂f/∂x`. The two-argument form defines both
partials `(∂f/∂x, ∂f/∂y)` — the tuple maps positionally to `Val{1}` and
`Val{2}` respectively.

Inside `expr`, the argument names `x`, `y` refer to the corresponding
[`AbstractPointwise`](@ref) nodes (not numeric values). Use the pointwise
arithmetic operators and `One{eltype(x)}()` / `Zero(eltype(x))` for
structural identities, or wrap literals in `Fill(Constant(v))`.

**Examples:**

```julia
@pointwise_rule sin(x)  = Pointwise(cos, (x,))
@pointwise_rule exp(x)  = Pointwise(exp, (x,))
@pointwise_rule log(x)  = Pointwise(/, (One{eltype(x)}(), x))
@pointwise_rule *(x, y) = (y, x)
```

Each call expands to one or more `derivative(::typeof(f), ::Val{i}, ...) = ...`
methods on `Vararg{AbstractPointwise}`. Existing hand-written methods (e.g. for
`+`, `-`, `*`, `/`, `^`) are equivalent and interoperate with this macro.
"""
macro pointwise_rule(assignment)
    # `@pointwise_rule f(args...) = expr` is parsed by Julia as a single
    # assignment expression Expr(:(=), :(f(args...)), :expr).
    assignment isa Expr && assignment.head === :(=) ||
        throw(ArgumentError("@pointwise_rule: expected `f(args...) = expr`, got `$assignment`"))
    call = assignment.args[1]
    rhs  = assignment.args[2]
    # Parse `f(args...)` from the LHS call expression.
    call isa Expr && call.head === :call ||
        throw(ArgumentError("@pointwise_rule: expected a function call on the left, got `$call`"))
    f    = call.args[1]
    args = call.args[2:end]
    nargs = length(args)

    # Unwrap the LineNumberNode-padded :block Julia adds around the RHS of `=`.
    rhs = Meta.isexpr(rhs, :block) ?
        last(filter(!Base.Fix2(isa, LineNumberNode), rhs.args)) : rhs

    # RHS: either a single expr (unary) or a tuple (one element per argument).
    exprs = if rhs isa Expr && rhs.head === :tuple
        rhs.args
    else
        [rhs]
    end
    length(exprs) == nargs || throw(ArgumentError(
        "@pointwise_rule: $nargs argument(s) in `$call` but $(length(exprs)) " *
        "partial expression(s) on the right"))

    # arg declarations for the method signature: each is `name::AbstractPointwise`.
    sig = [:($(a)::AbstractPointwise) for a in args]

    # Generate one `derivative` method per partial.
    methods = map(enumerate(exprs)) do (i, expr)
        quote
            StencilCalculus.derivative(::typeof($(esc(f))), ::Val{$i}, $(esc.(sig)...)) = $(esc(expr))
        end
    end
    Expr(:block, methods...)
end

# --- Derivative table (frule-shape: ∂f/∂(arg i); term-side, Vararg{AbstractPointwise}) ---

_pe(args) = mapreduce(eltype, promote_type, args)   # promoted element type

derivative(::typeof(+), ::Val,    args::Vararg{AbstractPointwise}) = One{_pe(args)}()
derivative(::typeof(-), ::Val{1}, x::AbstractPointwise)            = Pointwise(-, (One{eltype(x)}(),))
derivative(::typeof(-), ::Val{1}, x::AbstractPointwise, y::AbstractPointwise) = One{_pe((x, y))}()
derivative(::typeof(-), ::Val{2}, x::AbstractPointwise, y::AbstractPointwise) = Pointwise(-, (One{_pe((x, y))}(),))
derivative(::typeof(*), ::Val{1}, x::AbstractPointwise, y::AbstractPointwise) = y
derivative(::typeof(*), ::Val{2}, x::AbstractPointwise, y::AbstractPointwise) = x
derivative(::typeof(/), ::Val{1}, x::AbstractPointwise, y::AbstractPointwise) = Pointwise(/, (One{_pe((x, y))}(), y))
derivative(::typeof(/), ::Val{2}, x::AbstractPointwise, y::AbstractPointwise) = Pointwise(-, (Pointwise(/, (x, Pointwise(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x::AbstractPointwise, n::AbstractPointwise) =
    Pointwise(*, (n, Pointwise(^, (x, Pointwise(-, (n, One{_pe((n,))}()))))))
derivative(::typeof(sin),  ::Val{1}, x::AbstractPointwise) = Pointwise(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x::AbstractPointwise) = Pointwise(-, (Pointwise(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x::AbstractPointwise) = Pointwise(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x::AbstractPointwise) = Pointwise(/, (One{_pe((x,))}(), x))
derivative(::typeof(sqrt), ::Val{1}, x::AbstractPointwise) =
    Pointwise(/, (One{_pe((x,))}(), Pointwise(*, (Fill(Constant(2)), Pointwise(sqrt, (x,))))))
derivative(::typeof(tan),  ::Val{1}, x::AbstractPointwise) =
    # ∂tan(x)/∂x = 1 + tan²(x); avoids introducing sec.
    Pointwise(+, (One{_pe((x,))}(),
                  Pointwise(*, (Pointwise(tan, (x,)), Pointwise(tan, (x,))))))
derivative(::typeof(abs),  ::Val{1}, x::AbstractPointwise) =
    # ∂|x|/∂x = sign(x); undefined at x = 0 (caller's responsibility).
    Pointwise(sign, (x,))
derivative(f, ::Val, args::Vararg{AbstractPointwise}) =
    throw(ArgumentError("no pointwise derivative rule for $(f)"))

# --- Per-offset contribution collection (Slot path) ------------------------

const _Contrib = Pair{StaticShift, AbstractPointwise}

_slotsym(::Slot{S}) where {S} = S

# Position-independent leaves (Fill, One) have no Slot dependence. `Zero` is
# subsumed by `Fill` (it is `Fill{<:Null}`).
_diff(::Union{Fill, One}, ::Slot) = _Contrib[]

_diff(::Slot{S2, T}, ::Slot{S}) where {S2, T, S} =
    S2 === S ? _Contrib[ô => One{T}()] : _Contrib[]

function _diff(sh::Shifted, ::Slot{S}) where {S}
    sl = sh.term
    sl isa Slot || throw(ArgumentError(
        "differentiate expects normal-form input (shifts on slots); got a " *
        "Shifted over $(typeof(sl)). Call simplify first."))
    _slotsym(sl) === S ? _Contrib[sh.shift => One{eltype(sh)}()] : _Contrib[]
end

function _diff(t::Pointwise, v::Slot)
    out = _Contrib[]
    for (i, arg) in enumerate(t.args)
        sub = _diff(arg, v)
        isempty(sub) && continue
        dfn = derivative(t.fn, Val(i), t.args...)
        for (sh, partial) in sub
            # For *, arg 1 should be left-multiplied by `partial` (non-commutative
            # correctness for SVector/SMatrix coefficient terms — mirrors Q3=(A) in
            # StencilCore/differentiate.jl).
            term = if t.fn === (*) && i == 1
                simplify(Pointwise(*, (partial, dfn)))
            else
                simplify(Pointwise(*, (dfn, partial)))
            end
            push!(out, sh => term)
        end
    end
    return out
end

# --- Differentiation w.r.t. a Symbolic (scalar parameter) ------------------
# A Symbolic appears in a term only inside a `Fill`, so the derivative
# collapses to a per-cell broadcast coefficient. Walking strategy: like the
# Slot path, but without spatial offsets — accumulate one AbstractPointwise.

_diff_scalar(::Union{Slot, Shifted, One}, ::Var) = nothing
function _diff_scalar(f::Fill{<:AbstractScalar}, v::Var)
    d = differentiate(f.val, v)
    d isa Null ? nothing : Fill(d)
end
_diff_scalar(::Fill, ::Var) = nothing                      # literal Fill: no dependence

function _diff_scalar(t::Pointwise, v::Var)
    out = nothing
    for (i, arg) in enumerate(t.args)
        sub = _diff_scalar(arg, v)
        sub === nothing && continue
        dfn     = derivative(t.fn, Val(i), t.args...)
        contrib = if t.fn === (*) && i == 1
            simplify(Pointwise(*, (sub, dfn)))
        else
            simplify(Pointwise(*, (dfn, sub)))
        end
        out = (out === nothing) ? contrib : simplify(Pointwise(+, (out, contrib)))
    end
    return out
end

# --- Grouping, reverse-lex ordering, assembly (Slot path) ------------------

# Sum coefficients sharing an offset; preserve first-seen order.
function _group(contribs::Vector{_Contrib})
    shifts = StaticShift[]
    coefs  = AbstractPointwise[]
    for (sh, c) in contribs
        idx = findfirst(==(sh), shifts)
        if idx === nothing
            push!(shifts, sh); push!(coefs, c)
        else
            coefs[idx] = simplify(Pointwise(+, (coefs[idx], c)))
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
    differentiate(t::AbstractPointwise, ::Slot{S}) -> StencilCore.Stencil{RowAccess}

Differentiate `t` with respect to the field named `S` (matched on the symbol
only). Returns a row-anchored `Stencil` in structure-of-arrays form: reverse-
lex-ordered offsets `shifts` and a parallel tuple `terms` whose `k`-th entry is
the coefficient `∂t/∂(f[σ_k])`. Throws if `t` does not depend on `S`
(identically-zero derivative).
"""
function differentiate(t::AbstractPointwise, v::Slot{S}) where {S}
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
    differentiate(t::AbstractPointwise, ::Var{S}) -> AbstractPointwise

Differentiate `t` with respect to the named scalar parameter `S`. A `Var`
appears in `t` only inside a [`Fill`](@ref), so the derivative is a per-cell
broadcast coefficient (a single `AbstractPointwise`, not a `Stencil`). Throws if
`t` does not depend on `S`.
"""
function differentiate(t::AbstractPointwise, v::Var{S}) where {S}
    out = _diff_scalar(simplify(t), v)
    out === nothing && throw(ArgumentError(
        "the expression does not depend on Var :$(S); its derivative " *
        "is identically zero"))
    simplify(out)
end
