# Hand-rolled rule rewriter for `AbstractTerm`. A *rule* is
# `(::AbstractTerm) -> Union{Nothing, AbstractTerm}` (`nothing` =
# inapplicable). `simplify` post-walks the tree (children first), applies the
# first matching rule per node, and repeats the whole pass to a fixed point.
# Equality for the fixed-point check is `===`, which is structural here: every
# node is an immutable struct bottoming out in egal data.
#
# Method on the same generic as `StencilCore.simplify(::AbstractScalar, …)` —
# disjoint dispatch keeps the two algebras separate.

# --- Identity predicates ----------------------------------------------------
# Type-dispatched on the structural identities (`Zero`/`One`/`Null`/`Unity`)
# and value-dispatched on a literal `Fill(Const(v))`. A `Fill` wrapping a
# symbolic scalar (e.g. `Fill(Symbolic{S})`, `Fill(Scalar(…))`) is *never*
# treated as an identity (the symbolic value may take any runtime value).

_is_term_zero(::Zero)           = true
_is_term_zero(::Fill{<:Null})   = true
_is_term_zero(f::Fill{<:Const}) = iszero(f.val.val)
_is_term_zero(::AbstractTerm)   = false

_is_term_one(::One)             = true
_is_term_one(::Fill{<:Unity})   = true
_is_term_one(f::Fill{<:Const})  = isone(f.val.val)
_is_term_one(::AbstractTerm)    = false

# --- Default rules ---------------------------------------------------------

# 1. Shift composition: Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁ + s₂, t).
rule_shift_compose(::AbstractTerm) = nothing
rule_shift_compose(t::Shifted{Sh, T, U}) where {Sh, T, U<:Shifted} =
    Shifted(t.shift + t.term.shift, t.term.term)

# 2. Shift pushdown over a Term: Shifted(s, f(a…)) → f(Shifted(s, a)…).
rule_shift_pushdown(::AbstractTerm) = nothing
rule_shift_pushdown(t::Shifted{Sh, T, U}) where {Sh, T, U<:Term} =
    Term(t.term.fn, map(a -> Shifted(t.shift, a), t.term.args))

# 3. Shift over a position-independent leaf (Fill / Zero / One) is a no-op.
rule_shift_const(::AbstractTerm) = nothing
rule_shift_const(t::Shifted{Sh, T, U}) where {Sh, T, U<:Union{Fill, Zero, One}} =
    t.term

# 4. Identity / annihilator.
rule_identity(::AbstractTerm) = nothing
function rule_identity(t::Term)
    f, a = t.fn, t.args
    if f === (+) && length(a) == 2
        _is_term_zero(a[1]) && return a[2]
        _is_term_zero(a[2]) && return a[1]
    elseif f === (-) && length(a) == 2
        _is_term_zero(a[2]) && return a[1]
        _is_term_zero(a[1]) && return Term(-, (a[2],))      # 0 - b = -b
    elseif f === (*) && length(a) == 2
        (_is_term_zero(a[1]) || _is_term_zero(a[2])) && return Zero{eltype(t)}()
        _is_term_one(a[1]) && return a[2]
        _is_term_one(a[2]) && return a[1]
    elseif f === (/) && length(a) == 2
        _is_term_one(a[2])  && return a[1]
        _is_term_zero(a[1]) && return Zero{eltype(t)}()
    elseif f === (-) && length(a) == 1                       # double negation
        inner = a[1]
        inner isa Term && inner.fn === (-) && length(inner.args) == 1 &&
            return inner.args[1]
    end
    return nothing
end

# 5. Scalar precedence: a `Term` whose args are *all* `Fill` is the broadcast
#    of a scalar-tree expression. Collapse it into one `Fill(Scalar(fn, vals…))`,
#    simplified in scalar-land. Operationally this is the dual of `asterm`:
#    Fills cluster, scalars regroup. The simplified inner scalar may reduce
#    to a `Const`, `Null`, etc.
rule_fill_collapse(::AbstractTerm) = nothing
function rule_fill_collapse(t::Term)
    all(a -> a isa Fill, t.args) || return nothing
    Fill(simplify(Scalar(t.fn, map(a -> a.val, t.args))))
end

const DEFAULT_RULES = (
    rule_shift_compose,
    rule_shift_pushdown,
    rule_shift_const,
    rule_identity,
    rule_fill_collapse,
)

# --- Rewriter --------------------------------------------------------------

# Apply the first matching rule to `t` (else return `t` unchanged).
function _apply(t::AbstractTerm, rules)
    for r in rules
        u = r(t)
        u === nothing || return u
    end
    return t
end

# Rebuild a node with simplified children, reusing the node when unchanged
# (so `===` detects quiescence).
_rebuild(t::AbstractTerm, rules) = t                        # leaves
function _rebuild(t::Term, rules)
    newargs = map(a -> _rewrite(a, rules), t.args)
    newargs === t.args ? t : Term(t.fn, newargs)
end
function _rebuild(t::Shifted, rules)
    newterm = _rewrite(t.term, rules)
    newterm === t.term ? t : Shifted(t.shift, newterm)
end
# A `Fill{<:AbstractScalar}` delegates to scalar-side `simplify` so its inner
# scalar reaches scalar-side normal form too.
function _rebuild(t::Fill{<:AbstractScalar}, rules)
    newval = simplify(t.val)
    newval === t.val ? t : Fill(newval)
end

_rewrite(t::AbstractTerm, rules) = _apply(_rebuild(t, rules), rules)

"""
    simplify(t::AbstractTerm, rules = DEFAULT_RULES; maxsteps = 64)

Rewrite `t` to a normal form by post-walking and applying `rules` to a fixed
point. The default rules push shifts down to leaves (merging nested shifts),
apply additive/multiplicative identities, and collapse all-Fill `Term`s into
a single `Fill(Scalar(…))` (the scalar-precedence rule). Method on the same
generic as [`StencilCore.simplify`](@ref) for `AbstractScalar`.
"""
function simplify(t::AbstractTerm, rules = DEFAULT_RULES; maxsteps::Int = 64)
    for _ in 1:maxsteps
        t′ = _rewrite(t, rules)
        t′ === t && return t
        t = t′
    end
    @warn "StencilCalculus.simplify hit the step budget; returning current form" maxsteps
    return t
end
