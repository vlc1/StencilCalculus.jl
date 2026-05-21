# Hand-rolled rule rewriter. A *rule* is `(::AbstractTerm) -> Union{Nothing,
# AbstractTerm}` (`nothing` = inapplicable). `simplify` post-walks the tree
# (children first), applies the first matching rule per node, and repeats the
# whole pass to a fixed point. Equality for the fixed-point check is `===`,
# which is structural here: every node is an immutable struct bottoming out in
# egal data (Symbols/Types in parameters, Numbers in `Const`).

# --- Default rules ---------------------------------------------------------

# 1. Shift composition: Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁ + s₂, t).
rule_shift_compose(::AbstractTerm) = nothing
rule_shift_compose(t::Shifted{Sh, T, U}) where {Sh, T, U<:Shifted} =
    Shifted(t.shift + t.term.shift, t.term.term)

# 2. Shift pushdown over a Term: Shifted(s, f(a…)) → f(Shifted(s, a)…).
rule_shift_pushdown(::AbstractTerm) = nothing
rule_shift_pushdown(t::Shifted{Sh, T, U}) where {Sh, T, U<:Term} =
    Term(t.term.fn, map(a -> Shifted(t.shift, a), t.term.args))

# 3. Shift over a constant is a no-op.
rule_shift_const(::AbstractTerm) = nothing
rule_shift_const(t::Shifted{Sh, T, U}) where {Sh, T, U<:Union{Const, Zero, One}} =
    t.term

# 4. Identity / annihilator — by dispatch on the Zero/One *types* (never an
#    `iszero`/`isone` probe on a `Const`).
rule_identity(::AbstractTerm) = nothing
function rule_identity(t::Term)
    f, a = t.fn, t.args
    if f === (+) && length(a) == 2
        a[1] isa Zero && return a[2]
        a[2] isa Zero && return a[1]
    elseif f === (-) && length(a) == 2
        a[2] isa Zero && return a[1]
        a[1] isa Zero && return Term(-, (a[2],))           # 0 - b = -b
    elseif f === (*) && length(a) == 2
        (a[1] isa Zero || a[2] isa Zero) && return Zero{eltype(t)}()
        a[1] isa One && return a[2]
        a[2] isa One && return a[1]
    elseif f === (/) && length(a) == 2
        a[2] isa One  && return a[1]
        a[1] isa Zero && return Zero{eltype(t)}()
    elseif f === (-) && length(a) == 1                      # double negation
        inner = a[1]
        inner isa Term && inner.fn === (-) && length(inner.args) == 1 &&
            return inner.args[1]
    end
    return nothing
end

# 5. Constant folding over an allow-listed pure operator (produces a `Const`,
#    never a `Zero`/`One` — the pre-simplified-input assumption).
const _FOLDABLE = (+, -, *, /, \, ^, min, max)
rule_fold(::AbstractTerm) = nothing
function rule_fold(t::Term)
    (all(a -> a isa Const, t.args) && any(==(t.fn), _FOLDABLE)) || return nothing
    Const(t.fn(map(a -> a.value, t.args)...))
end

const DEFAULT_RULES = (
    rule_shift_compose,
    rule_shift_pushdown,
    rule_shift_const,
    rule_identity,
    rule_fold,
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

_rewrite(t::AbstractTerm, rules) = _apply(_rebuild(t, rules), rules)

"""
    simplify(t::AbstractTerm, rules = DEFAULT_RULES; maxsteps = 64)

Rewrite `t` to a normal form by post-walking and applying `rules` to a fixed
point. The default rules push shifts down to leaves (merging nested shifts),
apply additive/multiplicative identities on `Zero`/`One`, and fold constants.
Spurious non-zeros from un-simplified input (e.g. `Const(0)`) are **not**
recognised — the user is assumed to supply reasonably simplified expressions.
"""
function simplify(t::AbstractTerm, rules = DEFAULT_RULES; maxsteps::Int = 64)
    for _ in 1:maxsteps
        t′ = _rewrite(t, rules)
        t′ === t && return t
        t = t′
    end
    @warn "GridAlgebra.simplify hit the step budget; returning current form" maxsteps
    return t
end
