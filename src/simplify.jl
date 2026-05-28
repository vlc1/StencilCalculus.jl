# Hand-rolled rule rewriter for `AbstractPointwise`. A *rule* is
# `(::AbstractPointwise) -> Union{Nothing, AbstractPointwise}` (`nothing` =
# inapplicable). `simplify` post-walks the tree (children first), applies the
# first matching rule per node, and repeats the whole pass to a fixed point.
# Equality for the fixed-point check is `===`, which is structural here: every
# node is an immutable struct bottoming out in egal data.
#
# Method on the same generic as `StencilCore.simplify(::AbstractScalar, …)` —
# disjoint dispatch keeps the two algebras separate.

# --- Identity predicates ----------------------------------------------------
# Type-dispatched on the structural identities (`Zero`/`IdentityStencil`/`Null`/`Unity`)
# and value-dispatched on a literal `Fill(Constant(v))`. A `Fill` wrapping a
# symbolic scalar (e.g. `Fill(Var{S})`, `Fill(Scalar(…))`) is *never*
# treated as an identity (the symbolic value may take any runtime value).

_is_term_zero(::Fill{<:Null})        = true
_is_term_zero(f::Fill{<:Constant})   = iszero(f.val.val)
_is_term_zero(::AbstractPointwise)        = false

_is_term_one(::IdentityStencil)      = true
_is_term_one(::Fill{<:Unity})        = true
_is_term_one(f::Fill{<:Constant})    = isone(f.val.val)
_is_term_one(::AbstractPointwise)    = false

# --- Default rules ---------------------------------------------------------

# 1. Shift composition: Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁ + s₂, t).
rule_shift_compose(::AbstractPointwise) = nothing
rule_shift_compose(t::Shifted{Sh, T, U}) where {Sh, T, U<:Shifted} =
    Shifted(t.shift + t.term.shift, t.term.term)

# 2. Shift pushdown over a Pointwise: Shifted(s, f(a…)) → f(Shifted(s, a)…).
rule_shift_pushdown(::AbstractPointwise) = nothing
rule_shift_pushdown(t::Shifted{Sh, T, U}) where {Sh, T, U<:Pointwise} =
    Pointwise(t.term.fn, map(a -> Shifted(t.shift, a), t.term.args))

# 3. Shift over a position-independent leaf (Fill / Zero / IdentityStencil) is a no-op.
rule_shift_const(::AbstractPointwise) = nothing
rule_shift_const(t::Shifted{Sh, T, U}) where {Sh, T, U<:Union{Fill, IdentityStencil}} =
    t.term

# 4. Identity / annihilator.
rule_identity(::AbstractPointwise) = nothing
function rule_identity(t::Pointwise)
    f, a = t.fn, t.args
    if f === (+) && length(a) == 2
        _is_term_zero(a[1]) && return a[2]
        _is_term_zero(a[2]) && return a[1]
    elseif f === (-) && length(a) == 2
        _is_term_zero(a[2]) && return a[1]
        _is_term_zero(a[1]) && return Pointwise(-, (a[2],))      # 0 - b = -b
    elseif f === (*) && length(a) == 2
        (_is_term_zero(a[1]) || _is_term_zero(a[2])) && return Zero(eltype(t))
        _is_term_one(a[1]) && return a[2]
        _is_term_one(a[2]) && return a[1]
    elseif f === (/) && length(a) == 2
        _is_term_one(a[2])  && return a[1]
        _is_term_zero(a[1]) && return Zero(eltype(t))
    end
    return nothing
end

# 5. Double negation: -(-x) → x.
rule_double_negation(::AbstractPointwise) = nothing
function rule_double_negation(t::Pointwise)
    t.fn === (-) && length(t.args) == 1 || return nothing
    inner = t.args[1]
    inner isa Pointwise && inner.fn === (-) && length(inner.args) == 1 &&
        return inner.args[1]
    return nothing
end

# 6. Scalar precedence: a `Pointwise` whose args are *all* `Fill` is the broadcast
#    of a scalar-tree expression. Collapse it into one `Fill(Scalar(fn, vals…))`,
#    simplified in scalar-land. Operationally this is the dual of `asterm`:
#    Fills cluster, scalars regroup. The simplified inner scalar may reduce
#    to a `Const`, `Null`, etc.
rule_fill_collapse(::AbstractPointwise) = nothing
function rule_fill_collapse(t::Pointwise)
    all(a -> a isa Fill, t.args) || return nothing
    Fill(simplify(Scalar(t.fn, map(a -> a.val, t.args))))
end

"""
    POINTWISE_DEFAULT_RULES

Default simplification rule set used by [`simplify`](@ref) on
`AbstractPointwise` terms. Each element is a callable with signature:

    rule(t::AbstractPointwise) -> Union{AbstractPointwise, Nothing}

A rule returns a replacement node when applicable, or `nothing` to pass to
the next rule. Rules are tried **in order**; the first non-`nothing` result
wins. The post-walk in `_rebuild` guarantees that children are already in
normal form when a rule fires on their parent.

Built-in rules (in application order):

| Rule                    | Action                                                    |
|:------------------------|:----------------------------------------------------------|
| `rule_shift_compose`    | `Shifted(s₁, Shifted(s₂, t)) → Shifted(s₁+s₂, t)`       |
| `rule_shift_pushdown`   | `Shifted(s, f(a…)) → f(Shifted(s,a)…)`                   |
| `rule_shift_const`      | `Shifted(s, Fill/IdentityStencil) → Fill/IdentityStencil` (Zero is a `Fill{<:Null}`) |
| `rule_identity`         | `0+x→x`, `x+0→x`, `0*x→0`, `1*x→x`, `x/1→x`, `0/x→0`   |
| `rule_double_negation`  | `-(-x) → x`                                              |
| `rule_fill_collapse`    | All-`Fill` `Pointwise` → `Fill(Scalar(fn,…))` (scalar-land) |

To extend, compose a new tuple that includes the new rule(s) and pass it as
the `rules` keyword argument to `simplify`:

```julia
my_rules = (my_rule, StencilCalculus.POINTWISE_DEFAULT_RULES...)
simplify(expr; rules = my_rules)
```
"""
const POINTWISE_DEFAULT_RULES = (
    rule_shift_compose,
    rule_shift_pushdown,
    rule_shift_const,
    rule_identity,
    rule_double_negation,
    rule_fill_collapse,
)

# --- Rewriter --------------------------------------------------------------

# Apply the first matching rule to `t` (else return `t` unchanged).
function _apply(t::AbstractPointwise, rules)
    for r in rules
        u = r(t)
        u === nothing || return u
    end
    return t
end

# Rebuild a node with simplified children, reusing the node when unchanged
# (so `===` detects quiescence).
_rebuild(t::AbstractPointwise, rules) = t                        # leaves
function _rebuild(t::Pointwise, rules)
    newargs = map(a -> _rewrite(a, rules), t.args)
    newargs === t.args ? t : Pointwise(t.fn, newargs)
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

_rewrite(t::AbstractPointwise, rules) = _apply(_rebuild(t, rules), rules)

"""
    simplify(t::AbstractPointwise, rules = POINTWISE_DEFAULT_RULES; maxsteps = 64)

Rewrite `t` to a normal form by post-walking and applying `rules` to a fixed
point. The default rules push shifts down to leaves (merging nested shifts),
apply additive/multiplicative identities, collapse double negations, and
collapse all-Fill `Pointwise`s into a single `Fill(Scalar(…))` (the scalar-
precedence rule). Method on the same generic as [`StencilCore.simplify`](@ref)
for `AbstractScalar`.
"""
function simplify(t::AbstractPointwise, rules = POINTWISE_DEFAULT_RULES; maxsteps::Int = 64)
    for _ in 1:maxsteps
        t′ = _rewrite(t, rules)
        t′ === t && return t
        t = t′
    end
    @warn "StencilCalculus.simplify hit the step budget; returning current form" maxsteps
    return t
end
