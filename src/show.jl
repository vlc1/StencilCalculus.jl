# Pretty-printing of grid expressions in component form. `show` always renders
# the *normal form* (`simplify`'d) of a term; it never mutates the term itself.
# Leaf renderings: `Slot` → `f[]`, shifted `Slot` → `f[ê₁]`, `Zero`/`One` →
# `0`/`1` glyphs (type-agnostic, parallel to the scalar-side `Null`/`Unity`),
# `Fill` → its wrapped value via the scalar-side show (e.g. `Fill(Const(2.0))`
# prints as `2.0`, `Fill(τ)` prints as `τ`).

const _INFIX = (:+, :-, :*, :/, :\, :^)

Base.show(io::IO, t::AbstractTerm) = _show(io, simplify(t))

_show(io::IO, ::Slot{S}) where {S} = print(io, S, "[]")
_show(io::IO, ::Zero)              = print(io, '0')
_show(io::IO, ::One)               = print(io, '1')
# A Fill is rendered as its wrapped value: AbstractScalar uses Core's scalar
# show; a literal uses Base.show directly. Either way, no `[]` (Fill has no
# spatial index).
_show(io::IO, f::Fill{T}) where {T<:AbstractScalar} = show(io, f.val)
_show(io::IO, f::Fill)                              = show(io, f.val)

function _show(io::IO, t::Shifted)
    t.term isa Slot || error("display expects normal-form input (shifts on slots)")
    print(io, _slotsym(t.term), '[')
    show(io, t.shift)
    print(io, ']')
end

function _show(io::IO, t::Term)
    op, args = _callsym(t.fn), t.args
    if length(args) == 2 && op in _INFIX
        print(io, '(')
        _show(io, args[1])
        print(io, ' ', op, ' ')
        _show(io, args[2])
        print(io, ')')
    elseif length(args) == 1 && op === :-
        print(io, '-')
        _show(io, args[1])
    else
        print(io, op, '(')
        for (i, a) in enumerate(args)
            i == 1 || print(io, ", ")
            _show(io, a)
        end
        print(io, ')')
    end
end

# The "with respect to" functor displays as a call of its alias: `∂(f[])` /
# `∂(τ)` depending on whether its target is a Slot or a Symbolic.
function Base.show(io::IO, d::Diff)
    print(io, "∂(")
    show(io, d.term)
    print(io, ')')
end
