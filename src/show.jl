# Pretty-printing of grid expressions in component form. `show` always renders
# the *normal form* (`simplify`'d) of a term; it never mutates the term itself.
# Leaf renderings: `Slot` → `f[]`, shifted `Slot` → `f[ê₁]`, `Scalar` → its
# symbol, `Const` → its value, `Zero`/`One` → `0`/`1` (the symbolic identities —
# *not* `Const(zero(T))`, which would fail for an abstract `T` like `Number`).

const _INFIX = (:+, :-, :*, :/, :\, :^)

Base.show(io::IO, t::AbstractTerm) = _show(io, simplify(t))

_show(io::IO, ::Slot{S})   where {S} = print(io, S, "[]")
_show(io::IO, ::Scalar{S}) where {S} = print(io, S)
_show(io::IO, c::Const)              = show(io, c.value)
_show(io::IO, ::Zero)                = print(io, '0')
_show(io::IO, ::One)                 = print(io, '1')

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
