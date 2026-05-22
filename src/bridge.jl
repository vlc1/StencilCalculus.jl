# Bridge: turn a differentiation result (`Stencil{RowAccess}`, symbolic
# coefficient) into an assemblable concrete-coefficient
# `LinearStencil`/`StarStencil{ColumnAccess}` — closing the
# differentiate → assemble loop. The CSC `assemble`/`update!`/`build` then
# live in StencilAssembly.

using StencilCore: as_linear, as_star, AccessStyle

# RowAccess → ColumnAccess. Offsets are invariant; each coefficient `g_σ`
# (value at the row) becomes `Shifted(-σ, g_σ)` (the value re-read at the
# column). Constant coefficients are unchanged (shift over a constant is a
# no-op). A no-op when already ColumnAccess.
function _to_column(sst::Stencil)
    AccessStyle(sst) === ColumnAccess() && return sst
    t = sst.term
    (t isa Term && t.fn === SVector) || throw(ArgumentError(
        "expected a combined SVector-valued coefficient term, got $(typeof(t))"))
    shifts = sst.shifts
    gs = t.args
    shifted = ntuple(k -> simplify(Shifted(-shifts[k], gs[k])), length(gs))
    Stencil(ColumnAccess, shifts, Term(SVector, shifted))
end

_npairs(s::StaticShift) = length(s.pairs)

"""
    densify(sst::Stencil) -> Stencil

Pad a single-axis `Stencil` to a *contiguous* offset range by inserting
type-level `Zero` coefficients at the missing offsets, so a gappy
differentiation result (e.g. offsets `{-2, 0}`) can narrow to `LinearStencil`.
A no-op (returns `sst`) for patterns that are already contiguous, multi-axis,
or pure-diagonal (ambiguous axis) — those go to `as_star` or stay as is.
"""
function densify(sst::Stencil)
    term = sst.term
    (term isa Term && term.fn === SVector) || return sst
    shifts, gs = sst.shifts, term.args
    all(s -> _npairs(s) <= 1, shifts) || return sst        # not single-axis
    D = 0
    for s in shifts
        a = _shift_maxaxis(s)
        a == 0 && continue
        D == 0 ? (D = a) : (D == a || return sst)          # multi-axis ⇒ bail
    end
    D == 0 && return sst                                   # only the diagonal ⇒ ambiguous
    offs = [_axis_offset(s, D) for s in shifts]
    lo, hi = minimum(offs), maximum(offs)
    (hi - lo + 1) == length(offs) && return sst            # already contiguous
    T = eltype(eltype(term))                               # scalar coefficient type
    newshifts = StaticShift[]
    newcoefs  = AbstractTerm[]
    for o in lo:hi
        push!(newshifts, SShift((SPair{D, o}(),)))         # SPair{D,0} normalizes to ô
        i = findfirst(==(o), offs)
        push!(newcoefs, i === nothing ? Zero{T}() : gs[i])
    end
    Stencil(typeof(AccessStyle(sst)), (newshifts...,), Term(SVector, (newcoefs...,)))
end

# Narrow to LinearStencil (single-axis contiguous) or StarStencil (canonical
# star); rethrow if neither.
function _narrow(st::Stencil)
    try
        return as_linear(st)
    catch e
        e isa ArgumentError || rethrow()
        return as_star(st)
    end
end

"""
    build_stencil(sst::Stencil, pairs::NamedTuple = (;); size = nothing)

Lower a (typically `differentiate`-produced) `Stencil` to an assemblable
`LinearStencil` / `StarStencil{ColumnAccess}` with a concrete coefficient:

1. convert `RowAccess → ColumnAccess` (shift each coefficient by `−offset`);
2. optionally [`densify`](@ref) (`pad = true`) to fill single-axis offset gaps
   with `Zero` so a gappy result still narrows to `LinearStencil`;
3. `materialize` the combined coefficient term against `pairs` (variable
   coefficients) — for a constant coefficient pass the mesh `size`;
4. narrow the offset pattern (`as_linear` / `as_star`), reusing the
   materialized coefficient verbatim.

The result is ready for `StencilAssembly.build` / `assemble`.
"""
function build_stencil(sst::Stencil, pairs::NamedTuple = (;); size = nothing, pad::Bool = false)
    cst = _to_column(sst)
    pad && (cst = densify(cst))
    coef = materialize(cst.term, pairs; size = size)
    _narrow(Stencil(ColumnAccess, cst.shifts, coef))
end
