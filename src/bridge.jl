# Bridge: turn a differentiation result (`Stencil{RowAccess}`, symbolic
# coefficient) into an assemblable concrete-coefficient
# `LinearStencil`/`StarStencil{ColumnAccess}` — closing the
# differentiate → assemble loop. The CSC `assemble`/`update!`/`build` then
# live in StencilAssembly.

using StencilCore: as_linear, as_star, AccessStyle, LinearStencil, StarStencil
import StencilCore: _interlace

# The symbolic-term half of StencilCore's SoA→AoS coefficient combiner: stack
# the per-offset coefficient terms into one SVector-valued term.
_interlace(terms::NTuple{M, AbstractPointwise}) where {M} = Pointwise(SVector, terms)

# RowAccess → ColumnAccess. Offsets are invariant; each per-offset coefficient
# `g_σ` (value at the row) becomes `Shifted(-σ, g_σ)` (the value re-read at the
# column). Constant coefficients are unchanged (shift over a constant is a
# no-op). A no-op when already ColumnAccess. Uses the explicit-T `Stencil{T}`
# ctor so all-wildcard inputs (e.g. `differentiate(f, f)`) round-trip cleanly.
function _to_column(sst::Stencil)
    AccessStyle(sst) === ColumnAccess() && return sst
    shifted = map((s, g) -> simplify(Shifted(-s, g)), sst.shifts, sst.terms)
    Stencil(eltype(sst), ColumnAccess, sst.shifts, shifted)
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
    shifts, gs = sst.shifts, sst.terms
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
    T = mapreduce(eltype, promote_type, gs)                # scalar coefficient type
    newshifts = StaticShift[]
    newterms  = AbstractPointwise[]
    for o in lo:hi
        push!(newshifts, SShift((SPair{D, o}(),)))         # SPair{D,0} normalizes to ô
        i = findfirst(==(o), offs)
        push!(newterms, i === nothing ? Zero(T) : gs[i])
    end
    Stencil(eltype(sst), typeof(AccessStyle(sst)), (newshifts...,), (newterms...,))
end

# Narrow to LinearStencil (single-axis contiguous) or StarStencil (canonical
# star); rethrow if neither. Interlaces the SoA `terms` into the AoS coefficient.
function _narrow(st::Stencil)
    try
        return as_linear(st)
    catch e
        e isa ArgumentError || rethrow()
        return as_star(st)
    end
end

# Materialize a narrowed stencil's (still symbolic) SVector-valued coefficient
# into a concrete array, rebuilding the same Linear/Star stencil.
_materialize_coef(ln::LinearStencil{D, O, L, E, A, S}, pairs; size) where {D, O, L, E, A, S} =
    LinearStencil{D}(S, ln.offsets, materialize(ln.term, pairs; size = size))
_materialize_coef(ss::StarStencil{L, N, M, E, A, S}, pairs; size) where {L, N, M, E, A, S} =
    StarStencil{L}(S, materialize(ss.term, pairs; size = size))

"""
    build_stencil(sst::Stencil, pairs::NamedTuple = (;); size = nothing)

Lower a (typically `differentiate`-produced) `Stencil` to an assemblable
`LinearStencil` / `StarStencil{ColumnAccess}` with a concrete coefficient:

1. convert `RowAccess → ColumnAccess` (shift each per-offset coefficient by
   `−offset`);
2. optionally [`densify`](@ref) (`pad = true`) to fill single-axis offset gaps
   with `Zero` so a gappy result still narrows to `LinearStencil`;
3. narrow the offset pattern (`as_linear` / `as_star`), which switches the
   structure-of-arrays `terms` to the array-of-structs `SVector` coefficient;
4. `materialize` that coefficient against `pairs` (variable coefficients) — for
   a constant coefficient pass the mesh `size`.

The result is ready for `StencilAssembly.build` / `assemble`.
"""
function build_stencil(sst::Stencil, pairs::NamedTuple = (;); size = nothing, pad::Bool = false)
    cst = _to_column(sst)
    pad && (cst = densify(cst))
    _materialize_coef(_narrow(cst), pairs; size = size)
end
