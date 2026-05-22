# Bridge: turn a differentiation result (`Stencil{RowAccess}`, symbolic
# coefficient) into an assemblable concrete-coefficient
# `LinearStencil`/`StarStencil{ColumnAccess}` — closing the
# differentiate → assemble loop. The CSC `assemble`/`update!`/`build` then
# live in CartesianOperators.

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
2. `materialize` the combined coefficient term against `pairs` (variable
   coefficients) — for a constant coefficient pass the mesh `size`;
3. narrow the offset pattern (`as_linear` / `as_star`), reusing the
   materialized coefficient verbatim.

The result is ready for `CartesianOperators.build` / `assemble`.
"""
function build_stencil(sst::Stencil, pairs::NamedTuple = (;); size = nothing)
    cst = _to_column(sst)
    coef = materialize(cst.term, pairs; size = size)
    _narrow(Stencil(ColumnAccess, cst.shifts, coef))
end
