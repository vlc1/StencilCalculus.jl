module GridAlgebra

using AbstractTrees
import StaticArrays: SVector       # imported (not just `using`) so we can add a method
using StencilCore: AbstractTerm, StaticShift, SShift, StaticPair, SPair,
                   ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

include("terms.jl")       # Slot, Scalar, Const, Zero, One, Term, Shifted
include("operators.jl")   # operator overloads, SVector interception, getindex sugar, DSL functors
include("trees.jl")       # AbstractTrees interface

# Concrete term types.
export AbstractTerm, Slot, Scalar, Const, Zero, One, Term, Shifted

# Non-local DSL functors (+ Unicode aliases).
export FwdDiff, BwdDiff, FwdSum, BwdSum, δ₊, δ₋, σ₊, σ₋

# Re-exported offset vocabulary (DSL sugar) from StencilCore.
export StaticShift, SShift, StaticPair, SPair
export ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

end # module GridAlgebra
