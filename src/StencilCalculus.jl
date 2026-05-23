module StencilCalculus

using AbstractTrees
import StaticArrays: SVector       # imported (not just `using`) so we can add a method
using RuntimeGeneratedFunctions
using StencilCore: AbstractTerm, AbstractScalar,
                   Symbolic, Const, Null, Unity, Scalar, _assert_concrete,
                   StaticShift, SShift, StaticPair, SPair,
                   Stencil, RowAccess, ColumnAccess, dim, offset,
                   ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉,
                   @symbolic, var"@const"
import StencilCore: simplify, materialize, differentiate, derivative,
                    _scalar_body_expr

RuntimeGeneratedFunctions.init(@__MODULE__)

include("terms.jl")        # Slot, Zero, One, Term, Shifted, Fill
include("operators.jl")    # operator overloads, SVector interception, getindex sugar, DSL functors
include("trees.jl")        # AbstractTrees interface
include("simplify.jl")     # rule rewriter for AbstractTerm
include("differentiate.jl")# symbolic differentiation → Stencil{RowAccess} / AbstractTerm
include("materialize.jl")  # codegen → LazyArray; code_string
include("bridge.jl")       # Stencil{RowAccess} → assemblable Linear/Star{ColumnAccess}
include("show.jl")         # component-form display of normal-form terms

# Concrete term types + their constructor macros.
export AbstractTerm, Slot, Zero, One, Term, Shifted, Fill, @slot

# Re-export the scalar algebra from StencilCore so users get the full CAS
# from a single `using StencilCalculus`.
export AbstractScalar, Symbolic, Const, Null, Unity, Scalar
export @symbolic, var"@const"

# Non-local DSL functors (+ Unicode aliases).
export FwdDiff, BwdDiff, FwdSum, BwdSum, δ₊, δ₋, σ₊, σ₋

# Rewriting + differentiation (shared generic with StencilCore — same function,
# disjoint methods on AbstractTerm vs AbstractScalar).
export simplify, differentiate, derivative, Diff, ∂

# Materialization / codegen + the assembly bridge.
export materialize, code_string, LazyArray, build_stencil, densify

# Re-exported offset vocabulary (DSL sugar) from StencilCore.
export StaticShift, SShift, StaticPair, SPair
export ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

end # module StencilCalculus
