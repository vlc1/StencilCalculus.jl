module StencilCalculus

using AbstractTrees
import StaticArrays: SVector, StaticArray  # imported (not just `using`) so we can add methods
using RuntimeGeneratedFunctions
using StencilCore: AbstractPointwise, AbstractScalar,
                   Var, Constant, Null, Unity, Scalar, _assert_concrete,
                   StaticShift, SShift, StaticPair, SPair,
                   Stencil, RowAccess, ColumnAccess, dim, offset,
                   ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉,
                   var"@var"
import StencilCore: simplify, materialize, differentiate, derivative,
                    _scalar_body_expr, _unity_space, _is_eltype_wildcard

RuntimeGeneratedFunctions.init(@__MODULE__)

include("pointwise.jl")    # Slot, Zero, One, Pointwise, Shifted, Fill
include("operators.jl")    # operator overloads, SVector interception, getindex sugar, DSL functors
include("trees.jl")        # AbstractTrees interface
include("simplify.jl")     # rule rewriter for AbstractPointwise
include("differentiate.jl")# symbolic differentiation → Stencil{RowAccess} / AbstractPointwise
include("materialize.jl")  # codegen → LazyArray; code_string
include("bridge.jl")       # Stencil{RowAccess} → assemblable Linear/Star{ColumnAccess}
include("apply.jl")        # `*(stencil, pointwise)` shells (bodies TBD)
include("show.jl")         # component-form display of normal-form terms

# Concrete pointwise types + their constructor macros.
export AbstractPointwise, Slot, Zero, One, Pointwise, Shifted, Fill, @slot

# Re-export the scalar algebra from StencilCore so users get the full CAS
# from a single `using StencilCalculus`.
export AbstractScalar, Var, Constant, Null, Unity, Scalar
export var"@var"

# Non-local DSL functors (+ Unicode aliases).
export FwdDiff, BwdDiff, FwdSum, BwdSum, δ₊, δ₋, σ₊, σ₋

# Rewriting + differentiation (shared generic with StencilCore — same function,
# disjoint methods on AbstractPointwise vs AbstractScalar).
export simplify, differentiate, derivative, Diff, ∂
export POINTWISE_DEFAULT_RULES, @pointwise_rule

# Materialization / codegen + the assembly bridge.
export materialize, code_string, LazyArray, build_stencil, densify

# Re-exported offset vocabulary (DSL sugar) from StencilCore.
export StaticShift, SShift, StaticPair, SPair
export ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

end # module StencilCalculus
