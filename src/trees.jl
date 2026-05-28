# AbstractTrees interface for AbstractPointwise — lets traversals / rewriters /
# pretty-printing dispatch on the generic tree API. Internal nodes expose
# their operator / shift as the node value and their operands as children;
# leaves are childless. The scalar-side analogue lives in
# StencilCore/src/scalar_trees.jl.

AbstractTrees.nodevalue(t::Pointwise)    = t.fn
AbstractTrees.children(t::Pointwise)     = t.args

AbstractTrees.nodevalue(t::Shifted) = t.shift
AbstractTrees.children(t::Shifted)  = (t.term,)

AbstractTrees.nodevalue(::Slot{S}) where {S} = S
AbstractTrees.children(::Slot)               = ()

AbstractTrees.nodevalue(::IdentityStencil{T, U}) where {T, U} = one(U)
AbstractTrees.children(::IdentityStencil)                     = ()

AbstractTrees.nodevalue(::DiagonalStencil) = :diag
AbstractTrees.children(d::DiagonalStencil) = (d.term,)

# `Fill` covers both literals and the scalar-side identities (`Null`/`Unity`),
# the latter of which is what the [`Zero`](@ref) alias wraps. The wrapped value
# is the node value; the Fill itself is a leaf.
AbstractTrees.nodevalue(f::Fill) = f.val
AbstractTrees.children(::Fill)   = ()
