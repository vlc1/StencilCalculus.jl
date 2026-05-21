# AbstractTrees interface — lets traversals / rewriters / pretty-printing
# dispatch on the generic tree API. Internal nodes expose their operator /
# shift as the node value and their operands as children; leaves are childless.

AbstractTrees.nodevalue(t::Term)    = t.fn
AbstractTrees.children(t::Term)     = t.args

AbstractTrees.nodevalue(t::Shifted) = t.shift
AbstractTrees.children(t::Shifted)  = (t.term,)

AbstractTrees.nodevalue(::Slot{S}) where {S}   = S
AbstractTrees.children(::Slot)                 = ()

AbstractTrees.nodevalue(::Scalar{S}) where {S} = S
AbstractTrees.children(::Scalar)               = ()

AbstractTrees.nodevalue(c::Const) = c.value
AbstractTrees.children(::Const)   = ()

AbstractTrees.nodevalue(::Zero{T}) where {T} = zero(T)
AbstractTrees.children(::Zero)               = ()

AbstractTrees.nodevalue(::One{T}) where {T} = one(T)
AbstractTrees.children(::One)               = ()
