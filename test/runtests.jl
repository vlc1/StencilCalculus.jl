using GridAlgebra
using Test
using AbstractTrees
using StaticArrays: SVector

@testset "GridAlgebra" begin

    @testset "leaf construction + eltype" begin
        f = Slot{:f, Float64}()
        @test f isa AbstractTerm{Float64}
        @test eltype(f) === Float64
        @test Slot{:g}() isa Slot{:g, Number}          # default T
        @test Scalar{:τ}() isa Scalar{:τ, Number}
        @test eltype(Scalar{:τ, Real}()) === Real
        @test eltype(Const(2.0)) === Float64
        @test Const(3).value === 3
        @test eltype(Zero{Float64}()) === Float64
        @test eltype(One{Int}()) === Int
    end

    @testset "operator overloading builds Terms" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        @test (f + g) isa Term{typeof(+)}
        @test eltype(f + g) === Float64
        # numeric literals wrap into Const
        t = f / 2
        @test t isa Term{typeof(/)}
        @test t.args[2] === Const(2)
        @test eltype(f / 2) === Float64
        # unary
        @test (-f) isa Term{typeof(-)}
        @test eltype(sin(f)) === Float64
        # mixed eltype promotes
        @test eltype(Slot{:a, Float64}() + Slot{:b, Int}()) === Float64
    end

    @testset "Term eltype: promote_op + Union{} throws" begin
        a = Slot{:a, Float64}(); b = Slot{:b, Float64}()
        # SVector interception → SVector-valued term
        v = SVector(a, b)
        @test v isa Term{<:Any}
        @test eltype(v) === SVector{2, Float64}
        # Genuine inhomogeneity is unconstructable.
        @test_throws ArgumentError Term(+, (Slot{:s, String}(), Slot{:n, Float64}()))
    end

    @testset "getindex shift sugar" begin
        f = Slot{:f, Number}()
        @test f[] === f                                 # zero shift = identity
        @test f[ô] === f
        sh = f[-ê₁]
        @test sh isa Shifted
        @test sh.shift === -ê₁
        @test sh.term === f
        # composition adds shifts; cancellation returns the bare slot
        @test f[-ê₁][ê₁] === f
        @test f[-ê₁][-ê₁].shift === -2ê₁
        # the DSL expression from the design docs builds
        g = f[-2ê₁] - 4f[-ê₁] + 3f[]
        @test g isa Term
    end

    @testset "non-local functors" begin
        f = Slot{:f, Float64}()
        d = δ₊{1}(f)                                    # f[i+1] - f[i]
        @test d isa Term{typeof(-)}
        @test d.args[1] isa Shifted
        @test d.args[1].shift === ê₁
        @test d.args[2] === f
        @test eltype(d) === Float64
        @test δ₊ === FwdDiff && σ₋ === BwdSum
        # scalar fall-throughs (type-call, like δ₊{1}(f))
        @test FwdDiff{1}(3.0) === 0.0
        @test FwdSum{2}(3.0) === 6.0
    end

    @testset "AbstractTrees interface" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        t = f + g
        @test AbstractTrees.nodevalue(t) === (+)
        @test AbstractTrees.children(t) === (f, g)
        @test AbstractTrees.children(f) === ()
        @test AbstractTrees.nodevalue(f) === :f
        @test AbstractTrees.nodevalue(Const(2.5)) === 2.5
        sh = f[ê₁]
        @test AbstractTrees.children(sh) === (f,)
        @test AbstractTrees.nodevalue(sh) === ê₁
    end

end
