using GridAlgebra
using Test
using AbstractTrees
using StaticArrays: SVector
using StencilCore: Stencil, LinearStencil, StarStencil, AccessStyle, RowAccess,
                   as_linear, as_star

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

    @testset "simplify" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        Z = Zero{Float64}(); I1 = One{Float64}()

        # No-op on a normal-form expression / a leaf.
        @test simplify(f) === f
        @test simplify(f + g) == f + g

        # Shift composition (built manually, bypassing getindex's eager merge).
        nested = Shifted(ê₁, Shifted(ê₁, f))
        @test simplify(nested) === f[2ê₁]
        @test simplify(Shifted(ê₁, Shifted(-ê₁, f))) === f      # cancels to identity

        # Shift pushdown to leaves.
        d = δ₊{1}(f + g)                                         # Shifted(ê₁, f+g) - (f+g)
        s = simplify(d)
        @test s == (f[ê₁] + g[ê₁]) - (f + g)

        # Shift over a constant is a no-op.
        @test simplify(Shifted(ê₁, Const(2.0))) === Const(2.0)

        # Identity / annihilator on Zero/One.
        @test simplify(f + Z) === f
        @test simplify(Z + f) === f
        @test simplify(f * I1) === f
        @test simplify(f * Z) === Z
        @test simplify(f - Z) === f
        @test simplify(Z - f) == -f
        @test simplify(f / I1) === f

        # Double negation.
        @test simplify(-(-f)) === f

        # Constant folding — produces Const, NOT Zero (strict, no auto-fold).
        @test simplify(Const(2.0) + Const(3.0)) === Const(5.0)
        @test simplify(Const(2.0) * Const(0.0)) === Const(0.0)
        @test !(simplify(Const(2.0) * Const(0.0)) isa Zero)

        # A nested mix folds and collapses.
        @test simplify((f + Z) * (Const(2.0) + Const(3.0))) == f * Const(5.0)
    end

    @testset "differentiate" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()

        @testset "forward difference δ₊{1}(f)" begin
            sst = differentiate(δ₊{1}(f), f)            # f[i+1] - f[i]
            @test sst isa Stencil
            @test AccessStyle(sst) === RowAccess()
            @test sst.shifts === (ô, ê₁)                # reverse-lex (offset 0, then +1)
            @test eltype(sst.term) === SVector{2, Float64}
            # narrows to a contiguous LinearStencil along axis 1, offsets 0:1
            ln = as_linear(sst)
            @test ln isa LinearStencil{1, 0, 2}
            @test AccessStyle(ln) === RowAccess()
            @test ln.term === sst.term                  # verbatim
        end

        @testset "the design-doc example" begin
            # g = f[i-2] - 4 f[i-1] + 3 f[i]; ∂/∂f has constant coefficients.
            expr = f[-2ê₁] - 4f[-ê₁] + 3f[]
            sst = differentiate(expr, f)
            @test sst.shifts === (-2ê₁, -ê₁, ô)
            @test as_linear(sst) isa LinearStencil{1, -2, 3}
        end

        @testset "variable coefficient: ∂(f*g)/∂f = g" begin
            sst = differentiate(f * g, f)
            @test sst.shifts === (ô,)                   # local (diagonal)
            # the lone coefficient is g (a Slot ⇒ a position-dependent coefficient)
            @test sst.term == Term(SVector, (g,))
        end

        @testset "nonlinear: ∂(f*f)/∂f = f + f" begin
            sst = differentiate(f * f, f)
            @test sst.shifts === (ô,)
            @test sst.term == Term(SVector, (f + f,))   # summed at the shared offset
        end

        @testset "Laplacian-shape narrows to a star" begin
            # δ₋{1}(δ₊{1}(f)) + δ₋{2}(δ₊{2}(f)) ⇒ 2-D L=1 five-point star.
            lap = δ₋{1}(δ₊{1}(f)) + δ₋{2}(δ₊{2}(f))
            sst = differentiate(lap, f)
            @test sst.shifts === (-ê₂, -ê₁, ô, ê₁, ê₂)  # reverse-lex star order
            @test as_star(sst) isa StarStencil{1, 2, 5}
        end

        @testset "independent / constant ⇒ error" begin
            @test_throws ArgumentError differentiate(g, f)        # g ≠ f
            @test_throws ArgumentError differentiate(Const(2.0) * g, f)
        end
    end

    @testset "materialize" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}(); τ = Scalar{:τ, Float64}()

        @testset "local elementwise" begin
            fv = collect(1.0:6.0); gv = collect(10.0:10.0:60.0)
            la = materialize(f * g + 2, (f = fv, g = gv))
            @test la isa LazyArray{Float64, 1}
            @test axes(la) == (1:6,)
            @test la[3] == fv[3] * gv[3] + 2
            @test collect(la) == fv .* gv .+ 2
        end

        @testset "shift shrinks axes (forward difference)" begin
            fv = rand(16)
            la = materialize(δ₊{1}(f), (f = fv,))     # f[i+1] - f[i]
            @test axes(la) == (1:15,)
            @test la[1] == fv[2] - fv[1]
            @test [la[i] for i in 1:15] == diff(fv)
        end

        @testset "scalar parameter (un-indexed)" begin
            fv = collect(1.0:5.0)
            la = materialize(τ * f, (f = fv, τ = 0.5))
            @test la[4] == 0.5 * fv[4]
            @test collect(la) == 0.5 .* fv
        end

        @testset "2-D + intersection of shifted axes" begin
            fv = reshape(collect(1.0:20.0), 4, 5)
            la = materialize(f[ê₁] - f[ê₂], (f = fv,))   # f[i+1,j] - f[i,j+1]
            @test axes(la) == (1:3, 1:4)
            @test la[2, 3] == fv[3, 3] - fv[2, 4]
        end
    end

    @testset "code_string" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        src = code_string(g * δ₊{1}(f); name = :advect)
        @test occursin("function advect(args, i)", src)
        @test occursin("args.f[i + 1]", src)
        @test occursin("args.g[i]", src)
        # round-trips to a runnable kernel
        fn = eval(Meta.parse(src))
        fv = collect(1.0:5.0); gv = collect(2.0:6.0)
        @test fn((; f = fv, g = gv), 2) == gv[2] * (fv[3] - fv[2])
    end

end
