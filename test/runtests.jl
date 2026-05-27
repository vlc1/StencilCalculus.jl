using StencilCalculus
using Test
using AbstractTrees
using StaticArrays: SVector, SMatrix, SUnitRange
using StencilCore: AbstractStencil, Stencil, LinearStencil, StarStencil, AccessStyle,
                   RowAccess, ColumnAccess, as_linear, as_star
using StencilAssembly: build

@testset "StencilCalculus" begin

    @testset "leaf construction + eltype" begin
        f = Slot{:f, Float64}()
        @test f isa AbstractPointwise{Float64}
        @test eltype(f) === Float64
        @test Slot{:g}() isa Slot{:g, Float64}                  # default T = Float64
        # `Zero` is the `Fill{Null{T}}` alias; the bool-shape collapse means
        # `Zero(Float64)` literally has eltype `Bool` (it stores a Null{Bool}).
        # Promotion in surrounding arithmetic still yields the right cell type.
        @test eltype(Zero(Float64)) === Bool
        @test eltype(One{Int}()) === Int
        # Outer ctors bool-shape the cell type, so `Zero(Integer)` and
        # `Zero(Bool)` are the same singleton.
        @test Zero(Integer) === Zero(Bool)
        # T must be concrete: an abstract eltype can never be materialized.
        @test_throws ArgumentError Slot{:f, Number}()
        @test_throws ArgumentError One{Real}()

        # Fill: literal payload (eltype = T) and AbstractScalar payload
        # (eltype unwraps the scalar via recursive specialization).
        @test Fill(2.5) isa Fill{Float64} <: AbstractPointwise{Float64}
        @test eltype(Fill(2)) === Int
        τ = Var{:τ, Float64}()
        @test Fill(τ) isa Fill{Var{:τ, Float64}} <: AbstractPointwise
        @test eltype(Fill(τ)) === Float64                      # recursive eltype
        @test eltype(Fill(Constant(2.0))) === Float64
    end

    @testset "constructor macros" begin
        @slot a                                                 # default T = Float64
        @slot b Float32
        @var τ
        @var dt Float32
        α = Constant(1)
        β = Constant(2.5)
        @test a === Slot{:a, Float64}()
        @test b === Slot{:b, Float32}()
        @test τ === Var{:τ, Float64}()
        @test dt === Var{:dt, Float32}()
        @test α === Constant(1)
        @test β === Constant(2.5)
        # the bound name drives the symbol parameter and composes in expressions
        @slot ψ
        @test ψ === Slot{:ψ, Float64}()
        @test (τ .* ψ) isa Pointwise{typeof(*)}                      # Symbolic .× Slot → Pointwise
    end

    @testset "broadcast builds Pointwise" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        @test (f .+ g) isa Pointwise{typeof(+)}
        @test eltype(f .+ g) === Float64
        # numeric literals wrap into Fill(Constant(…))
        t = f ./ 2
        @test t isa Pointwise{typeof(/)}
        @test t.args[2] === Fill(Constant(2))
        @test eltype(f ./ 2) === Float64
        # unary
        @test (.-f) isa Pointwise{typeof(-)}
        @test eltype(sin.(f)) === Float64
        # mixed eltype promotes
        @test eltype(Slot{:a, Float64}() .+ Slot{:b, Int}()) === Float64
        # AbstractScalar .× AbstractPointwise wraps the scalar in a Fill leaf.
        τ = Var{:τ, Float64}()
        @test (τ .* f) isa Pointwise{typeof(*)}
        @test (τ .* f).args[1] === Fill(τ)

        # Each `.op` produces one Pointwise node — nested dot calls preserve
        # tree shape (no Broadcast.flatten / fusion).
        nested = f .* (g .+ Fill(Constant(2.0)))
        @test nested == Pointwise(*, (f, Pointwise(+, (g, Fill(Constant(2.0))))))

        # Un-dotted Base operators are NOT overloaded on AbstractPointwise for scalar eltypes.
        @test_throws MethodError f * g
        @test_throws MethodError f + g
        @test_throws MethodError sin(f)
        @test_throws MethodError -f
        @test_throws MethodError 2 * f
        # Broadcast with no AbstractPointwise operand errors.
        ψ = Var{:ψ, Float64}()
        @test_throws ArgumentError τ .* ψ
        @test_throws ArgumentError 2.0 .* τ
        @test_throws ArgumentError sin.(τ)
    end

    @testset "Pointwise eltype: promote_op + Union{} throws" begin
        a = Slot{:a, Float64}(); b = Slot{:b, Float64}()
        # SVector interception → SVector-valued term
        v = SVector(a, b)
        @test v isa Pointwise{<:Any}
        @test eltype(v) === SVector{2, Float64}
        # Genuine inhomogeneity is unconstructable.
        @test_throws ArgumentError Pointwise(+, (Slot{:s, String}(), Slot{:n, Float64}()))
    end

    @testset "un-dotted ops for StaticArray slots" begin
        @slot u SVector{2, Float64}
        @slot v SVector{2, Float64}

        # + and - build Pointwise in the same way as .+ and .-
        @test u + v === Pointwise(+, (u, v))
        @test u - v === Pointwise(-, (u, v))
        @test -u    === Pointwise(-, (u,))
        @test eltype(u + v) === SVector{2, Float64}

        # 2u / u * 2 wraps the number via asterm → Fill(Constant(…))
        @test 2u    === Pointwise(*, (Fill(Constant(2)),   u))
        @test u * 2 === Pointwise(*, (u, Fill(Constant(2))))
        @test eltype(2u) === SVector{2, Float64}

        # Scalar Float64 slots still raise MethodError for un-dotted ops.
        f = Slot{:f, Float64}()
        @test_throws MethodError f + u   # mixed: f is not <:StaticArray
    end

    @testset "StaticArray constant in broadcast" begin
        V2  = SVector{2, Float64}
        M22 = SMatrix{2, 2, Float64, 4}

        @slot u V2
        mat = M22(1., 0., 0., 1.)

        # mat .* u  →  Pointwise(*, (Fill(Constant(mat)), u))
        result = mat .* u
        @test result isa Pointwise{typeof(*)}
        @test result.args[1] === Fill(Constant(mat))
        @test result.args[2] === u
        @test eltype(result) === SVector{2, Float64}  # matrix * vector → vector

        # reversed order
        result2 = u .* mat
        @test result2 isa Pointwise{typeof(*)}
        @test result2.args[1] === u
        @test result2.args[2] === Fill(Constant(mat))
    end

    @testset "getindex shift sugar" begin
        f = Slot{:f, Float64}()
        @test f[] === f                                         # zero shift = identity
        @test f[ô] === f
        sh = f[-ê₁]
        @test sh isa Shifted
        @test sh.shift === -ê₁
        @test sh.term === f
        # composition adds shifts; cancellation returns the bare slot
        @test f[-ê₁][ê₁] === f
        @test f[-ê₁][-ê₁].shift === -2ê₁
        # the DSL expression from the design docs builds
        g = f[-2ê₁] .- 4 .* f[-ê₁] .+ 3 .* f[]
        @test g isa Pointwise

        # Position-independent term leaves are shift-invariant.
        Z = Zero(Float64); I1 = One{Float64}(); φ = Fill(Constant(3.0))
        @test Z[ê₁] === Z && Z[] === Z
        @test I1[ê₁] === I1 && I1[] === I1
        @test φ[3ê₁ + ê₂] === φ && φ[] === φ
        # A bare AbstractScalar is also shift-invariant (handled in Core).
        τ = Var{:τ, Float64}()
        @test τ[] === τ && τ[ê₁] === τ && τ[3ê₁ + ê₂] === τ
    end

    @testset "display (normal-form, component form)" begin
        f = Slot{:f, Float64}(); ϕ = Slot{:ϕ, Float64}()
        τ = Var{:τ, Float64}(); x = Slot{:x, Float64}()
        @test repr(f) == "f[]"                                  # component form
        @test repr(ϕ[ê₁]) == "ϕ[ê₁]"                            # shifted slot
        @test repr(f[-2ê₁]) == "f[-2ê₁]"
        @test repr(τ) == "τ"                                    # bare symbolic
        @test repr(Constant(2.0)) == "2.0"                         # bare const (scalar-side)
        @test repr(Fill(Constant(2.0))) == "2.0"                   # Fill renders its payload
        @test repr(Fill(τ)) == "τ"                              # symbolic Fill
        @test repr(Zero(Float64)) == "0"                        # type-agnostic glyphs
        @test repr(One{Float64}()) == "1"
        @test repr(τ .* δ₊{1}(f)) == "(τ * (f[ê₁] - f[]))"      # infix
        @test repr(Pointwise(exp, (x,))) == "exp(x[])"          # call form
        @test repr(SVector(f, x)) == "SVector(f[], x[])"
        @test repr(.-f) == "-f[]"                               # unary minus
        # the ∂ / Diff functor (over a Slot vs a Var)
        @test repr(∂(f)) == "∂(f[])"
        @test repr(∂(τ)) == "∂(τ)"
        # display shows the normal form: f .- 0 collapses to f[]
        @test repr(f .- Zero(Float64)) == "f[]"
    end

    @testset "non-local functors" begin
        f = Slot{:f, Float64}()
        d = δ₊{1}(f)                                            # f[i+1] - f[i]
        @test d isa Pointwise{typeof(-)}
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
        t = f .+ g
        @test AbstractTrees.nodevalue(t) === (+)
        @test AbstractTrees.children(t) === (f, g)
        @test AbstractTrees.children(f) === ()
        @test AbstractTrees.nodevalue(f) === :f
        @test AbstractTrees.nodevalue(Fill(Constant(2.5))) === Constant(2.5)
        # The `Zero` alias = `Fill{<:Null}`, so its nodevalue is the wrapped Null
        # (via the generic `Fill` overload) — not a numeric zero literal.
        @test AbstractTrees.nodevalue(Zero(Float64)) === Null{Bool}()
        @test AbstractTrees.nodevalue(One{Int}()) === 1
        sh = f[ê₁]
        @test AbstractTrees.children(sh) === (f,)
        @test AbstractTrees.nodevalue(sh) === ê₁
    end

    @testset "simplify" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        Z = Zero(Float64); I1 = One{Float64}()

        # No-op on a normal-form expression / a leaf.
        @test simplify(f) === f
        @test simplify(f .+ g) == f .+ g

        # Shift composition (built manually, bypassing getindex's eager merge).
        nested = Shifted(ê₁, Shifted(ê₁, f))
        @test simplify(nested) === f[2ê₁]
        @test simplify(Shifted(ê₁, Shifted(-ê₁, f))) === f      # cancels to identity

        # Shift pushdown to leaves.
        d = δ₊{1}(f .+ g)                                       # Shifted(ê₁, f.+g) - (f.+g)
        s = simplify(d)
        @test s == (f[ê₁] .+ g[ê₁]) .- (f .+ g)

        # Shift over a Fill (position-independent) is a no-op.
        @test simplify(Shifted(ê₁, Fill(Constant(2.0)))) === Fill(Constant(2.0))
        @test simplify(Shifted(ê₁, Zero(Float64))) === Z         # also Zero / One
        @test simplify(Shifted(ê₁, One{Float64}())) === I1

        # Identity / annihilator on Zero/One (type-dispatched).
        @test simplify(f .+ Z) === f
        @test simplify(Z .+ f) === f
        @test simplify(f .* I1) === f
        @test simplify(f .* Z) === Z
        @test simplify(f .- Z) === f
        @test simplify(Z .- f) == .-f
        @test simplify(f ./ I1) === f

        # Identity on `Fill{<:Null}` / `Fill{<:Unity}` (type-dispatched through Fill).
        nf = Fill(Null(Float64)); uf = Fill(Unity(Float64))
        @test simplify(f .* uf) === f
        @test simplify(f .* nf) === Zero(Float64)
        @test simplify(f .+ nf) === f

        # Value-based identity on `Fill{<:Const}` (mathematically correct;
        # the previous "strict no-auto-fold" stance is intentionally walked
        # back — the new design treats a literal Fill(Constant(0)) as a real zero).
        @test simplify(f .* Fill(Constant(0.0))) === Zero(Float64)
        @test simplify(f .* Fill(Constant(1.0))) === f

        # Double negation.
        @test simplify(.-(.-f)) === f

        # Scalar-side folding through scalar simplify (Core's rule_fold_scalar).
        @test simplify(Constant(2.0) + Constant(3.0)) === Constant(5.0)
        @test simplify(Constant(2.0) * Constant(0.0)) === Constant(0.0)

        # rule_fill_collapse: all-Fill `Pointwise` becomes one Fill-of-Scalar,
        # whose inner scalar is then simplified by Core.
        coll = simplify(Fill(Constant(2.0)) .+ Fill(Constant(3.0)))
        @test coll === Fill(Constant(5.0))
        @test simplify(Fill(Constant(2.0)) .* Fill(Var{:τ, Float64}())) ==
              Fill(Scalar(*, (Constant(2.0), Var{:τ, Float64}())))

        # A nested mix folds (in scalar-land), collapses to a Fill, then leaves
        # the outer Pointwise(*) intact (no further annihilation since 5 ≠ 0/1).
        @test simplify((f .+ Z) .* (Fill(Constant(2.0)) .+ Fill(Constant(3.0)))) == f .* 5.0

        # Double-negation rule (now a named rule, separately testable).
        @test simplify(.-(.-f)) === f
        @test simplify(.-(.-f) .+ g) == f .+ g

        # POINTWISE_DEFAULT_RULES is exported and composable.
        @test POINTWISE_DEFAULT_RULES isa Tuple
        my_rules = (POINTWISE_DEFAULT_RULES...,)
        @test simplify(.-(.-f), my_rules) === f
    end

    @testset "differentiate" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()

        @testset "forward difference δ₊{1}(f)" begin
            sst = differentiate(δ₊{1}(f), f)                    # f[i+1] - f[i]
            @test sst isa Stencil
            @test AccessStyle(sst) === RowAccess()
            @test sst.shifts === (ô, ê₁)                        # reverse-lex (offset 0, then +1)
            @test sst.terms == (Pointwise(-, (One{Float64}(),)), One{Float64}())   # SoA: one coef per offset
            # narrows to a contiguous LinearStencil along axis 1, offsets 0:1
            ln = as_linear(sst)
            @test ln isa LinearStencil{1, 0, 2}
            @test AccessStyle(ln) === RowAccess()
            @test ln.term == Pointwise(SVector, sst.terms)           # SoA terms interlaced into AoS
        end

        @testset "the design-doc example" begin
            # g = f[i-2] - 4 f[i-1] + 3 f[i]; ∂/∂f has constant coefficients.
            # Literals are Float64 to match the slot's eltype — required for
            # the strict eltype-uniformity check in Stencil's ctor.
            expr = f[-2ê₁] .- 4.0 .* f[-ê₁] .+ 3.0 .* f[]
            sst = differentiate(expr, f)
            @test sst.shifts === (-2ê₁, -ê₁, ô)
            @test as_linear(sst) isa LinearStencil{1, -2, 3}
        end

        @testset "variable coefficient: ∂(f.*g)/∂f = g" begin
            sst = differentiate(f .* g, f)
            @test sst.shifts === (ô,)                           # local (diagonal)
            @test sst.terms == (g,)
        end

        @testset "nonlinear: ∂(f.*f)/∂f = f .+ f" begin
            sst = differentiate(f .* f, f)
            @test sst.shifts === (ô,)
            @test sst.terms == (f .+ f,)                        # summed at the shared offset
        end

        @testset "Laplacian-shape narrows to a star" begin
            # δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f)) ⇒ 2-D L=1 five-point star.
            lap = δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f))
            sst = differentiate(lap, f)
            @test sst.shifts === (-ê₂, -ê₁, ô, ê₁, ê₂)          # reverse-lex star order
            @test as_star(sst) isa StarStencil{1, 2, 5}
        end

        @testset "independent / constant ⇒ error" begin
            @test_throws ArgumentError differentiate(g, f)            # g ≠ f
            @test_throws ArgumentError differentiate(Constant(2.0) .* g, f)  # constant-coef ⊥ f
        end

        @testset "∂ / Diff functor + differentiation w.r.t. a Var" begin
            τ = Var{:τ, Float64}()
            # w.r.t. a Var → an AbstractPointwise (no spatial offsets)
            @test ∂(τ)(τ .* f) === f
            @test differentiate(τ .* f, τ) === f
            @test ∂(f)(τ .* f) == differentiate(τ .* f, f)      # w.r.t. a Slot → Stencil
            @test ∂(f)(τ .* f) isa Stencil
            # a Slot and a Var of the same symbol do not collide
            s = Slot{:τ, Float64}()
            @test ∂(τ)(τ .* s) === s                            # ∂/∂(symbolic τ)
            @test ∂(s)(τ .* s) isa Stencil                      # ∂/∂(slot τ)
            # independence throws for the Var path too
            @test_throws ArgumentError differentiate(f, τ)
            # the default-typed pipeline works now that T is concrete (Float64)
            @slot fd
            @var td
            @test ∂(td)(td .* fd) === fd
            @test ∂(fd)(td .* fd) isa Stencil
        end

        @testset "tan and abs derivative rules" begin
            # Both tan and abs are valid broadcast unary ops on AbstractPointwise.
            @test differentiate(tan.(f), f) isa Stencil
            @test differentiate(abs.(f), f) isa Stencil
            # ∂tan(f)/∂f = 1 + tan²(f) — check structure
            d_tan = only(differentiate(tan.(f), f).terms)
            @test d_tan isa Pointwise   # a non-trivial expression
            # ∂|f|/∂f = sign(f)
            d_abs = only(differentiate(abs.(f), f).terms)
            @test d_abs isa Pointwise{typeof(sign)}
        end

        @testset "@pointwise_rule macro" begin
            # Define a new primitive `myop` and its derivative via the macro.
            myop(x::Float64) = x^3
            @pointwise_rule myop(x) = Pointwise(*, (Fill(Constant(3.0)), Pointwise(^, (x, Fill(Constant(2.0))))))
            s = Slot{:s, Float64}()
            d = differentiate(Pointwise(myop, (s,)), s)
            @test d isa Stencil
            coef = only(d.terms)
            @test coef isa Pointwise   # 3 * s^2
            # The macro errors with the wrong arity (2 LHS args, 3 RHS partials).
            # The ArgumentError is wrapped in a LoadError by @eval.
            @test_throws "2 argument(s)" @eval @pointwise_rule myop2(x, y) = (y, x, y)
        end
    end

    @testset "materialize" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}(); τ = Var{:τ, Float64}()

        @testset "local elementwise" begin
            fv = collect(1.0:6.0); gv = collect(10.0:10.0:60.0)
            la = materialize(f .* g .+ 2, (f = fv, g = gv))
            @test la isa LazyArray{Float64, 1}
            @test axes(la) == (1:6,)
            @test la[3] == fv[3] * gv[3] + 2
            @test collect(la) == fv .* gv .+ 2
        end

        @testset "shift shrinks axes (forward difference)" begin
            fv = rand(16)
            la = materialize(δ₊{1}(f), (f = fv,))               # f[i+1] - f[i]
            @test axes(la) == (1:15,)
            @test la[1] == fv[2] - fv[1]
            @test [la[i] for i in 1:15] == diff(fv)
        end

        @testset "symbolic parameter (un-indexed)" begin
            fv = collect(1.0:5.0)
            la = materialize(τ .* f, (f = fv, τ = 0.5))
            @test la[4] == 0.5 * fv[4]
            @test collect(la) == 0.5 .* fv
        end

        @testset "2-D + intersection of shifted axes" begin
            fv = reshape(collect(1.0:20.0), 4, 5)
            la = materialize(f[ê₁] .- f[ê₂], (f = fv,))         # f[i+1,j] - f[i,j+1]
            @test axes(la) == (1:3, 1:4)
            @test la[2, 3] == fv[3, 3] - fv[2, 4]
        end
    end

    @testset "code_string" begin
        f = Slot{:f, Float64}(); g = Slot{:g, Float64}()
        src = code_string(g .* δ₊{1}(f); name = :advect)
        @test occursin("function advect(args, i)", src)
        @test occursin("args.f[i + 1]", src)
        @test occursin("args.g[i]", src)
        # round-trips to a runnable kernel
        fn = eval(Meta.parse(src))
        fv = collect(1.0:5.0); gv = collect(2.0:6.0)
        @test fn((; f = fv, g = gv), 2) == gv[2] * (fv[3] - fv[2])
    end

    @testset "build_stencil bridge" begin
        f = Slot{:f, Float64}(); ψ = Slot{:ψ, Float64}()

        @testset "constant coefficient → LinearStencil, end-to-end" begin
            # ∂(δ₊{1}(f))/∂f is a constant forward-difference stencil.
            sst = differentiate(δ₊{1}(f), f)
            n = 6
            st = build_stencil(sst; size = (n,))
            @test st isa LinearStencil{1, 0, 2, Float64, <:Any, ColumnAccess}
            @test st.term[1] == SVector(-1.0, 1.0)              # column-anchored coefficients
            ref = build(LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), n)),
                        (1:n,), (1:n,))
            @test build(st, (1:n,), (1:n,)) == ref
            A = build(st, (1:n,), (1:n,)); fv = rand(n)
            @test (A * fv)[1:n-1] ≈ diff(fv)
        end

        @testset "Laplacian (constant) → StarStencil, end-to-end" begin
            lap = δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f))            # +Laplacian: f[i±1]+f[j±1]-4f
            sst = differentiate(lap, f)
            n1, n2 = 5, 4
            st = build_stencil(sst; size = (n1, n2))
            @test st isa StarStencil{1, 2, 5, Float64, <:Any, ColumnAccess}
            @test st.term[2, 2] == SVector(1.0, 1.0, -4.0, 1.0, 1.0)
            ref = build(StarStencil{1}(fill(SVector(1.0, 1.0, -4.0, 1.0, 1.0), n1, n2)),
                        (1:n1, 1:n2), (1:n1, 1:n2))
            @test build(st, (1:n1, 1:n2), (1:n1, 1:n2)) == ref
        end

        @testset "variable coefficient: column-anchored shift" begin
            # ∂(ψ .* δ₊{1}(f))/∂f: row-anchored (-ψ, ψ); column-anchored
            # (-ψ[c], ψ[c-1]). Materialize over ψ; coefficient axes shrink.
            sst = differentiate(ψ .* δ₊{1}(f), f)
            @test AccessStyle(sst) === RowAccess()
            ψv = collect(1.0:8.0)
            st = build_stencil(sst, (ψ = ψv,))
            @test st isa LinearStencil{1, 0, 2, Float64, <:Any, ColumnAccess}
            # column c=3 (within the shrunk axes): SVector(-ψ[3], ψ[2])
            @test st.term[3] == SVector(-ψv[3], ψv[2])
            @test axes(st.term, 1) == 2:8                       # shrunk by the −1 coefficient shift
        end

        @testset "offset-padding (densify) for a gappy result" begin
            f = Slot{:f, Float64}()
            sst = differentiate(f[-2ê₁] .+ 3.0 .* f[], f)       # offsets {-2, 0} — gap at -1
            @test sst.shifts === (-2ê₁, ô)

            # densify fills the gap with a Zero coefficient.
            d = densify(sst)
            @test d.shifts === (-2ê₁, -ê₁, ô)
            @test d.terms[2] isa Zero                           # inserted at offset -1

            n = 7
            # without padding, the gappy stencil cannot narrow
            @test_throws ArgumentError build_stencil(sst; size = (n,))
            # with padding it narrows and assembles
            st = build_stencil(sst; size = (n,), pad = true)
            @test st isa LinearStencil{1, -2, 3, Float64, <:Any, ColumnAccess}
            @test st.term[1] == SVector(1.0, 0.0, 3.0)
            ref = build(LinearStencil{1}(SUnitRange(-2, 0), fill(SVector(1.0, 0.0, 3.0), n)),
                        (1:n,), (1:n,))
            @test build(st, (1:n,), (1:n,)) == ref

            # densify is a no-op on an already-contiguous result and on a star
            @test densify(differentiate(δ₊{1}(f), f)) === differentiate(δ₊{1}(f), f) ||
                  densify(differentiate(δ₊{1}(f), f)).shifts === (ô, ê₁)
            lap = δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f))
            @test densify(differentiate(lap, f)).shifts === (-ê₂, -ê₁, ô, ê₁, ê₂)
        end
    end

    @testset "stencil * pointwise (shells)" begin
        # The `*(::AbstractStencil, ::AbstractPointwise)` surface is reserved
        # for stencil application but currently stubbed; calling any of the
        # three concrete-subtype methods throws `"not yet implemented"`.
        f = Slot{:f, Float64}()

        # Build one instance of each concrete stencil subtype. `Stencil` and
        # `LinearStencil` come naturally from differentiation; `StarStencil`
        # comes from the Laplacian pattern.
        sst_general = differentiate(δ₊{1}(f), f)                  # Stencil{RowAccess}
        @test sst_general isa Stencil

        sst_linear = build_stencil(differentiate(δ₊{1}(f), f); size = (6,))
        @test sst_linear isa LinearStencil

        sst_star = build_stencil(
            differentiate(δ₋{1}(δ₊{1}(f)) .+ δ₋{2}(δ₊{2}(f)), f); size = (5, 4))
        @test sst_star isa StarStencil

        # Each shell throws ErrorException with the expected message fragment.
        for st in (sst_general, sst_linear, sst_star)
            err = try
                st * f
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("not yet implemented", err.msg)
        end

        # Regression guard: the deliberate "no `*` between two AbstractPointwise"
        # discipline still holds. The shells dispatch only on AbstractStencil
        # left operands and must not accidentally widen pointwise-pointwise
        # multiplication.
        @test_throws MethodError f * f
        let g = Slot{:g, Float64}()
            @test_throws MethodError f * g
        end

        # And the dispatch contract is visible: exactly three * methods whose
        # first positional type is a concrete AbstractStencil subtype.
        ms = methods(*)
        stencil_methods = filter(m -> begin
            sig = m.sig
            sig isa UnionAll && (sig = Base.unwrap_unionall(sig))
            length(sig.parameters) >= 2 || return false
            t = sig.parameters[2]
            t isa Type && t <: AbstractStencil
        end, collect(ms))
        @test length(stencil_methods) == 3
    end

end
