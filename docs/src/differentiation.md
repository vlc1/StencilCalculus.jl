# Differentiation, display, and the algebra behind it

This page collects the design decisions around *differentiating* grid
expressions and *displaying* them, and then steps back to look at the
mathematical structure they sit in. It is part design record, part roadmap:
each section flags clearly what the package does **today** versus what is a
**direction** not yet built.

## Two parallel algebras

A grid expression has two leaf kinds, with sibling algebras that mirror each
other (see also [StencilCore](https://vlc1.github.io/StencilCore.jl/dev/)):

| role                    | `AbstractScalar` (Core)        | `AbstractTerm` (here)       |
|-------------------------|--------------------------------|-----------------------------|
| named substitution leaf | `Symbolic{S, T}`               | `Slot{S, T}`                |
| literal carrier         | `Const{T}` with `val::T`       | — (via `Fill(Const(…))`)    |
| interior tree node      | `Scalar{F, A, T}`              | `Term{F, A, T}`             |
| additive identity       | `Null{T}`                      | `Zero{T}`                   |
| multiplicative identity | `Unity{T}`                     | `One{T}`                    |

A scalar has no spatial extent (one materialized value); a term is array-like
(per-cell). The two are bridged by **[`Fill{T}`](@ref)**, a term that wraps a
single value (literal or `AbstractScalar`) and broadcasts it spatially.
`Term.args` is always `Tuple{Vararg{AbstractTerm}}` — scalars never appear
inside a `Term` directly; they enter through a `Fill` leaf at the operator
boundary.

## Scalars are shift-invariant

A bare `AbstractScalar` — a [`Symbolic`](@ref), a [`Const`](@ref), a
[`Null`](@ref), a [`Unity`](@ref), or an interior `Scalar` tree — has no
spatial extent. Translating it is therefore the identity:

```julia
@symbolic τ Float64
τ[ê₁]              # === τ
τ[3ê₁ + ê₂]        # === τ
Const(2.0)[ê₁]     # === Const(2.0)
```

A `Fill` in term-land carries the same property: it is the spatially-invariant
broadcast of one value, so indexing it is also a no-op:

```julia
Fill(τ)[ê₁]        # === Fill(τ)
Fill(Const(2.0))[3ê₁ + ê₂]   # === Fill(Const(2.0))
```

The same rule lives in `simplify` as
`rule_shift_const`: `Shifted(s, ::Fill) → ::Fill`, and analogously for `Zero`
and `One`. A shift can never reach a position-independent leaf, whether it is
written directly (`f[ê₁]` collapses) or produced by pushing a shift down a
tree.

## Displaying the normal form

`show` renders a term in **component form**, and it renders the term's *normal
form* — it calls [`simplify`](@ref) first (display only; the object is never
mutated). The conventions:

| term                       | shows as              |
|----------------------------|-----------------------|
| `Slot{:f}()`               | `f[]`                 |
| `f[ê₁]` (a shift)          | `f[ê₁]`               |
| `Zero{T}()`                | `0`                   |
| `One{T}()`                 | `1`                   |
| `Fill(Const(2.0))`         | `2.0`                 |
| `Fill(Symbolic{:τ,T}())`   | `τ`                   |
| `Fill(Scalar(*, (Const(2), τ)))` | `(2 * τ)`       |
| `Term`                     | infix for `+ - * / \ ^`, else call form |

```julia
@slot f
@symbolic τ
repr(τ * δ₊{1}(f))           # "(τ * (f[ê₁] - f[]))"
repr(f - Zero{Float64}())    # "f[]"  — display is of the simplified form
```

A `Fill` prints as its wrapped value via the scalar-side `show`, without
square brackets — `Fill(τ)` is `τ`, *not* `τ[]`. The bracket notation is
reserved for spatially-indexed leaves (`Slot` and `Shifted`).

!!! note "Why the glyphs `0`/`1`"
    `Zero`/`One` (and their scalar-side analogues `Null`/`Unity`) are
    *symbolic* identities — structure, not data — so they print as the bare
    glyphs `0`/`1` regardless of `T`. `Zero{Float64}` shows `0`, not `0.0`.
    The display stays type-agnostic and no value is ever constructed, which
    keeps it faithful to their role as the structural neutrals that drive
    `simplify` and make the chain rule collapse.

## Differentiation: the concrete behaviour

[`differentiate`](@ref) (and its sugar `∂`, below) walks the expression with a
ChainRules-style `frule` table (`derivative`) and the chain rule. There are
two methods on the same generic, dispatched by the kind of the variable:

- **`differentiate(t::AbstractTerm, ::Slot)`** walks the *term* tree and
  collects one coefficient per lattice offset — a row-anchored `Stencil`.
- **`differentiate(t::AbstractTerm, ::Symbolic)`** walks the term tree but
  defers `Fill{<:AbstractScalar}` leaves to the **scalar-side**
  [`differentiate`](@ref) (a third method, on the same generic, for
  `AbstractScalar → AbstractScalar`). The result is re-wrapped in a `Fill` on
  the way back into term-land, producing a single broadcast coefficient — an
  `AbstractTerm`, not a `Stencil`.

**Is the derivative of `One{T}()` equal to `Zero{T}()`?** Effectively yes,
but no literal `Zero` object is produced for a Slot-independent leaf.
`One`, `Zero`, `Fill` and a `Shifted` of an unrelated `Slot` all contribute
the **empty** set of offset/coefficient pairs to the Slot derivative — which
is exactly "zero" in this representation. A derivative that is empty
everywhere makes `differentiate` throw rather than return a degenerate
stencil: the package deliberately does not fabricate spurious zeros.

On the scalar side the same story plays out with `Null` instead of an empty
set: a `Const`, `Null`, `Unity`, or a `Symbolic` that does not match the
differentiation variable returns `Null{T}()`. The chain-rule walker
short-circuits `Null` sub-derivatives, so an unsupported primitive does not
error when its branch contributes nothing.

### `∂` — the "with respect to" functor

[`∂`](@ref Diff) (alias `Diff`) wraps the variable and lowers to
`differentiate`:

```julia
∂(v)(e) == differentiate(e, v)
```

The variable's *kind* decides the result type:

- **`∂(slot)` → a [`Stencil`](@ref).** A slot is a spatially-extended field, so
  the derivative carries offsets — it is a row-anchored stencil.
- **`∂(symbolic)` → an `AbstractTerm`.** A symbolic has no spatial extent, so
  the per-offset structure collapses to a single coefficient term:

```julia
@symbolic τ Float64
@slot f Float64
∂(τ)(τ * f)        # === f          (a term)
∂(f)(τ * f)        # a Stencil      (offset ô, coefficient Fill(τ))
```

A `Slot` and a `Symbolic` that happen to share a symbol do **not** collide:
the differentiation variable is matched by *instance type*, not just by its
name, so `∂(Symbolic{:τ}())` and `∂(Slot{:τ}())` differentiate against
different leaves.

## The scalar-precedence rule

A `Term` whose arguments are *all* `Fill` is the broadcast of a scalar
expression — there is no spatial structure for the operation to act on
cell-by-cell. `simplify` therefore collapses it into a single
`Fill(Scalar(fn, vals…))`, with the inner scalar then simplified by
StencilCore's scalar `simplify`:

```julia
simplify(Fill(Const(2.0)) + Fill(Const(3.0)))            # Fill(Const(5.0))
simplify(Fill(Const(2.0)) * Fill(Symbolic{:τ,Float64}())) ==
    Fill(Scalar(*, (Const(2.0), Symbolic{:τ,Float64}())))
```

The rule only fires when *every* argument is a `Fill` — a `Term(*, (Fill(τ),
f))` (with a real Slot `f`) stays a `Term`, because there is genuine spatial
structure to broadcast over.

The corresponding identity check (`f * Fill(Const(0)) → Zero`) uses a small
helper, `_is_term_zero` / `_is_term_one`, that is **type-dispatched** on
`Zero`/`One` and on `Fill{<:Null}` / `Fill{<:Unity}`, and **value-dispatched**
on a literal `Fill{<:Const}` (via `iszero`/`isone` on the wrapped value). A
`Fill` wrapping a symbolic scalar (`Fill(Symbolic{S})`, `Fill(Scalar(…))`)
is never treated as an identity — its runtime value is unknown.

## Beyond `Number`: tensor-valued fields (direction)

`AbstractTerm{T}` and `AbstractScalar{T}` already carry the materialized
element type `T`. Nothing in the *structure* of differentiation assumes
`T <: Number`; only the `derivative` rules do. If a slot is `SVector`-valued
and we differentiate with respect to an `SVector`-valued variable, the
natural coefficient is an **`SMatrix`** — the local Jacobian block, by the
usual convention

```
J[i, j] = ∂(output_i) / ∂(input_j)         # rows: output, columns: input
```

so `Number × Number → Number`, `SVector{m} × SVector{n} → SMatrix{m, n}`.
Supporting this fully would require three things, none of them yet built:

1. **Matrix-valued `derivative` rules** for the vector primitives (e.g.
   `∂(A * x)/∂x = A`, `∂‖x‖/∂x = xᵀ/‖x‖`).
2. **A non-commutative chain rule.** With scalar coefficients the order of the
   chain-rule product is irrelevant; with matrix blocks it is not — the product
   must be assembled outer-to-inner in the correct order.
3. **An element-type promotion** `_jac_eltype(T_out, T_in)` driving the
   coefficient's type, replacing the scalar `*` currently used to combine
   partials.

The point worth recording: the *type-as-structure* design already reserves the
slot for this (the element type is tracked); it is the rule table and the
combiner that would grow.

## The hidden duality: variance

There is a tensor-calculus structure under all of this. A Jacobian
`∂Fⁱ / ∂xⱼ` has one **contravariant** index (the output `i`, from `F`) and one
**covariant** index (the input `j`, from the variable `x`) — it is a mixed
`(1,1)`-tensor, i.e. a linear map `V → W ≅ W ⊗ V*`. This is not decoration; it
maps directly onto two existing design choices.

- **`AccessStyle` is *which index is anchored*.** A stencil's coefficient is
  indexed by a single mesh point plus an offset; the anchor says whether that
  point is the *input* or the *output*. `ColumnAccess` (CSC) anchors the input
  (covariant) index — the column is the variable we differentiate against;
  `RowAccess` (CSR-to-be) anchors the output (contravariant) index — the row is
  the equation. `differentiate` emits `RowAccess` because the row is where the
  derivative is naturally evaluated.
- **CSC ↔ CSR is the transpose.** The bridge converts a row-anchored result to a
  column-anchored one by shifting each coefficient by `-σ`. That sign flip is
  exactly the index swap of a transpose / adjoint: a shift acting *covariantly*
  on the input becomes a shift on the output. The implicit Jacobian `∂F/∂ψ` and
  the explicit `−∂F/∂φ` are maps out of two different tangent spaces; their
  adjoints live in the dual.

The practical upshot is that "offsets are invariant under `AccessStyle`, only
the anchoring shifts" (a `StencilCore` rule) is the discrete shadow of raising
and lowering one index of a `(1,1)`-tensor.

## Stencils as morphisms: application, algebra, composition

An `AbstractStencil` is a *transformation of fields*: it should take a term and
return a term. That operation is **not implemented today**, but it is the right
way to think about the type, and it closes a loop with differentiation.

### Applying a stencil to a term (direction)

The natural definition is

```
S(t) = Σ_k  coef_k · t[σ_k]
```

— sum the coefficient at each offset times the field read at that offset. With
this, for an `F` that is **linear and homogeneous in `f`**, differentiating and
then re-applying recovers the original:

```
differentiate(F, f)  applied back to  f   ==   F        (F linear in f)
```

This is *not* the identity for a general `F`: constants and other slots are
annihilated by `∂/∂f`, so only the part of `F` that is linear and homogeneous in
`f` comes back. Stating it precisely is the useful part — it is the discrete
"a linear map equals its own Jacobian."

### A monoid and a ring, not a group

The intuition that *applying* a stencil is a kind of **multiplication** is
correct, and composing two stencils is composition of linear operators. But the
structure is a **monoid** under composition (associative; the identity is the
single-offset `ô` stencil with coefficient `1`) and, together with addition, a
**unital associative algebra** — *not a group*. A finite-difference stencil has
no stencil inverse (the inverse of a differential operator is a dense integral
operator), so inverses are simply absent.

Concretely:

- **Constant coefficients** form a *commutative* ring — the Laurent polynomials
  in the shift operators, `ℝ[ê₁^±¹, …, ê_N^±¹]`. Composition is convolution of
  offset patterns; everything commutes.
- **Variable coefficients** form a *non-commutative* ring — a difference-operator
  (Ore/skew) algebra, because a shift does not commute with multiplication by a
  position-dependent coefficient:

  ```
  ê ∘ (a · )  =  (Sₐ · ) ∘ ê        where  Sₐ  is a shifted by ê
  ```

  This commutation law is precisely what a correct composition routine has to
  honour.

### Composing stencils (direction)

Composition `S₁ * S₂` is a modest extension of what already exists: the
`StaticShift` `+` algebra already *composes offsets*, so a composition routine
walks pairs of offsets, adds them, and multiplies the coefficients — applying
the shift commutation above to the inner coefficient. It is deferred, not
difficult.

## A category-theory reading

The cleanest frame for everything above is categorical.

- **Objects** are field spaces over the mesh (an element type plus an index
  domain). **Morphisms** are linear stencils. Composition is associative and
  every object has an identity stencil, so this is a **category** `𝒮` — a
  subcategory of `R`-modules. Hom-sets are themselves vector spaces (stencils
  add), so `𝒮` is **enriched in `Vect`**; the endomorphisms of one object are
  the operator algebra of the previous section.
- **Differentiation is a differential combinator.** The pairing of a
  `derivative` table with the chain rule is exactly the structure axiomatised by
  **Cartesian differential categories** (Blute–Cockett–Seely) and **tangent
  categories** (Cockett–Cruttwell): an operator `D[-]` sending a map to its
  linearization, with the chain rule as its coherence law. What the package
  implements today is this combinator on the `Slot`/`Symbolic` fragment.
- **`AccessStyle` is the dagger.** CSC versus CSR — a morphism versus its
  transpose — is the adjoint, i.e. the `† : 𝒮 → 𝒮ᵒᵖ` structure. The bridge's
  `-σ` shift is the concrete formula for it.
- **"Jacobian, you said?" resolved.** On a mesh of dimension `> 1` these
  operators are not literally Jacobian *matrices* until the unknowns are given a
  linear numbering. That numbering is a faithful functor `𝒮 → FinVect_ℝ`
  (`LinearIndices` flattening each field space to `ℝⁿ`), under which a stencil
  becomes its honest `SparseMatrixCSC`. The abuse of calling the stencil a
  "Jacobian" is the informal name for its image under this functor.

What is implemented: the linear category of stencils, the parallel scalar
algebra, and the differential combinator on slots and symbolics. What is
aspirational: stencil application and composition (`*`), the dagger as an
explicit operation, and the tensor-valued (`SMatrix`) blocks of the section
above.
