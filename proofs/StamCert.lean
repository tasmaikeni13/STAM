/-
  Machine-checked statements of the elementary facts the STAM error analysis rests on.

  These are not the whole analysis -- the minimax lower bound and the rate separation are
  citations to classical nonparametric statistics, and the empirical certificate is by
  construction assumption-free.  What is formalised here is exactly the part that is
  original arithmetic and therefore worth checking mechanically:

    1. `alloc_lower_bound` / `alloc_attained`
         The budget-allocation optimum.  Splitting a budget between anchor count and
         per-anchor sample size gives an error of the form  a/u^3 + b*u  in
         u = sqrt(n); its minimum over u > 0 is exactly 4*(a*b^3/27)^(1/4), attained at
         u = (3a/b)^(1/4).  Substituting a = c1*M3*R^3 and b = c2*sigma*sqrt(kappa/C)
         turns this into the C^(-3/8) rate.

    2. `pu_error_bound`
         A partition of unity carries local accuracy to global accuracy with no loss.
         This is the step that lets a Taylor remainder bound on each patch become a
         bound on the whole rendered surface.

    3. `interpolation_floor`
         An estimator that reproduces its observations exactly has, at those
         observations, an error equal to the observation noise -- identically, not
         asymptotically.  This is why plotting a raw noisy grid cannot converge.

    4. `debias_unbiased`
         Subtracting an unbiased variance estimate from the squared residual gives an
         unbiased estimate of the squared model error.  The certificate is built on this.

    5. `sum_dist_sq_center`
         The sum of squared distances to a point is minimised at the mean, with an exact
         decomposition.  This is why the optimal *affine* plane is centred at the mean of
         the trajectory rather than anchored at its starting point.
-/

import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.Calculus.Taylor

namespace Stam

open Finset

/-! ## 1. Budget allocation -/

/-- **Allocation lower bound.**  For positive `a`, `b`, the two-term error
`a/u^3 + b*u` is bounded below by `4 * (a*b^3/27)^(1/4)` for every `u > 0`.

With `u = sqrt n`, `a = c₁ M₃ R³` (the approximation term, which falls as `n^(-3/2)`) and
`b = c₂ σ sqrt(κ/C)` (the Monte-Carlo term, which grows as `n^(1/2)` once the budget is
split `n` ways), the bound reads
`E ≥ 4 (c₁M₃R³ (c₂σ)³ / 27)^(1/4) (κ/C)^(3/8)`: no allocation of a budget `C` beats the
`C^(-3/8)` rate. -/
theorem alloc_lower_bound {a b u : ℝ} (ha : 0 < a) (hb : 0 < b) (hu : 0 < u) :
    4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) ≤ a / u ^ 3 + b * u := by
  have h1 : (0 : ℝ) ≤ a / u ^ 3 := by positivity
  have h2 : (0 : ℝ) ≤ b * u / 3 := by positivity
  -- Four-term AM-GM with equal weights, splitting the linear term into three copies.
  have key := Real.geom_mean_le_arith_mean4_weighted
    (by norm_num : (0:ℝ) ≤ 1/4) (by norm_num : (0:ℝ) ≤ 1/4)
    (by norm_num : (0:ℝ) ≤ 1/4) (by norm_num : (0:ℝ) ≤ 1/4)
    h1 h2 h2 h2 (by norm_num)
  have hprod : (a / u ^ 3) * (b * u / 3) * (b * u / 3) * (b * u / 3) = a * b ^ 3 / 27 := by
    field_simp
    ring
  rw [← Real.mul_rpow h1 h2, ← Real.mul_rpow (by positivity) h2,
      ← Real.mul_rpow (by positivity) h2, hprod] at key
  linarith

/-- **The bound is attained.**  Equality holds at `u = (3a/b)^(1/4)`, so the lower bound
above is the exact minimum and the allocation rule is optimal, not merely feasible. -/
theorem alloc_attained {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∃ u : ℝ, 0 < u ∧ a / u ^ 3 + b * u = 4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) := by
  set u : ℝ := (3 * a / b) ^ ((1 : ℝ) / 4) with hu_def
  have hpos : 0 < 3 * a / b := by positivity
  have hu : 0 < u := Real.rpow_pos_of_pos hpos _
  have hu4 : u ^ 4 = 3 * a / b := by
    rw [hu_def, ← Real.rpow_natCast ((3 * a / b) ^ ((1 : ℝ) / 4)) 4,
        ← Real.rpow_mul hpos.le]
    norm_num
  refine ⟨u, hu, ?_⟩
  -- At the optimum the two terms are in the ratio 1 : 3.
  have hsplit : a / u ^ 3 = b * u / 3 := by
    field_simp
    nlinarith [hu4, hu, pow_pos hu 3, pow_pos hu 4]
  have hval : (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) = b * u / 3 := by
    have h4 : (b * u / 3) ^ (4 : ℕ) = a * b ^ 3 / 27 := by
      have : u ^ 4 * b = 3 * a := by
        field_simp at hu4
        linarith [hu4]
      nlinarith [this, hb, hu]
    rw [← h4, ← Real.rpow_natCast (b * u / 3) 4, ← Real.rpow_mul (by positivity)]
    norm_num
  rw [hsplit, hval]
  ring

/-! ## 2. Partition of unity -/

/-- **Local accuracy transfers to global accuracy.**  If non-negative weights summing to
one blend local models each within `ε` of the target, the blend is within `ε`.

This is what makes the Taylor patch bound usable: the remainder estimate holds on each
patch separately, and the partition of unity carries it to the whole domain with no
constant lost -- unlike a scattered-data interpolant, whose global error involves a
Lebesgue constant that grows with the design. -/
theorem pu_error_bound {ι : Type*} (s : Finset ι) (w Q : ι → ℝ) (f ε : ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hsum : ∑ i ∈ s, w i = 1)
    (hloc : ∀ i ∈ s, |f - Q i| ≤ ε) :
    |f - ∑ i ∈ s, w i * Q i| ≤ ε := by
  have key : ∑ i ∈ s, w i * (f - Q i) = f - ∑ i ∈ s, w i * Q i := by
    simp only [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
  rw [← key]
  calc |∑ i ∈ s, w i * (f - Q i)| ≤ ∑ i ∈ s, |w i * (f - Q i)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i ∈ s, w i * |f - Q i| := by
        refine Finset.sum_congr rfl fun i hi => ?_
        rw [abs_mul, abs_of_nonneg (hw i hi)]
    _ ≤ ∑ i ∈ s, w i * ε :=
        Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hloc i hi) (hw i hi)
    _ = ε := by rw [← Finset.sum_mul, hsum, one_mul]

/-! ## 3. The interpolation floor -/

/-- **An interpolant inherits its observations' noise exactly.**  If a reconstruction
reproduces the observed values `y i + ξ i` at the observation sites, then its squared
error there is `ξ i ^ 2` -- identically, for every design and every budget.

Consequently a rendered surface produced by interpolating a grid of noisy loss estimates
has an error floor set by the per-point Monte-Carlo noise, and refining the grid without
increasing the per-point sample size cannot lower it. -/
theorem interpolation_floor {ι : Type*} (s : Finset ι) (R y ξ : ι → ℝ)
    (hR : ∀ i ∈ s, R i = y i + ξ i) :
    ∑ i ∈ s, (R i - y i) ^ 2 = ∑ i ∈ s, ξ i ^ 2 := by
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [hR i hi]
  ring

/-! ## 4. The variance-corrected certificate -/

open MeasureTheory

/-- **The certificate is unbiased.**  Let `e` be the (deterministic) error of the
reconstruction at a probe point and `ξ` the zero-mean noise of the probe, with
`∫ ξ² = v`.  Then the variance-corrected squared residual `(e + ξ)² - v` has expectation
exactly `e²`.

Without the correction the estimator would report `e² + v` and could never certify an
error below the probe's own noise floor -- which is precisely the regime a well-allocated
design operates in. -/
theorem debias_unbiased {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (ξ : Ω → ℝ) (e v : ℝ)
    (hint : Integrable ξ μ) (hsq : Integrable (fun ω => ξ ω ^ 2) μ)
    (hmean : ∫ ω, ξ ω ∂μ = 0) (hvar : ∫ ω, ξ ω ^ 2 ∂μ = v) :
    ∫ ω, ((e + ξ ω) ^ 2 - v) ∂μ = e ^ 2 := by
  have hrw : (fun ω => (e + ξ ω) ^ 2 - v)
      = fun ω => (e ^ 2 - v) + (2 * e * ξ ω + ξ ω ^ 2) := by
    funext ω; ring
  rw [hrw, integral_add (integrable_const _) ((hint.const_mul (2 * e)).add hsq),
      integral_add (hint.const_mul (2 * e)) hsq, integral_const, integral_const_mul,
      hmean, hvar]
  simp

/-! ## 5. Why the optimal affine plane is centred at the mean -/

/-- **Sum of squared distances decomposes about the mean.**  For any point `w`,

`∑ ‖v i - w‖² = ∑ ‖v i - m‖² + n ‖m - w‖²`,  where `m` is the mean of the `v i`.

Applied to `v i = (I - P)(θ i)`, the residual of a trajectory to an affine plane with a
*fixed* direction subspace is minimised by placing the plane's origin at the mean of the
trajectory.  Anchoring at `θ₀` instead pays the extra term `n ‖m - θ₀‖²` (projected off
the plane), which is why the mean-centred construction is the Eckart--Young optimum among
affine planes and the `θ₀`-anchored one is not. -/
theorem sum_dist_sq_center {ι : Type*} [Fintype ι] {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (v : ι → E) (w : E)
    (m : E) (hm : (Fintype.card ι : ℝ) • m = ∑ j, v j) :
    ∑ i, ‖v i - w‖ ^ 2
      = (∑ i, ‖v i - m‖ ^ 2) + (Fintype.card ι : ℝ) * ‖m - w‖ ^ 2 := by
  have expand : ∀ i, ‖v i - w‖ ^ 2
      = ‖v i - m‖ ^ 2 + 2 * inner (𝕜 := ℝ) (v i - m) (m - w) + ‖m - w‖ ^ 2 := by
    intro i
    have : v i - w = (v i - m) + (m - w) := by abel
    rw [this, norm_add_sq_real]
  simp only [expand]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul]
  have hzero : ∑ i, inner (𝕜 := ℝ) (v i - m) (m - w) = 0 := by
    rw [← Finset.sum_inner]
    have : ∑ i, (v i - m) = (0 : E) := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← hm]
      abel
    rw [this, inner_zero_left]
  rw [← Finset.mul_sum, hzero]
  ring

/-! ## 6. The Taylor patch remainder, instantiated -/

/-- **Second-order patch error.**  Along the segment from an anchor `x₀` to a query point
`x`, if the third derivative of the restricted loss is bounded by `M₃` then the
second-order Taylor model errs by at most `M₃ |x - x₀|³ / 6`.

This is `taylor_mean_remainder_lagrange` at `n = 2`, with `3! = 6`; combined with
`pu_error_bound` it gives the `O(h³)` approximation term that drives the allocation. -/
theorem taylor2_patch_bound {f : ℝ → ℝ} {x₀ x M₃ : ℝ} (hx : x₀ < x)
    (hf : ContDiffOn ℝ 2 f (Set.Icc x₀ x))
    (hf' : ∀ y ∈ Set.Ioo x₀ x,
      HasDerivAt (iteratedDerivWithin 2 f (Set.Icc x₀ x))
        (iteratedDerivWithin 3 f (Set.Icc x₀ x) y) y)
    (hM : ∀ y ∈ Set.Ioo x₀ x, |iteratedDerivWithin 3 f (Set.Icc x₀ x) y| ≤ M₃) :
    |f x - taylorWithinEval f 2 (Set.Icc x₀ x) x₀ x| ≤ M₃ * (x - x₀) ^ 3 / 6 := by
  obtain ⟨c, hc, hEq⟩ := taylor_mean_remainder_lagrange hx hf hf'
  rw [hEq]
  have hxpos : (0 : ℝ) ≤ x - x₀ := by linarith
  have h6 : ((2 + 1)! : ℝ) = 6 := by norm_num [Nat.factorial]
  rw [h6]
  rw [abs_div, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 6), abs_mul,
      abs_of_nonneg (by positivity : (0:ℝ) ≤ (x - x₀) ^ (2 + 1))]
  have := hM c hc
  have hpow : (0 : ℝ) ≤ (x - x₀) ^ (2 + 1) := by positivity
  gcongr
  norm_num

end Stam
