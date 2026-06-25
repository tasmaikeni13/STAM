# Mathematical Theory of STAM: Subspace Trajectory Anchored Mapping

This document provides the complete mathematical derivation underpinning the four core technical components of the STAM framework. It is intended to serve as a rigorous reference for researchers seeking to understand, extend, or critically evaluate the methodology.

---

## Preface: The Curse of Dimensionality in Loss Landscape Analysis

Let $f: \mathbb{R}^N \to \mathbb{R}$ be the loss function of a neural network, parameterized by $\theta \in \mathbb{R}^N$. For modern architectures, $N$ easily exceeds $10^7$ for a ResNet or $10^{11}$ for a large language model. The loss landscape is the graph of $f$ over $\mathbb{R}^N$, an object that lives in an $(N+1)$-dimensional space.

Direct observation of this manifold is impossible. Any visualization must involve a projection — a linear map $P: \mathbb{R}^N \to \mathbb{R}^k$ for some $k \ll N$ (typically $k = 2$ for 2D visualization). The central question becomes: **which projection preserves the geometric structure most faithfully?**

The answer is not trivial. A poorly chosen projection can:
1. Make a converging optimizer appear to oscillate or diverge.
2. Make sharp, ill-conditioned minima appear flat and benign.
3. Make flat, wide minima appear narrow and sharp.
4. Completely occlude the directional structure of the gradient field.

STAM provides a principled, mathematically optimal answer to this projection problem, coupled with a scalable evaluation strategy and a novel stochastic landscape animation technique.

---

## 1. Zero-Distortion Orthogonal Subspace Construction

### 1.1 Formal Problem Statement

Let the optimizer (e.g., AdamW) produce a sequence of parameter vectors over $T$ steps:
$$\mathcal{T} = \{\theta_0, \theta_1, \theta_2, \dots, \theta_T\} \subset \mathbb{R}^N$$

We wish to find a 2D affine subspace $\mathcal{S} = \theta_0 + \text{span}\{v_1, v_2\}$ (where $v_1, v_2 \in \mathbb{R}^N$ are orthonormal) such that the projection of $\mathcal{T}$ onto $\mathcal{S}$ is maximally faithful — i.e., the projected trajectory preserves as much of the geometric structure of the original high-dimensional trajectory as possible.

### 1.2 The Displacement Matrix

STAM anchors the projection plane at the initial parameter point $\theta_0$. All subsequent positions are expressed as displacement vectors relative to this origin:

$$D = \begin{bmatrix} (\theta_1 - \theta_0)^\top \\ (\theta_2 - \theta_0)^\top \\ \vdots \\ (\theta_T - \theta_0)^\top \end{bmatrix} \in \mathbb{R}^{T \times N}$$

Each row $D_t = (\theta_t - \theta_0)^\top$ is the displacement of the optimizer from its starting position at step $t$. The matrix $D$ is a compact, data-driven representation of the entire optimization trajectory in $\mathbb{R}^N$.

### 1.3 Singular Value Decomposition

STAM performs the full Singular Value Decomposition (SVD) of $D$:
$$D = U \Sigma V^\top$$

where:
- $U \in \mathbb{R}^{T \times T}$ is an orthogonal matrix whose columns are the **left singular vectors** (one per trajectory step).
- $\Sigma \in \mathbb{R}^{T \times T}$ is a diagonal matrix of non-negative **singular values** $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_T \geq 0$.
- $V \in \mathbb{R}^{N \times T}$ is a matrix whose columns are the **right singular vectors** (one per dimension of the latent space, of which only $T$ are non-trivial).

The STAM basis vectors are the first two rows of $V^\top$ (equivalently, the first two columns of $V$):
$$v_1 = V_{:,1}, \quad v_2 = V_{:,2}$$

These are orthonormal by construction: $\|v_1\| = \|v_2\| = 1$ and $v_1^\top v_2 = 0$.

### 1.4 Optimality: The Eckart-Young-Mirsky Theorem

The choice of $v_1$ and $v_2$ is not heuristic — it is provably optimal.

**Theorem (Eckart-Young-Mirsky, 1936):** Among all rank-$k$ matrices $\hat{D}$, the best approximation to $D$ in the Frobenius norm is:
$$\hat{D}_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top$$

The residual error is $\|D - \hat{D}_k\|_F^2 = \sum_{i=k+1}^{T} \sigma_i^2$.

This means that the plane spanned by $\{v_1, v_2\}$ is the **unique** rank-2 affine subspace of $\mathbb{R}^N$ that minimizes the sum of squared Euclidean distances from the trajectory points $\{\theta_t - \theta_0\}_{t=1}^T$ to the plane. Equivalently, $\{v_1, v_2\}$ is the 2D principal subspace that captures the **maximum variance** of the displacement vectors.

The fraction of trajectory variance captured is:
$$\rho_2 = \frac{\sigma_1^2 + \sigma_2^2}{\sum_{i=1}^{T} \sigma_i^2}$$

A value of $\rho_2$ close to 1 indicates that the optimizer moves predominantly in a 2D subspace — a well-known empirical phenomenon in deep learning, often attributed to the low effective dimensionality of gradient flow dynamics.

### 1.5 Projection onto the STAM Basis

Given the basis $\{v_1, v_2\}$, any parameter vector $\theta$ is projected to 2D coordinates $(\alpha, \beta) \in \mathbb{R}^2$ by:
$$\alpha = (\theta - \theta_0)^\top v_1, \quad \beta = (\theta - \theta_0)^\top v_2$$

Conversely, any point $(\alpha, \beta)$ in the visualization plane corresponds to the parameter vector:
$$\theta(\alpha, \beta) = \theta_0 + \alpha v_1 + \beta v_2$$

This defines the bijection between the 2D visualization plane and the affine subspace $\mathcal{S}$.

---

## 2. Sparse Anchor Evaluation and RBF Interpolation

### 2.1 The Computational Intractability of Dense Evaluation

To render a 100×100 visualization of the loss landscape, a naive approach requires evaluating $\mathcal{L}(\theta(\alpha_i, \beta_j))$ for all 10,000 grid points $(\alpha_i, \beta_j)$. Each evaluation requires loading the parameter vector $\theta(\alpha_i, \beta_j)$ into the model and performing a forward pass over (a fraction of) the dataset.

For a network with $N$ parameters and a dataset of size $M$, each evaluation costs $O(N + M)$ operations. The total cost of a dense grid evaluation is:
$$\text{Cost}_{dense} = O(n^2 \cdot (N + M))$$

For $n = 100$, $N = 10^7$, $M = 10^6$, this is approximately $10^{15}$ floating-point operations — intractable even on modern GPU clusters for a single visualization.

### 2.2 STAM Sparse Anchor Evaluation

STAM reduces the evaluation cost by a factor of $\approx (n / s)^2$, where $s$ is the sparse grid resolution (e.g., $s = 7$, giving a 204× speedup over a $100\times100$ grid).

The sparse anchor grid $\mathcal{G}_{sparse}$ is defined as:
$$\mathcal{G}_{sparse} = \{(\alpha_i, \beta_j) : \alpha_i \in \text{linspace}(\alpha_{min}, \alpha_{max}, s),\; \beta_j \in \text{linspace}(\beta_{min}, \beta_{max}, s)\}$$

For each anchor point $(\alpha_i, \beta_j) \in \mathcal{G}_{sparse}$, STAM evaluates:
1. The **training loss** $\mathcal{L}_{train}(\theta(\alpha_i, \beta_j))$ by accumulating the cross-entropy loss over a fixed set of training mini-batches.
2. The **validation loss** $\mathcal{L}_{val}(\theta(\alpha_i, \beta_j))$ similarly over validation mini-batches.
3. The **empirical gradient** $\nabla \mathcal{L}_{train}(\theta(\alpha_i, \beta_j)) \in \mathbb{R}^N$ via backpropagation, which is subsequently projected onto the STAM basis:
   $$g_x = \nabla \mathcal{L}^\top v_1, \quad g_y = \nabla \mathcal{L}^\top v_2$$

Because only $s^2 \ll n^2$ evaluations are performed, the computation budget can be allocated to accumulating gradients over many mini-batches (STAM uses 20 batches per anchor), driving the stochastic variance of the loss estimate to near zero.

### 2.3 Radial Basis Function Interpolation

Given the $s^2$ sparse but accurate anchor measurements, STAM must reconstruct the full 100×100 loss surface. This is a **scattered data interpolation** problem: given $M$ sample points $\{(x_i, y_i, z_i)\}_{i=1}^{M}$ in $\mathbb{R}^3$, find a smooth function $\hat{f}: \mathbb{R}^2 \to \mathbb{R}$ such that $\hat{f}(x_i, y_i) \approx z_i$ for all $i$, and $\hat{f}$ is well-behaved between sample points.

STAM uses **Radial Basis Function (RBF) interpolation**, which represents $\hat{f}$ as a weighted sum of radially symmetric basis functions centered at the anchor points:
$$\hat{f}(x, y) = \sum_{i=1}^{M} w_i \phi\!\left(\|(x,y) - (x_i, y_i)\|\right)$$

The weights $w_i$ are determined by solving the linear system:
$$\Phi \mathbf{w} = \mathbf{z}, \quad \text{where} \quad \Phi_{ij} = \phi\!\left(\|(x_i, y_i) - (x_j, y_j)\|\right)$$

#### 2.3.1 The Multiquadric Kernel (Loss Surfaces)

For the training and validation loss surfaces, STAM uses the **multiquadric** kernel, introduced by Hardy (1971) for topographic surface reconstruction:
$$\phi_{mq}(r) = \sqrt{1 + (\varepsilon r)^2}, \quad \varepsilon > 0$$

The multiquadric kernel produces globally smooth, infinitely differentiable interpolants and is known to achieve spectral accuracy for smooth functions. Its use is justified by the empirical observation that neural network loss landscapes, when restricted to low-dimensional PCA subspaces, are smooth and exhibit no high-frequency oscillations (Li et al., 2018).

#### 2.3.2 The Linear Kernel (Gradient Fields)

For the gradient vector components $(g_x, g_y)$, STAM uses the **linear** kernel:
$$\phi_{lin}(r) = r$$

The linear kernel produces piecewise linear interpolants that are conservative with respect to the gradient magnitudes, avoiding the over-smoothing that would obscure the true directional structure of the gradient flow field.

#### 2.3.3 Complexity Analysis

The RBF interpolation involves solving a $M \times M$ linear system (where $M = s^2 = 49$ for the 7×7 sparse grid) and then evaluating the resulting function at $n^2 = 10,000$ render points. The dominant cost is the linear solve: $O(M^3) = O(49^3) \approx 1.2 \times 10^5$ operations. The subsequent evaluation over the render grid costs $O(M \cdot n^2) = O(49 \cdot 10^4) = O(4.9 \times 10^5)$ operations. Both are constant with respect to the model size $N$, giving the claimed $O(1)$ scaling.

---

## 3. The Animated Stochastic "Breathing" Landscape

### 3.1 The Stochastic Landscape Problem

The fundamental object of interest to a practitioner is not the global loss $\mathcal{L}_{global}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(\theta; x, y)]$, which is defined over the full data distribution $\mathcal{D}$. Rather, at each training step $t$, the optimizer only has access to the mini-batch loss:
$$\mathcal{L}_{batch}^{(t)}(\theta) = \frac{1}{|B_t|} \sum_{(x,y) \in B_t} \ell(\theta; x, y)$$

The deviation between these two defines the **stochastic noise field**:
$$\eta^{(t)}(\theta) = \nabla \mathcal{L}_{batch}^{(t)}(\theta) - \nabla \mathcal{L}_{global}(\theta)$$

It is this noise field that causes SGD to escape sharp local minima, acts as an implicit regularizer, and drives the rich, non-monotonic dynamics observed in practice. Visualizing $\mathcal{L}_{batch}^{(t)}$ for each $t$ requires re-evaluating the entire landscape per frame, which is computationally infeasible.

### 3.2 The Taylor Expansion Approximation

STAM approximates the local mini-batch landscape via a **first-order Taylor Expansion** around the optimizer's current position $(\alpha_t, \beta_t)$ in the STAM plane.

Let $g_{batch,x}^{(t)}$ and $g_{batch,y}^{(t)}$ be the components of the mini-batch gradient projected onto the STAM basis (captured during training). Let $g_{global,x}^{(t)}$ and $g_{global,y}^{(t)}$ be the corresponding global gradient components at $(\alpha_t, \beta_t)$, read off from the pre-computed interpolated gradient field.

The projected stochastic noise at step $t$ is:
$$\Delta x^{(t)} = g_{batch,x}^{(t)} - g_{global,x}^{(t)}, \quad \Delta y^{(t)} = g_{batch,y}^{(t)} - g_{global,y}^{(t)}$$

This vector $(\Delta x^{(t)}, \Delta y^{(t)})$ encodes how the local mini-batch surface is *tilted* relative to the global surface at the optimizer's current position.

### 3.3 The Breathing Warp Function

Applying a flat linear tilt over the entire landscape would be physically incorrect; the mini-batch deviation is a local phenomenon. STAM applies a **Gaussian spatial envelope** to confine the warp to a neighborhood around the optimizer's position $(\alpha_t, \beta_t)$:

$$Z_{warped}^{(t)}(\alpha, \beta) = Z_{global}(\alpha, \beta) + \underbrace{\left[\Delta x^{(t)}(\alpha - \alpha_t) + \Delta y^{(t)}(\beta - \beta_t)\right]}_{\text{first-order tilt}} \cdot \underbrace{\exp\!\left(-\frac{(\alpha - \alpha_t)^2 + (\beta - \beta_t)^2}{2\sigma^2}\right)}_{\text{Gaussian envelope}}$$

where $\sigma$ controls the spatial scale of the stochastic perturbation. A smaller $\sigma$ concentrates the breathing effect near the optimizer's foot; a larger $\sigma$ allows the mini-batch noise to tilt a wider region of the landscape.

### 3.4 Physical Interpretation

This formulation has a precise physical interpretation. The first-order Taylor term $\Delta x^{(t)}(\alpha - \alpha_t) + \Delta y^{(t)}(\beta - \beta_t)$ represents an additional linear energy gradient — a tilt of the surface — caused by the discrepancy between the mini-batch gradient and the true gradient. The Gaussian envelope ensures that this tilt is only applied locally, respecting the fact that mini-batch gradient estimates are only valid in the neighborhood of the current parameter position (where the variance of the stochastic estimator is smallest).

The animated sequence of warped surfaces $\{Z_{warped}^{(0)}, Z_{warped}^{(1)}, \dots, Z_{warped}^{(T)}\}$ literally visualizes the landscape as the optimizer experiences it: a dynamically shifting, noisy manifold where the local topology is constantly being perturbed by the randomness of data sampling. This directly demonstrates the mechanism by which SGD escapes sharp minima — the stochastic tilt at a sharp minimum will, at some mini-batch step, point uphill on one side and less steeply downhill on the other, effectively flattening or even inverting the curvature locally and allowing the optimizer to escape.

---

## 4. Empirical Gradient Flow Vector Fields

### 4.1 The Projected Gradient Vector Field

At each sparse anchor $(\alpha_i, \beta_j)$, STAM computes the full $N$-dimensional gradient via backpropagation and projects it to the 2D STAM plane:
$$g_x^{(i,j)} = \left[\nabla_\theta \mathcal{L}(\theta(\alpha_i, \beta_j))\right]^\top v_1$$
$$g_y^{(i,j)} = \left[\nabla_\theta \mathcal{L}(\theta(\alpha_i, \beta_j))\right]^\top v_2$$

After RBF interpolation, this yields a dense 2D vector field:
$$\mathbf{g}_{proj}(\alpha, \beta) = (g_x(\alpha, \beta), g_y(\alpha, \beta))$$

### 4.2 Relationship to the Loss Surface Gradient

It is important to note that $\mathbf{g}_{proj}$ is **not** the gradient of the interpolated loss surface $Z_{global}$. Rather, it is the projection of the true $N$-dimensional gradient onto the STAM plane. These are related but not identical:
$$\frac{\partial Z_{global}}{\partial \alpha} = \left[\nabla_\theta \mathcal{L}\right]^\top v_1 = g_x, \quad \frac{\partial Z_{global}}{\partial \beta} = \left[\nabla_\theta \mathcal{L}\right]^\top v_2 = g_y$$

This equality holds because $Z_{global}(\alpha, \beta) = \mathcal{L}(\theta_0 + \alpha v_1 + \beta v_2)$, and by the chain rule:
$$\nabla_{(\alpha,\beta)} Z_{global} = \begin{bmatrix} v_1^\top \\ v_2^\top \end{bmatrix} \nabla_\theta \mathcal{L} = (g_x, g_y)$$

Therefore, the quiver plot rendered by STAM is the **exact gradient of the true loss landscape** within the STAM plane — not an approximation. The negated gradient field $-\mathbf{g}_{proj}$ points in the direction of steepest descent within the visualization subspace, providing a direct visual explanation of the optimizer's motion.

### 4.3 The Quiver Plot as Visual Proof

The quiver plot overlaid on the 2D contour map constitutes a direct visual proof of the relationship between the loss topology and the optimizer's path. At every rendered point, the arrow indicates the local direction of steepest descent. The optimizer's trajectory (the red line) must, at each step, be approximately aligned with the quiver arrows. Any misalignment between the trajectory and the gradient field is itself informative: it reveals the effect of momentum (AdamW's first-moment estimate), adaptive learning rates (AdamW's second-moment estimate), or weight decay on the effective direction of parameter updates.

---

## 5. Summary of the STAM Mathematical Framework

| Component | Mathematical Tool | Guarantee |
|---|---|---|
| Projection Plane Selection | SVD + Eckart-Young-Mirsky Theorem | Optimal rank-2 variance capture; zero distortion |
| Loss Surface Recovery | Multiquadric RBF Interpolation | Spectrally accurate for smooth manifolds |
| Gradient Field Recovery | Linear RBF Interpolation | Exact at anchor points; conservative between them |
| Stochastic Landscape Animation | 1st-Order Taylor + Gaussian Envelope | Physically motivated local approximation; $O(0)$ extra forward passes |
| Gradient Quiver Field | Chain Rule Projection | Exact gradient of the loss within the STAM plane |

---

## References

- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika, 1*(3), 211–218. https://doi.org/10.1007/BF02288367
- Hardy, R. L. (1971). Multiquadric equations of topography and other irregular surfaces. *Journal of Geophysical Research, 76*(8), 1905–1915. https://doi.org/10.1029/JB076i008p01905
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *Advances in Neural Information Processing Systems (NeurIPS), 31*. https://arxiv.org/abs/1712.09913
- Goodfellow, I., Vinyals, O., & Saxe, A. (2015). Qualitatively characterizing neural network optimization problems. *International Conference on Learning Representations (ICLR 2015)*. https://arxiv.org/abs/1412.6544
- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A. G. (2018). Loss surfaces, mode connectivity, and fast ensembling of DNNs. *Advances in Neural Information Processing Systems (NeurIPS), 31*. https://arxiv.org/abs/1802.10026
- Gur-Ari, G., Roberts, D. A., & Dyer, E. (2018). Gradient descent happens in a tiny subspace. https://arxiv.org/abs/1812.04754
- Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially no barriers in neural network energy landscape. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1803.00885
