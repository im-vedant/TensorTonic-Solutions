## The Problem: Internal Covariate Shift

Deep neural networks consist of many layers stacked on top of each other. Each layer receives input from the layer below and produces output for the layer above. As training progresses and weights update, the distribution of inputs to each layer changes constantly. This phenomenon is called **internal covariate shift**.

The problem is practical: if the inputs to a layer suddenly shift in scale or center, the layer's weights, which were tuned for the old distribution, become suboptimal. The network must constantly readjust, slowing down training.

**Example of the problem:**

Suppose a layer receives inputs that initially have mean 0 and standard deviation 1. After a weight update in an earlier layer, the inputs might shift to mean 5 and standard deviation 10. The current layer's weights, which were calibrated for the original distribution, will now produce wildly different outputs, potentially causing instability.

Normalization techniques address this by standardizing inputs to have a consistent distribution before each layer processes them.

---

## What Layer Normalization Does

Layer normalization transforms the activations across the **feature dimension** so that, for each individual sample and each individual position, the features have zero mean and unit variance:

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

- $x \in \mathbb{R}^{d_{model}}$ is the input vector for one token at one position
- $\mu$ is the mean of all $d_{model}$ components of $x$
- $\sigma^2$ is the variance of all $d_{model}$ components of $x$
- $\gamma \in \mathbb{R}^{d_{model}}$ is a learned scale parameter (initialized to 1)
- $\beta \in \mathbb{R}^{d_{model}}$ is a learned shift parameter (initialized to 0)
- $\epsilon$ is a small constant for numerical stability (typically $10^{-6}$)

---

## Computing Mean and Variance

For a single input vector $x = [x_1, x_2, \ldots, x_d]$ where $d = d_{model}$:

**Mean:**

$$
\mu = \frac{1}{d} \sum_{j=1}^{d} x_j
$$

**Variance:**

$$
\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2
$$

**Normalization:**

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad \text{for each } j = 1, \ldots, d
$$

After normalization, the $\hat{x}$ vector has approximately zero mean and unit variance across its components.

The crucial point is that normalization happens across the **feature dimension** (across the $d_{model}$ components), not across the batch or across sequence positions. Each token is normalized independently.

---

## Worked Example

Consider a vector $x = [2, 4, 6, 8]$ with $d_{model} = 4$, $\gamma = [1, 1, 1, 1]$, $\beta = [0, 0, 0, 0]$, $\epsilon = 10^{-6}$.

**Step 1: Compute mean**

$$
\mu = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5
$$

**Step 2: Compute variance**

$$
\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5
$$

**Step 3: Normalize**

$$
\hat{x}_1 = \frac{2 - 5}{\sqrt{5 + 10^{-6}}} = \frac{-3}{2.236} \approx -1.342
$$

$$
\hat{x}_2 = \frac{4 - 5}{\sqrt{5}} = \frac{-1}{2.236} \approx -0.447
$$

$$
\hat{x}_3 = \frac{6 - 5}{\sqrt{5}} = \frac{1}{2.236} \approx 0.447
$$

$$
\hat{x}_4 = \frac{8 - 5}{\sqrt{5}} = \frac{3}{2.236} \approx 1.342
$$

**Result:** $[-1.342, -0.447, 0.447, 1.342]$

**Verification:**
- Mean: $(-1.342 - 0.447 + 0.447 + 1.342)/4 = 0$ (zero mean)
- Variance: approximately 1 (unit variance)

---

## Why Learnable $\gamma$ and $\beta$?

After normalization, all vectors have mean 0 and variance 1. But this might not be the best representation for every layer. What if the optimal input distribution for a downstream layer actually has mean 2 and variance 3?

The learnable parameters $\gamma$ and $\beta$ allow the network to undo the normalization if that is beneficial:

$$
y_j = \gamma_j \hat{x}_j + \beta_j
$$

- $\gamma_j$ controls the scale of dimension $j$
- $\beta_j$ controls the shift of dimension $j$
- These are learned per-dimension, giving $2 \times d_{model}$ additional parameters

**At initialization** ($\gamma = \mathbf{1}$, $\beta = \mathbf{0}$), the affine transform is the identity, so LayerNorm simply normalizes. During training, the model learns the optimal scale and shift for each dimension.

**The identity escape hatch:**

If the optimal behavior is no normalization at all, the model can learn $\gamma_j = \sigma_j$ and $\beta_j = \mu_j$ (the original statistics), effectively undoing the normalization entirely. This means LayerNorm can never make the model worse than not having it; it only adds flexibility.

---

## Layer Normalization vs. Batch Normalization

The most important distinction in normalization techniques is **which dimension** the statistics are computed over.

**Batch Normalization (BatchNorm):**

Normalizes across the **batch dimension**. For each feature, compute the mean and variance across all samples in the batch.

$$
\mu_j^{BN} = \frac{1}{B} \sum_{b=1}^{B} x_{bj}, \quad \sigma_j^{2, BN} = \frac{1}{B} \sum_{b=1}^{B} (x_{bj} - \mu_j^{BN})^2
$$

- Statistics computed across the batch for each feature independently
- Each feature gets its own mean and variance
- Requires a sufficiently large batch for stable statistics
- During inference, uses running averages computed during training

**Layer Normalization (LayerNorm):**

Normalizes across the **feature dimension**. For each sample, compute the mean and variance across all features.

$$
\mu^{LN} = \frac{1}{d} \sum_{j=1}^{d} x_j, \quad \sigma^{2, LN} = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu^{LN})^2
$$

- Statistics computed across features for each sample independently
- Each sample gets its own mean and variance
- No dependence on batch size
- Identical behavior during training and inference

---

## Why LayerNorm for Transformers?

BatchNorm was the dominant normalization technique when the Transformer was introduced (2017), having proven transformative for convolutional neural networks. So why did the Transformer choose LayerNorm instead?

**Variable sequence lengths:**

In NLP, sequences within a batch often have different lengths, padded to the maximum length. BatchNorm would compute statistics that mix real tokens with padding tokens, introducing noise. LayerNorm normalizes each token independently, so padding is irrelevant.

**Batch size independence:**

LayerNorm produces the same output regardless of batch size. This is important because:

- Inference often uses batch size 1 (single query)
- BatchNorm at batch size 1 has undefined statistics (a single sample has no variance across the batch)
- LayerNorm works identically during training and inference

**Autoregressive generation:**

During text generation, the model produces one token at a time. There is no "batch" of tokens to compute statistics over at each step. LayerNorm handles this naturally because it only needs the features of the current token.

**Parallel across positions:**

Each token position is normalized independently, which aligns with the Transformer's parallel processing of positions. BatchNorm would introduce dependencies between positions within the batch.

---

## The Epsilon ($\epsilon$) Parameter

The small constant $\epsilon$ added inside the square root prevents division by zero:

$$
\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**When is $\epsilon$ needed?**

If all components of $x$ are identical (e.g., $x = [3, 3, 3, 3]$), then $\sigma^2 = 0$. Without $\epsilon$, we would divide by zero, producing NaN values that propagate through the network and destroy training.

**What value to use?**

- $\epsilon = 10^{-6}$ is the most common choice (used in the original Transformer)
- $\epsilon = 10^{-5}$ is used in some implementations (e.g., BERT)
- The exact value rarely matters as long as it is small enough not to distort the normalization but large enough to prevent numerical issues

**Gradient implications:**

Even when $\sigma^2$ is not exactly zero, very small variances produce very large normalized values (because we divide by a very small number). The $\epsilon$ provides a floor that prevents extreme values and keeps gradients stable.

---

## Pre-Norm vs. Post-Norm

The Transformer uses LayerNorm in a specific pattern called "post-norm":

**Post-Norm (original Transformer):**

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

LayerNorm is applied **after** the residual addition.

**Pre-Norm (GPT-2, GPT-3, LLaMA, and most modern Transformers):**

$$
\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

LayerNorm is applied **before** the sublayer, and the residual connection bypasses the normalization.

**Why pre-norm became dominant:**

- **Gradient flow**: In pre-norm, the residual connection creates a clean skip path from the input to the output. Gradients can flow through this path without passing through LayerNorm, which can dampen gradients.
- **Training stability**: Pre-norm models are significantly easier to train, especially for deep models (50+ layers). Post-norm models often require careful learning rate warmup to avoid divergence.
- **Depth scaling**: Pre-norm allows successful training of models with hundreds of layers, while post-norm struggles beyond a few dozen without careful tuning.

The original Transformer paper used post-norm, but the field has largely moved to pre-norm due to its superior training dynamics.

---

## Where LayerNorm Appears in the Transformer

In the original (post-norm) Transformer encoder, LayerNorm appears twice per block:

**After self-attention:**

$$
x_1 = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))
$$

**After the feed-forward network:**

$$
x_2 = \text{LayerNorm}(x_1 + \text{FFN}(x_1))
$$

Each LayerNorm has its own learnable $\gamma$ and $\beta$ parameters, so the two normalizations can learn different scales and shifts optimized for their respective positions in the architecture.

With $N$ encoder blocks, there are $2N$ LayerNorm operations in the encoder alone.

---

## Gradient Flow Through LayerNorm

Understanding how gradients flow through LayerNorm reveals why it stabilizes training.

**Forward pass:**

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**Backward pass (simplified):**

The Jacobian of LayerNorm is:

$$
\frac{\partial \hat{x}_j}{\partial x_k} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left(\delta_{jk} - \frac{1}{d} - \frac{\hat{x}_j \hat{x}_k}{d}\right)
$$

where $\delta_{jk}$ is the Kronecker delta (1 if $j = k$, 0 otherwise).

**Key insight:**

The gradient has three components:

- $\delta_{jk}$: the direct gradient (same as without normalization)
- $-1/d$: a correction that distributes gradient equally across all dimensions (from the mean subtraction)
- $-\hat{x}_j \hat{x}_k / d$: a correction that accounts for the variance normalization

The $1/\sqrt{\sigma^2 + \epsilon}$ scaling factor ensures that even if the input scale varies wildly, the gradients remain in a manageable range. This is the core mechanism by which LayerNorm stabilizes training.

---

## RMSNorm: A Simplification

RMSNorm (Root Mean Square Layer Normalization) is a simplified variant that has gained popularity in modern architectures like LLaMA:

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}
$$

where:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2}
$$

**Differences from LayerNorm:**

- No mean subtraction (does not re-center to zero mean)
- No $\beta$ parameter (only scale, no shift)
- Slightly simpler computation (no mean calculation)

**Why it works:**

Research has shown that the re-centering (mean subtraction) in LayerNorm is often not the critical component. The re-scaling (division by standard deviation) is what primarily stabilizes training. RMSNorm keeps only the re-scaling, reducing computational cost by about 10-15%.

---

## Normalization Across the Feature Dimension: The Geometric View

Geometrically, LayerNorm projects the input vector onto a hypersphere of radius $\sqrt{d}$ centered at the origin (after accounting for $\gamma$ and $\beta$).

**Before normalization:**

Input vectors can point anywhere in $\mathbb{R}^d$ with arbitrary magnitude. Two vectors with the same direction but different magnitudes ($[1, 2, 3]$ and $[100, 200, 300]$) would produce very different downstream activations.

**After normalization:**

Both vectors become $[-1.22, 0, 1.22]$ (the same normalized direction). The magnitude information is discarded, and only the direction (relative relationships between components) is preserved.

This is beneficial because in many cases, the relative pattern of activations (which components are large vs. small) is more informative than the absolute scale.

---

## Parameter Count and Computational Cost

**Parameters per LayerNorm:**

- $\gamma$: $d_{model}$ parameters
- $\beta$: $d_{model}$ parameters
- Total: $2 \times d_{model}$ parameters

For $d_{model} = 512$: $2 \times 512 = 1024$ parameters per LayerNorm.

With 2 LayerNorms per encoder block and 6 encoder blocks: $2 \times 6 \times 1024 = 12{,}288$ parameters. This is tiny compared to the attention and FFN parameters.

**Computational cost:**

For each token, LayerNorm requires:
- One pass to compute the mean: $d_{model}$ additions
- One pass to compute the variance: $d_{model}$ subtractions, multiplications, additions
- One pass to normalize and apply $\gamma$, $\beta$: $d_{model}$ operations

Total: $O(d_{model})$ per token. This is negligible compared to the $O(d_{model}^2)$ cost of attention and FFN layers.

---

## Numerical Stability Considerations

When implementing LayerNorm, numerical precision matters:

**Computing variance:**

The naive formula $\sigma^2 = E[x^2] - (E[x])^2$ can suffer from **catastrophic cancellation** when $E[x^2]$ and $(E[x])^2$ are very close. The two-pass formula (first compute $\mu$, then compute $\sigma^2 = E[(x-\mu)^2]$) is numerically more stable.

**Half-precision (float16):**

Modern GPUs use 16-bit floating point for speed. LayerNorm can lose precision in float16 because:

- The mean and variance calculations accumulate many small values
- The division by $\sqrt{\sigma^2 + \epsilon}$ amplifies any error

A common solution is to compute LayerNorm statistics in float32 even when the rest of the model uses float16. This is called "mixed precision" and is the standard practice in large model training.

---

## Historical Context

**Batch Normalization (2015):**

Ioffe and Szegedy introduced BatchNorm, which dramatically accelerated training of convolutional networks. However, its batch-size dependence and behavior change between training and inference were limitations.

**Layer Normalization (2016):**

Ba, Kiros, and Hinton proposed LayerNorm specifically to address these limitations. They showed it was particularly effective for RNNs and other sequence models.

**The Transformer (2017):**

Vaswani et al. adopted LayerNorm in the Transformer architecture, establishing it as the standard normalization for attention-based models.

**Pre-norm revolution (2018-2020):**

Researchers discovered that moving LayerNorm before the sublayer (pre-norm) significantly improved training stability, enabling much deeper models.

**RMSNorm (2019-present):**

Zhang and Sennrich showed that the re-centering component of LayerNorm is often unnecessary, leading to the simpler RMSNorm used in modern architectures like LLaMA.

The trajectory shows a consistent trend toward simpler, more computationally efficient normalization that preserves the core benefit of stabilizing activations.
