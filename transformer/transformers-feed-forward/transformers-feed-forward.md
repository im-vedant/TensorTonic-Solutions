## The Missing Piece: Non-Linearity

Attention is a powerful mechanism, but it has a fundamental limitation: it is **linear in the values**. The attention output for each query is a weighted sum of value vectors:

$$
\text{output}_i = \sum_j \alpha_{ij} \mathbf{v}_j
$$

No matter how sophisticated the attention weights $\alpha_{ij}$ are, the output is always a convex combination of the input values. This means attention alone cannot compute non-linear functions of the input.

Consider a simple example: attention cannot compute the product of two input features, or the maximum of two values, or any function that requires non-linear interaction between features. For the Transformer to be a universal function approximator, it needs a non-linear component.

The position-wise feed-forward network (FFN) provides this essential non-linearity. Together, attention and FFN form a complete computational unit: attention handles token-to-token interaction, and FFN handles feature-to-feature transformation.

---

## The Feed-Forward Network Formula

The FFN in the original Transformer is a simple two-layer neural network with a ReLU activation:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Breaking this down:

- $x \in \mathbb{R}^{d_{model}}$ is the input vector for one token
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ is the first weight matrix (expansion)
- $b_1 \in \mathbb{R}^{d_{ff}}$ is the first bias
- $\max(0, \cdot)$ is the ReLU activation function
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ is the second weight matrix (projection)
- $b_2 \in \mathbb{R}^{d_{model}}$ is the second bias

The computation flows through three stages: **expand**, **activate**, **project**.

---

## The Three Stages

**Stage 1: Expand**

$$
h = xW_1 + b_1
$$

The input vector of dimension $d_{model}$ is projected to a higher-dimensional space of dimension $d_{ff}$. In the original Transformer, $d_{ff} = 4 \times d_{model}$, so a 512-dimensional input becomes 2048-dimensional.

This expansion creates a richer representation with more "room" for complex feature interactions.

**Stage 2: Activate (ReLU)**

$$
h' = \max(0, h)
$$

The ReLU activation applies an element-wise non-linearity: positive values pass through unchanged, negative values are set to zero.

This is the critical step that gives the FFN its power. Without this non-linearity, the two linear transformations would collapse into a single linear transformation ($xW_1W_2$), adding no expressiveness.

**Stage 3: Project back**

$$
\text{output} = h'W_2 + b_2
$$

The activated hidden representation is projected back to the original dimension $d_{model}$, so the output has the same shape as the input.

---

## Why "Position-Wise"?

The FFN is called "position-wise" because it is applied **independently and identically** to each token position in the sequence:

$$
\text{FFN}(x_1), \text{FFN}(x_2), \ldots, \text{FFN}(x_n)
$$

There is no information exchange between positions within the FFN. The same weights $W_1, b_1, W_2, b_2$ are shared across all positions, but each token's representation is transformed independently.

**Why no cross-position interaction?**

This is by design. In the Transformer architecture, the division of labor is clear:

- **Attention** handles cross-position interaction (letting tokens communicate)
- **FFN** handles within-position transformation (processing each token's features)

By separating these two functions, the architecture is clean and modular. The attention layer determines *what information to gather from other tokens*, and the FFN determines *how to process that information*.

**Equivalence to a $1 \times 1$ convolution:**

In convolutional network terms, the position-wise FFN is equivalent to two $1 \times 1$ convolutions applied to each position. This perspective makes clear that the FFN transforms features at each position without using any spatial (positional) context.

---

## The Expansion Ratio

The ratio $d_{ff} / d_{model}$ is typically 4 in the original Transformer:

- $d_{model} = 512 \rightarrow d_{ff} = 2048$
- $d_{model} = 768 \rightarrow d_{ff} = 3072$ (BERT-Base)
- $d_{model} = 1024 \rightarrow d_{ff} = 4096$ (Transformer Big)

**Why expand to 4x?**

The expansion creates a wider hidden layer that can represent more complex functions:

- With $d_{ff} = d_{model}$ (no expansion), the FFN is a bottleneck: it cannot represent functions more complex than a single linear transformation with non-linearity
- With $d_{ff} = 4 \times d_{model}$, the network has four times as many neurons in the hidden layer, enabling it to learn richer feature transformations

**The expand-then-compress pattern** is common across deep learning:

- In ResNets, the bottleneck block expands to $4 \times$ channels and compresses back
- In MobileNets, inverted residuals expand to $6 \times$ and compress back
- The principle is the same: expand into a high-dimensional space where complex transformations are easier, then compress back to the working dimension

**Why not expand more?**

Increasing $d_{ff}$ increases both parameters and computation. The FFN is already the most parameter-heavy component of a Transformer layer:

$$
\text{FFN parameters} = d_{model} \times d_{ff} + d_{ff} + d_{ff} \times d_{model} + d_{model} \approx 2 \times d_{model} \times d_{ff}
$$

For the original Transformer: $2 \times 512 \times 2048 \approx 2.1$ million parameters per FFN. With 6 encoder layers, that is 12.6 million parameters just for the FFN layers.

---

## Worked Example

Consider $d_{model} = 3$ and $d_{ff} = 6$.

**Input:** $x = [1.0, -0.5, 2.0]$

**Weights (simplified):**

$$
W_1 = \begin{pmatrix} 0.5 & -0.3 & 0.1 & 0.8 & -0.2 & 0.4 \\ 0.2 & 0.6 & -0.4 & 0.1 & 0.3 & -0.5 \\ -0.1 & 0.4 & 0.7 & -0.3 & 0.5 & 0.2 \end{pmatrix}
$$

$b_1 = [0.1, 0, 0, 0, 0, 0]$

**Step 1: Linear projection** ($xW_1 + b_1$)

$$
h = [1.0, -0.5, 2.0] \cdot W_1 + b_1
$$

Computing each element:

- $h_1 = 1.0(0.5) + (-0.5)(0.2) + 2.0(-0.1) + 0.1 = 0.5 - 0.1 - 0.2 + 0.1 = 0.3$
- $h_2 = 1.0(-0.3) + (-0.5)(0.6) + 2.0(0.4) + 0 = -0.3 - 0.3 + 0.8 = 0.2$
- $h_3 = 1.0(0.1) + (-0.5)(-0.4) + 2.0(0.7) + 0 = 0.1 + 0.2 + 1.4 = 1.7$
- $h_4 = 1.0(0.8) + (-0.5)(0.1) + 2.0(-0.3) + 0 = 0.8 - 0.05 - 0.6 = 0.15$
- $h_5 = 1.0(-0.2) + (-0.5)(0.3) + 2.0(0.5) + 0 = -0.2 - 0.15 + 1.0 = 0.65$
- $h_6 = 1.0(0.4) + (-0.5)(-0.5) + 2.0(0.2) + 0 = 0.4 + 0.25 + 0.4 = 1.05$

$h = [0.3, 0.2, 1.7, 0.15, 0.65, 1.05]$

**Step 2: ReLU**

$$
h' = \max(0, h) = [0.3, 0.2, 1.7, 0.15, 0.65, 1.05]
$$

All values are positive, so ReLU does not change anything in this example. If any were negative, they would become zero.

**Step 3: Project back**

Multiply by $W_2$ (shape $6 \times 3$) and add $b_2$ to get back to dimension 3.

The output has the same shape as the input: $[y_1, y_2, y_3]$, a 3-dimensional vector ready for the next layer.

---

## ReLU: The Activation Function

The original Transformer uses ReLU (Rectified Linear Unit):

$$
\text{ReLU}(x) = \max(0, x)
$$

**Properties:**

- Passes positive values through unchanged
- Sets negative values to zero
- Introduces sparsity: on average, about half the neurons are zero (inactive)
- Computationally cheap: just a comparison and a zero-assignment
- Gradient is either 0 (for negative inputs) or 1 (for positive inputs)

**The sparsity perspective:**

ReLU creates a sparse hidden representation: many of the $d_{ff}$ hidden units are zero for any given input. This sparsity means that different inputs activate different subsets of neurons, effectively partitioning the network's capacity across different input patterns.

This has been connected to the idea that **FFN layers function as key-value memories**, where each neuron acts as a "key" that activates for specific input patterns and retrieves a corresponding "value" (the corresponding column of $W_2$).

---

## Modern Activation Variants

While the original Transformer uses ReLU, modern Transformers have adopted smoother alternatives:

**GELU (Gaussian Error Linear Unit):**

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the standard Gaussian cumulative distribution function.

- Used in BERT, GPT-2, GPT-3
- Smoother than ReLU: instead of a hard cutoff at zero, GELU smoothly transitions from passing to blocking values
- Approximately: $\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$

**SwiGLU (Swish-Gated Linear Unit):**

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_3)
$$

where $\text{Swish}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.

- Used in LLaMA, PaLM, Mistral
- Introduces a gating mechanism: one linear projection controls the gate, another provides the content
- Requires a third weight matrix $W_3$, but empirically outperforms ReLU and GELU
- When using SwiGLU, $d_{ff}$ is typically adjusted to $\frac{8}{3} d_{model}$ to keep the parameter count similar

**Why the evolution?**

Research has shown that smoother activations (GELU) and gated activations (SwiGLU) produce better training dynamics and final model quality. The hard zero cutoff in ReLU creates "dead neurons" that never activate and cannot learn, while smoother activations allow small gradients even for near-zero inputs.

---

## FFN as Knowledge Storage

A fascinating finding from interpretability research is that FFN layers in trained Transformers appear to store **factual knowledge**.

**The key-value memory interpretation:**

Each neuron in the hidden layer can be viewed as a pattern detector:

- The row of $W_1$ corresponding to that neuron defines a "key" pattern
- The column of $W_2$ corresponding to that neuron defines a "value" pattern
- When the input matches the key (high dot product), the neuron activates and adds its value to the output

For example, a specific neuron might activate whenever the input represents "the capital of France" and contribute a value that pushes the output toward "Paris."

**Evidence:**

- Knocking out specific neurons in FFN layers can delete specific facts from the model's knowledge
- Editing specific neurons can change what the model "knows" (e.g., changing the capital of a country)
- The FFN parameters account for about two-thirds of a Transformer's total parameters, consistent with the idea that most of the model's "knowledge" is stored here

This interpretation highlights the complementary roles of attention and FFN: attention routes information between tokens (computation), while FFN stores and retrieves information (memory).

---

## Division of Labor: Attention vs. FFN

The alternating pattern of attention and FFN layers creates a powerful computational architecture:

**Attention layer:**

- Mixes information across positions
- Each token gathers relevant information from other tokens
- Output: contextually enriched representations
- Linear in values (no non-linear feature transformation)

**FFN layer:**

- Transforms features at each position independently
- Adds non-linear capacity
- Refines the contextual representations produced by attention
- Output: transformed representations ready for the next attention layer

**The iteration:**

Each encoder block applies attention then FFN. By stacking multiple blocks, the model iteratively:

1. Gathers information from context (attention)
2. Processes that information (FFN)
3. Gathers more information, now based on the processed representations (next attention)
4. Processes again (next FFN)

This iterative refinement is what allows deep Transformers to compute increasingly complex functions of the input.

---

## Dimensional Analysis

Tracking shapes through the FFN:

**Input:** $(B, L, d_{model})$

- $B$ = batch size
- $L$ = sequence length
- $d_{model}$ = model dimension

**After first linear:** $(B, L, d_{ff})$

- Expanded by factor of 4 in the last dimension
- $d_{ff} = 4 \times d_{model}$ typically

**After ReLU:** $(B, L, d_{ff})$

- Same shape, but approximately half the values are zero

**After second linear:** $(B, L, d_{model})$

- Back to the original dimension

**Key property:** The output has exactly the same shape as the input. This is necessary because the output must be compatible with the residual connection: $x + \text{FFN}(x)$.

---

## Parameter Count and Dominance

The FFN is the most parameter-heavy component in a Transformer layer:

**FFN parameters:**

- $W_1$: $d_{model} \times d_{ff}$ parameters
- $b_1$: $d_{ff}$ parameters
- $W_2$: $d_{ff} \times d_{model}$ parameters
- $b_2$: $d_{model}$ parameters
- Total: $\approx 2 \times d_{model} \times d_{ff} = 8 \times d_{model}^2$ (with $d_{ff} = 4d_{model}$)

**Attention parameters:**

- $W^Q, W^K, W^V, W^O$: $4 \times d_{model}^2$ parameters

**Ratio:**

$$
\frac{\text{FFN params}}{\text{Attention params}} = \frac{8 \times d_{model}^2}{4 \times d_{model}^2} = 2
$$

The FFN has roughly **twice as many parameters** as the attention layer. In a full Transformer, about two-thirds of the parameters are in FFN layers and one-third in attention layers.

This parameter distribution aligns with the knowledge storage interpretation: the model needs more capacity for storing knowledge (FFN) than for routing information (attention).

---

## Dropout in the FFN

The original Transformer applies dropout within the FFN for regularization:

$$
\text{FFN}(x) = \text{Dropout}(\max(0, xW_1 + b_1))W_2 + b_2
$$

Dropout randomly sets a fraction of the hidden activations to zero during training:

- Prevents co-adaptation of neurons (different neurons must learn independently useful features)
- Acts as a form of ensemble averaging (each training step uses a different random subset of neurons)
- Typical rate: 0.1 (10% of neurons zeroed out)
- Disabled during inference

---

## Computational Cost

**FLOPs per token:**

The FFN requires two matrix multiplications:

- $xW_1$: $d_{model} \times d_{ff}$ multiply-adds
- $h'W_2$: $d_{ff} \times d_{model}$ multiply-adds
- Total: $2 \times d_{model} \times d_{ff}$ FLOPs per token

For the original Transformer: $2 \times 512 \times 2048 = 2{,}097{,}152$ FLOPs per token per layer.

**Comparison to attention:**

Self-attention requires $O(L^2 \times d_{model})$ FLOPs, where $L$ is the sequence length. For short sequences ($L < d_{ff}$), the FFN dominates computation. For long sequences ($L > d_{ff}$), attention dominates.

In the original Transformer with $L = 512$ and $d_{ff} = 2048$: attention and FFN have roughly comparable cost. In modern long-context models with $L = 8192+$, attention becomes the bottleneck.

---

## Mixture of Experts: Scaling the FFN

A major recent development is the **Mixture of Experts (MoE)** approach, which scales the FFN capacity without proportionally increasing computation.

**The idea:**

Instead of one FFN, have $E$ expert FFNs. For each token, a learned router selects the top $k$ experts (typically $k = 1$ or $k = 2$):

$$
\text{MoE-FFN}(x) = \sum_{i \in \text{TopK}} g_i(x) \cdot \text{FFN}_i(x)
$$

where $g_i(x)$ is the gating weight for expert $i$.

**Benefits:**

- Total parameters scale with $E$ (more knowledge storage)
- Computation per token scales with $k$ (remains manageable)
- Allows models with trillions of parameters that are still fast to run

**Examples:**

- Switch Transformer: $E = 128$ experts, $k = 1$ (routes each token to exactly one expert)
- GShard: $E = 2048$ experts across multiple devices
- Mixtral: $E = 8$ experts, $k = 2$ (each token uses 2 of 8 experts)

The MoE approach fundamentally treats the FFN as a knowledge storage system and scales it by adding more "memory banks" (experts) while keeping the computational cost per token constant.
