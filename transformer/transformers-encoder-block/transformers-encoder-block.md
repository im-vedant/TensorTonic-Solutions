## The Building Block of the Transformer

The Transformer encoder is not a single monolithic network. It is a stack of identical blocks, each performing the same sequence of operations. The original Transformer uses $N = 6$ blocks stacked on top of each other, with each block refining the representations produced by the block below.

Each encoder block takes a sequence of vectors as input and produces a sequence of vectors of the same shape as output. This uniformity of interface is what allows blocks to be stacked: the output of block 1 feeds directly into block 2, and so on.

The encoder block is where all the individual components, multi-head attention, layer normalization, feed-forward network, and residual connections, come together into a unified architecture.

---

## The Block Structure

Each encoder block contains two sub-layers, each wrapped with a residual connection and layer normalization:

**Sub-layer 1: Self-attention**

$$
x' = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))
$$

**Sub-layer 2: Feed-forward network**

$$
\text{output} = \text{LayerNorm}(x' + \text{FFN}(x'))
$$

This pattern is described as "Add & Norm": add the sub-layer's output to its input (residual connection), then apply layer normalization.

The full computation graph for one encoder block:

1. Receive input $x$ (shape: $B \times L \times d_{model}$)
2. Compute self-attention: $\text{attn} = \text{MultiHeadAttention}(x, x, x)$
3. Add residual: $x_1 = x + \text{attn}$
4. Normalize: $x' = \text{LayerNorm}_1(x_1)$
5. Compute FFN: $\text{ffn} = \text{FFN}(x')$
6. Add residual: $x_2 = x' + \text{ffn}$
7. Normalize: $\text{output} = \text{LayerNorm}_2(x_2)$
8. Return output (shape: $B \times L \times d_{model}$, same as input)

---

## Residual Connections: The Gradient Highway

The residual connections ($x + \text{Sublayer}(x)$) are not optional decorations. They are essential for training deep networks. Without them, the Transformer would not work.

**The vanishing gradient problem:**

In a deep network without residual connections, the gradient at layer $l$ depends on the product of gradients through all subsequent layers:

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \prod_{i=l+1}^{L} \frac{\partial F_i}{\partial x_{i-1}} \cdot \frac{\partial \mathcal{L}}{\partial x_L}
$$

If each gradient factor has magnitude less than 1, this product shrinks exponentially with depth. A 6-layer network with factors of 0.5 would multiply gradients by $0.5^6 = 0.016$, effectively blocking learning in early layers.

**How residual connections fix this:**

With a residual connection $x_{l+1} = x_l + F_l(x_l)$, the gradient becomes:

$$
\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F_l}{\partial x_l}
$$

The identity matrix $I$ provides a direct path for gradients to flow through, regardless of what $\frac{\partial F_l}{\partial x_l}$ does. Even if the sub-layer gradient is zero, the gradient through the residual path is exactly 1.

**Through multiple layers:**

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(I + \sum_{\text{paths}} \prod_{\text{sub-layers on path}} \frac{\partial F}{\partial x}\right)
$$

The gradient is a sum over all possible paths through the network, including the direct "skip all" path. This sum ensures that at least one path contributes a substantial gradient, preventing vanishing.

---

## The Role of Layer Normalization in the Block

Layer normalization appears after each residual addition, serving several purposes:

**Stabilizing activations:**

As residual connections accumulate, the magnitude of the representations can grow with depth. Each time we add the sub-layer output to the input, the values get larger. LayerNorm resets the scale, preventing unbounded growth.

**Enabling consistent learning rates:**

If different layers have representations at different scales, they require different effective learning rates. LayerNorm ensures all layers have similarly-scaled inputs, so a single learning rate works across the entire network.

**Smoothing the loss landscape:**

Research has shown that normalization techniques smooth the optimization landscape, reducing the number of sharp valleys and making gradient descent more reliable.

---

## Self-Attention Within the Block

The attention sub-layer in the encoder block uses **self-attention**: queries, keys, and values all come from the same input:

$$
\text{MultiHeadAttention}(x, x, x)
$$

All three arguments are the same tensor $x$. This means every token in the sequence attends to every other token in the same sequence (including itself).

**What self-attention computes:**

For each token, self-attention asks: "Given my current representation, which other tokens in this sequence are most relevant to me?"

- A verb might attend to its subject and object
- A pronoun might attend to its antecedent
- A quantifier might attend to the noun it modifies

**The output:**

After self-attention, each token's representation has been enriched with information from the tokens it attended to. The representation of "it" now contains information about "cat" if the attention weights pointed there.

---

## The Feed-Forward Network Within the Block

After self-attention and its residual-plus-norm, the FFN applies an independent transformation to each position:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**Its role in the block:**

Self-attention aggregates information across positions but is linear in the values. The FFN provides:

- **Non-linearity**: The ReLU (or GELU/SwiGLU in modern variants) enables the network to compute non-linear functions
- **Feature transformation**: Each token's enriched representation is refined and transformed
- **Knowledge retrieval**: The FFN weights encode factual knowledge that the model can "look up"

**The alternating pattern:**

The attention-then-FFN pattern implements a two-phase processing cycle:

1. **Gather** (attention): collect relevant information from context
2. **Process** (FFN): transform the gathered information

By repeating this cycle $N$ times (once per block), the model can perform increasingly sophisticated reasoning.

---

## Stacking Blocks: Depth and Abstraction

The Transformer encoder stacks $N$ identical blocks. Each block has the same architecture but its own set of learned parameters.

**What happens at different depths?**

Research on trained Transformers has revealed a hierarchy of processing:

**Early layers (blocks 1-2):**

- Build basic contextual representations
- Capture local patterns (nearby words, common phrases)
- Resolve simple ambiguities (e.g., distinguishing homographs based on immediate context)
- Attention patterns tend to be local and syntactic

**Middle layers (blocks 3-4):**

- Build richer representations that combine multiple tokens' information
- Capture medium-range dependencies (e.g., subject-verb agreement across a clause)
- Attention patterns become more diverse and task-specific

**Late layers (blocks 5-6):**

- Produce task-ready representations
- Capture global patterns and high-level semantics
- Attention patterns are often the most specialized and interpretable
- The final layer's output is used directly for downstream tasks

This hierarchical processing is analogous to how convolutional networks process images: early layers detect edges and textures, middle layers detect parts and shapes, and late layers detect objects and scenes.

---

## Worked Example: Data Flow Through One Block

Consider a simple sequence of 3 tokens with $d_{model} = 4$, $h = 2$ heads ($d_k = 2$), and $d_{ff} = 8$.

**Input** $x$:

$$
x = \begin{pmatrix} 1.0 & 0.5 & -0.3 & 0.8 \\ -0.2 & 1.2 & 0.4 & -0.6 \\ 0.7 & -0.1 & 0.9 & 0.3 \end{pmatrix}
$$

**Step 1: Multi-head self-attention**

- Project $x$ into $Q, K, V$ using learned weights
- Split into 2 heads (each operating on 2 dimensions)
- Head 1 computes attention on first 2D subspace
- Head 2 computes attention on second 2D subspace
- Concatenate head outputs
- Apply output projection $W^O$
- Result: $\text{attn}$ has shape $(3, 4)$

Suppose $\text{attn} = \begin{pmatrix} 0.1 & -0.2 & 0.05 & 0.15 \\ 0.3 & 0.1 & -0.1 & 0.2 \\ -0.05 & 0.15 & 0.2 & -0.1 \end{pmatrix}$

**Step 2: Residual connection**

$$
x_1 = x + \text{attn} = \begin{pmatrix} 1.1 & 0.3 & -0.25 & 0.95 \\ 0.1 & 1.3 & 0.3 & -0.4 \\ 0.65 & 0.05 & 1.1 & 0.2 \end{pmatrix}
$$

**Step 3: Layer normalization**

For each row, subtract mean and divide by standard deviation, then apply $\gamma$ and $\beta$.

Row 1: mean $= 0.525$, values become centered around 0 with unit variance.

Result: $x'$ has the same shape $(3, 4)$ but with normalized values.

**Step 4: Feed-forward network**

- Expand: $x' W_1 + b_1$ gives shape $(3, 8)$
- ReLU: zero out negatives, same shape $(3, 8)$
- Project back: $h' W_2 + b_2$ gives shape $(3, 4)$
- Result: $\text{ffn}$ has shape $(3, 4)$

**Step 5: Second residual connection**

$$
x_2 = x' + \text{ffn}
$$

**Step 6: Second layer normalization**

$$
\text{output} = \text{LayerNorm}_2(x_2)
$$

**Final output**: Shape $(3, 4)$, same as input. This output can feed into the next encoder block.

---

## Parameters in One Encoder Block

For the original Transformer configuration ($d_{model} = 512$, $h = 8$, $d_{ff} = 2048$):

**Multi-head attention:**

- $W^Q, W^K, W^V, W^O$: $4 \times 512 \times 512 = 1{,}048{,}576$ parameters

**Feed-forward network:**

- $W_1$: $512 \times 2048 = 1{,}048{,}576$
- $b_1$: $2048$
- $W_2$: $2048 \times 512 = 1{,}048{,}576$
- $b_2$: $512$
- Total FFN: $2{,}099{,}712$ parameters

**Layer normalization (2x):**

- $\gamma_1, \beta_1$: $2 \times 512 = 1{,}024$
- $\gamma_2, \beta_2$: $2 \times 512 = 1{,}024$
- Total LayerNorm: $2{,}048$ parameters

**Total per block:** approximately $3{,}150{,}336$ parameters ($\approx 3.15$ million)

**Total encoder (6 blocks):** approximately $18{,}900{,}000$ parameters ($\approx 19$ million)

The FFN accounts for about two-thirds of each block's parameters, with attention accounting for the remaining third. LayerNorm contributes negligibly.

---

## Dropout in the Encoder Block

The original Transformer applies dropout at several points within the encoder block:

**After attention output** (before residual addition):

$$
x' = \text{LayerNorm}(x + \text{Dropout}(\text{MultiHeadAttention}(x, x, x)))
$$

**Within attention** (on the attention weights):

$$
\alpha' = \text{Dropout}(\text{softmax}(QK^T / \sqrt{d_k}))
$$

**After FFN output** (before residual addition):

$$
\text{output} = \text{LayerNorm}(x' + \text{Dropout}(\text{FFN}(x')))
$$

**Dropout rates:**

- Base model: $P_{drop} = 0.1$
- Large model: $P_{drop} = 0.3$

Dropout on attention weights is particularly interesting: it randomly prevents certain positions from attending to others, forcing the model to develop redundant attention patterns and not over-rely on any single position.

---

## Pre-Norm vs. Post-Norm Blocks

The original Transformer uses post-norm (normalize after the residual addition):

**Post-norm:**

$$
x' = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

Most modern Transformers use pre-norm (normalize before the sublayer):

**Pre-norm:**

$$
x' = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

**Why pre-norm became standard:**

In post-norm, the residual path passes through LayerNorm, which can dampen gradients. In pre-norm, the residual path is completely clean: $x' = x + \text{something}$. This means gradients flow through the residual path with no attenuation at all.

The difference is dramatic for deep models:

- Post-norm 6-layer models: train well, but deeper models (24+ layers) require careful warmup
- Pre-norm models: train stably even at 100+ layers without warmup
- GPT-2, GPT-3, LLaMA, and most modern LLMs use pre-norm

---

## The Encoder Stack

The full Transformer encoder is $N$ blocks stacked sequentially:

$$
h^{(0)} = \text{TokenEmbedding}(x) + \text{PositionalEncoding}
$$

$$
h^{(l)} = \text{EncoderBlock}_l(h^{(l-1)}) \quad \text{for } l = 1, \ldots, N
$$

$$
\text{EncoderOutput} = h^{(N)}
$$

Each block has its own set of parameters (its own attention weights, FFN weights, and LayerNorm parameters), but they all share the same architecture.

**Why identical blocks?**

The simplicity of identical blocks has several advantages:

- Easier to implement and debug
- Easier to analyze theoretically
- Allows techniques like weight sharing (using the same parameters for all blocks, as in Universal Transformers)
- Allows easy scaling: just add more blocks

---

## Computational Flow Summary

The complete data flow through the encoder, from text to final representations:

**Input pipeline:**

- Raw text $\rightarrow$ Tokenizer $\rightarrow$ Token IDs $[z_1, \ldots, z_n]$
- Token IDs $\rightarrow$ Embedding layer $\rightarrow$ Scaled embeddings $[\mathbf{e}_1, \ldots, \mathbf{e}_n]$
- Scaled embeddings $+$ Positional encodings $= h^{(0)}$

**Encoder blocks (repeated N times):**

- $h^{(0)} \rightarrow$ Block 1 $\rightarrow h^{(1)}$
- $h^{(1)} \rightarrow$ Block 2 $\rightarrow h^{(2)}$
- $\ldots$
- $h^{(N-1)} \rightarrow$ Block N $\rightarrow h^{(N)}$

**Output:**

- $h^{(N)}$ is the final encoder output, shape $(B, L, d_{model})$
- Each vector $h^{(N)}_i$ is a rich contextual representation of token $i$, informed by the entire input sequence
- This output can be used for classification, fed to a decoder, or used for any downstream task

---

## The Encoder Block in Different Transformer Variants

While the basic structure remains the same, different Transformer variants modify the encoder block:

**BERT (Bidirectional Encoder Representations from Transformers):**

- Uses the standard encoder block with post-norm
- 12 blocks for BERT-Base, 24 for BERT-Large
- GELU activation instead of ReLU in the FFN
- No masking in self-attention (bidirectional: every token sees every other token)

**GPT (Generative Pre-trained Transformer):**

- Uses only decoder blocks (no encoder)
- Decoder blocks are similar to encoder blocks but with causal masking
- Pre-norm in GPT-2 and later

**Vision Transformer (ViT):**

- Uses standard encoder blocks
- Input is image patches instead of text tokens
- A special learnable "class token" is prepended to the sequence
- The class token's final representation is used for classification

**T5 (Text-to-Text Transfer Transformer):**

- Uses both encoder and decoder blocks
- Relative position biases instead of absolute positional encodings
- Pre-norm throughout

The encoder block's design has proven remarkably versatile, requiring only minor modifications to work across text, images, audio, video, protein sequences, and more.

---

## Why This Architecture Works

The encoder block's success can be attributed to several interacting design principles:

**Separation of concerns:**

Attention handles communication (which tokens should interact), and FFN handles computation (how to transform the gathered information). This clean separation makes each component's role clear and learnable.

**Residual learning:**

Each sub-layer only needs to learn the "residual" or "correction" to the identity function. This is easier than learning the full transformation from scratch, especially in deep networks.

**Normalization:**

LayerNorm keeps activations in a stable range, preventing the cascading instabilities that plague deep networks.

**Composability:**

Because each block has the same input and output shape, blocks can be freely stacked, removed, or shared. This makes the architecture highly modular and scalable.

**Parallelism:**

All positions within a sequence are processed simultaneously (no sequential dependencies within a layer). All heads within multi-head attention are computed in parallel. This makes the Transformer extremely efficient on modern parallel hardware.

These principles, simple individually, interact synergistically to create an architecture that scales from small models (a few million parameters) to enormous ones (hundreds of billions) while maintaining trainability and performance.
