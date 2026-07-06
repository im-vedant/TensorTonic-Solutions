## Why One Attention Head Is Not Enough

Scaled dot-product attention computes a single set of attention weights: for each query position, it produces one probability distribution over all keys. This means every query can only express one notion of "relevance" at a time.

But language requires attending to multiple things simultaneously. Consider the sentence:

"The cat that sat on the mat ate the fish."

For the word "ate," the model needs to simultaneously:

- Identify the subject ("cat") to determine who is performing the action
- Identify the object ("fish") to determine what is being acted upon
- Note the relative clause ("that sat on the mat") to understand the full context

A single attention head must compress all of these different types of relationships into a single distribution. It might focus strongly on the subject and lose the object, or vice versa.

Multi-head attention solves this by running multiple attention operations in parallel, each with its own learned projection. Different heads can specialize in capturing different types of relationships.

---

## The Multi-Head Attention Formula

Multi-head attention consists of three stages: **project**, **attend**, and **combine**.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W^O
$$

where each head computes its own attention:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

and $\text{Attention}$ is the scaled dot-product attention:

$$
\text{Attention}(Q', K', V') = \text{softmax}\left(\frac{Q'K'^T}{\sqrt{d_k}}\right) V'
$$

---

## The Projection Matrices

Each head $i$ has its own set of learned projection matrices:

- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ - projects input into query space for head $i$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ - projects input into key space for head $i$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ - projects input into value space for head $i$

These projections are the key to multi-head attention. By learning different $W^Q$, $W^K$, $W^V$ for each head, the model allows each head to attend to different aspects of the input.

**Intuition:**

Think of each projection as a "lens" through which a head views the input:

- Head 1 might project tokens into a subspace where syntactic features (part of speech, grammatical role) are prominent
- Head 2 might project into a subspace where semantic features (meaning, topic) are prominent
- Head 3 might project into a subspace where positional features (nearby vs. distant) are prominent

Each head sees the same input tokens but through a different lens, allowing it to discover different patterns.

---

## Dimension Splitting

A critical design choice: the per-head dimension $d_k$ is set to:

$$
d_k = d_v = \frac{d_{model}}{h}
$$

where $h$ is the number of heads.

**Why this split?**

If each head used the full $d_{model}$ dimensions, multi-head attention would cost $h$ times more than single-head attention. By splitting the dimension, the total computational cost remains approximately the same:

- Single-head attention with dimension $d_{model}$: cost $\propto d_{model}^2$
- $h$-head attention with dimension $d_k = d_{model}/h$: cost $\propto h \cdot d_k^2 = h \cdot (d_{model}/h)^2 = d_{model}^2/h$

The total cost is actually $d_{model}^2/h$ per head, times $h$ heads, giving $d_{model}^2$ total. This is the same as single-head attention.

**The original Transformer's configuration:**

- $d_{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 512 / 8 = 64$ per head

Each head operates on a 64-dimensional subspace, but with 8 heads working in parallel, the model captures 8 different types of attention patterns simultaneously.

---

## Step-by-Step Computation

The full multi-head attention computation proceeds as follows:

**Step 1: Linear projections**

For each head $i$, project the input into query, key, and value subspaces:

$$
Q_i = Q W_i^Q, \quad K_i = K W_i^K, \quad V_i = V W_i^V
$$

Where $Q$, $K$, $V$ are the original input matrices (often the same matrix $X$ in self-attention).

**Step 2: Parallel attention**

Each head independently computes scaled dot-product attention:

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

Each head produces an output of shape $(L, d_v)$ where $L$ is the sequence length.

**Step 3: Concatenation**

The outputs of all heads are concatenated along the feature dimension:

$$
\text{Concat}(\text{head}_1, \ldots, \text{head}_h) \in \mathbb{R}^{L \times (h \cdot d_v)} = \mathbb{R}^{L \times d_{model}}
$$

This reassembles the full $d_{model}$-dimensional representation from the $h$ sub-representations.

**Step 4: Output projection**

The concatenated output is multiplied by the output projection matrix:

$$
\text{Output} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$.

This final projection serves a critical role: it allows information from different heads to interact and combine. Without it, the heads would be completely independent and unable to share information.

---

## Worked Example

Consider $d_{model} = 4$, $h = 2$ heads, so $d_k = 4/2 = 2$.

**Input** (3 tokens, self-attention so $Q = K = V = X$):

$$
X = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{pmatrix}
$$

**Head 1 projections** (project into first 2D subspace):

Suppose $W_1^Q = W_1^K = W_1^V$ selects approximately the first two dimensions:

$$
Q_1, K_1, V_1 \approx \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

Head 1 attention scores:

$$
Q_1 K_1^T = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}
$$

After scaling by $\sqrt{2} \approx 1.41$ and softmax, token 3 attends strongly to itself (score 2 before scaling, the highest). Head 1 captures one pattern.

**Head 2 projections** (project into second 2D subspace):

Suppose $W_2^Q = W_2^K = W_2^V$ selects approximately the last two dimensions:

$$
Q_2, K_2, V_2 \approx \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}
$$

Head 2 captures a completely different pattern because it looks at different features of the input.

**Concatenation:**

$$
\text{Concat}(\text{head}_1, \text{head}_2) = [\text{head}_1 | \text{head}_2] \in \mathbb{R}^{3 \times 4}
$$

**Output projection:**

$$
\text{Output} = \text{Concat}(\text{head}_1, \text{head}_2) \cdot W^O \in \mathbb{R}^{3 \times 4}
$$

The output has the same shape as the input: $(3, 4)$.

---

## The Reshape Trick

In practice, the $h$ heads are not computed with $h$ separate matrix multiplications. Instead, a single large matrix multiplication is used, followed by a reshape operation.

**Efficient implementation:**

1. **Single projection**: Multiply input by the full projection matrix

$$
Q_{\text{full}} = X W^Q, \quad W^Q \in \mathbb{R}^{d_{model} \times d_{model}}
$$

This produces $Q_{\text{full}}$ of shape $(B, L, d_{model})$

2. **Reshape**: Split the last dimension into $h$ heads

$$
(B, L, d_{model}) \rightarrow (B, L, h, d_k) \rightarrow (B, h, L, d_k)
$$

The transpose moves the head dimension before the sequence dimension so that each head can be processed in parallel using batched matrix multiplication.

3. **Batched attention**: Compute attention for all heads simultaneously

$$
\text{scores} = \frac{Q_{\text{reshaped}} K_{\text{reshaped}}^T}{\sqrt{d_k}}
$$

Shape: $(B, h, L, L)$ - one attention matrix per head per batch element.

4. **Reshape back**: After attention, reshape from $(B, h, L, d_v)$ to $(B, L, d_{model})$

$$
(B, h, L, d_v) \rightarrow (B, L, h, d_v) \rightarrow (B, L, d_{model})
$$

5. **Output projection**: Apply $W^O$

This reshape approach turns $h$ small matrix multiplications into one large batched multiplication, which is much more efficient on modern hardware.

---

## What Do Different Heads Learn?

Research into trained Transformers has revealed that different attention heads specialize in remarkably distinct patterns:

**Syntactic heads:**

- Some heads learn to attend from a verb to its subject, even across long distances
- Other heads learn to connect adjectives to the nouns they modify
- Some heads track opening and closing brackets or quotation marks

**Positional heads:**

- Some heads primarily attend to the immediately preceding token (bigram patterns)
- Others attend to a fixed offset (e.g., always 3 positions back)
- Some attend primarily to the first token in the sequence

**Semantic heads:**

- Some heads connect pronouns to their antecedents ("she" attends to "Mary")
- Others connect related concepts across the sentence

**Redundant heads:**

- Some heads appear to be redundant and can be pruned without significant performance loss
- This has led to research on efficient Transformers that use fewer heads in later layers

The diversity of these patterns is a direct consequence of the multi-head design: by giving each head its own projection matrices, the model allows specialization to emerge naturally through training.

---

## The Output Projection: Why It Matters

The output projection $W^O$ is sometimes overlooked, but it plays a crucial role.

**Without $W^O$:**

The output would be the raw concatenation of head outputs. Each head's contribution would be confined to its own slice of the $d_{model}$-dimensional output. Head 1's output would occupy dimensions $0$ to $d_v - 1$, head 2 would occupy $d_v$ to $2d_v - 1$, and so on.

**With $W^O$:**

The output projection allows the model to mix information across heads. The final output at each dimension can be a learned combination of information from all heads. This is essential because the most useful representation might combine syntactic information from head 1 with semantic information from head 3.

$$
\text{output}_d = \sum_{i=1}^{h} \sum_{j=1}^{d_v} W^O_{(i-1)d_v + j, d} \cdot \text{head}_{i,j}
$$

Each output dimension is a weighted sum of all values from all heads.

---

## Parameter Count

For a single multi-head attention layer:

- Query projections: $W^Q \in \mathbb{R}^{d_{model} \times d_{model}}$ - $d_{model}^2$ parameters
- Key projections: $W^K \in \mathbb{R}^{d_{model} \times d_{model}}$ - $d_{model}^2$ parameters
- Value projections: $W^V \in \mathbb{R}^{d_{model} \times d_{model}}$ - $d_{model}^2$ parameters
- Output projection: $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ - $d_{model}^2$ parameters

**Total: $4 \cdot d_{model}^2$ parameters** (ignoring biases)

For $d_{model} = 512$: $4 \times 512^2 = 1{,}048{,}576 \approx 1$ million parameters per attention layer.

Note that this count is independent of the number of heads $h$. Whether you use 1 head or 16 heads, the total number of parameters is the same. The number of heads only changes how the internal dimensions are split.

---

## The Three Uses of Multi-Head Attention in the Transformer

The original Transformer architecture uses multi-head attention in three distinct places:

**1. Encoder self-attention:**

- $Q = K = V =$ encoder input (or previous encoder layer output)
- Each token attends to all other tokens in the source sequence
- Builds rich contextual representations of the input

**2. Decoder self-attention (masked):**

- $Q = K = V =$ decoder input (or previous decoder layer output)
- Each token attends only to previous tokens (masked to prevent future information leakage)
- Builds autoregressive representations for generation

**3. Encoder-decoder cross-attention:**

- $Q =$ decoder representations
- $K = V =$ encoder output
- Decoder tokens attend to the full encoder output
- This is how the decoder accesses information from the source sequence

Each of these uses the same multi-head attention mechanism but with different inputs and different masking patterns.

---

## Scaling Up: How Head Count Changes With Model Size

As Transformers have grown larger, the number of heads has increased proportionally:

- **Transformer Base**: $d_{model} = 512$, $h = 8$, $d_k = 64$
- **Transformer Big**: $d_{model} = 1024$, $h = 16$, $d_k = 64$
- **BERT-Base**: $d_{model} = 768$, $h = 12$, $d_k = 64$
- **BERT-Large**: $d_{model} = 1024$, $h = 16$, $d_k = 64$
- **GPT-3 (175B)**: $d_{model} = 12288$, $h = 96$, $d_k = 128$

Notice that $d_k$ tends to stay around 64-128 as models scale up. The additional capacity comes from more heads (capturing more types of patterns) rather than larger head dimensions.

---

## Multi-Head Attention and Ensemble Learning

There is a deep connection between multi-head attention and ensemble methods in machine learning.

In ensemble learning, multiple models are trained independently and their predictions are combined (averaged or voted on). The diversity among models is what gives ensembles their power: different models make different errors, and combining them cancels out individual mistakes.

Multi-head attention follows the same principle:

- Each head is like an independent "model" of attention
- Each head learns different projection matrices and captures different patterns
- The output projection combines their contributions, much like an ensemble combiner

The key difference is that in multi-head attention, all heads are trained jointly and share the same loss function. This allows them to specialize cooperatively rather than redundantly.

---

## Connection to the Broader Architecture

Multi-head attention is always followed by two more operations in the Transformer:

1. **Residual connection**: The input is added back to the attention output

$$
\text{residual} = X + \text{MultiHead}(X, X, X)
$$

2. **Layer normalization**: The result is normalized

$$
\text{output} = \text{LayerNorm}(\text{residual})
$$

This "Add & Norm" pattern ensures stable training in deep networks. The residual connection allows gradients to flow directly through the network, and layer normalization keeps activations in a stable range.

After this, the output enters the feed-forward network, which applies a non-linear transformation independently to each token position.

---

## Common Misconceptions

**"More heads is always better":**

This is not true. Beyond a certain point, adding more heads with smaller $d_k$ actually hurts performance. Each head needs a minimum dimension to form useful attention patterns. Empirically, $d_k$ below 32 tends to degrade performance.

**"Each head is trained to look at a specific thing":**

Heads are not explicitly trained to capture syntax, semantics, or positions. These specializations emerge naturally from the training objective. Some heads may not develop clear, interpretable patterns at all.

**"Attention happens at the token level":**

In multi-head attention, attention does not happen on the raw token embeddings. It happens on the projected representations $QW^Q$, $KW^K$, $VW^V$. The projections can completely transform what "similarity" means. Two tokens with very different embeddings might have high compatibility in a particular head's projected subspace.

**"The number of heads affects the parameter count":**

As shown in the parameter count analysis above, the total number of parameters in multi-head attention is $4 \cdot d_{model}^2$, completely independent of $h$. Changing the number of heads changes how computations are structured internally but not the overall capacity of the layer.
