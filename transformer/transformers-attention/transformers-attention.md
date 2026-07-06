## The Core Idea: Dynamic Information Routing

In a traditional feed-forward network, every token is processed independently. Token at position 3 has no knowledge of token at position 7. But language is fundamentally relational: the meaning of a word depends heavily on the words around it.

Consider: "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? To answer this, the representation of "it" must somehow incorporate information from "animal." Attention is the mechanism that makes this possible.

**Attention allows every token to look at every other token and selectively gather information from the most relevant ones.** Instead of fixed, hard-coded connections, attention creates dynamic, input-dependent connections that change based on what the model is currently processing.

---

## The Attention Formula

The Transformer uses a specific form of attention called **scaled dot-product attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This single formula is the engine that powers modern language models, image classifiers, protein folders, and much more. Let us break it down piece by piece.

---

## Queries, Keys, and Values

The three inputs to attention, $Q$, $K$, and $V$, play distinct roles:

**Query ($Q$)**: "What am I looking for?"

- Each row of $Q$ represents a question being asked by one token
- The query encodes what kind of information this token needs
- Shape: $(n, d_k)$ where $n$ is the number of query positions

**Key ($K$)**: "What do I contain?"

- Each row of $K$ represents an advertisement of what information one token can provide
- The key is matched against queries to determine relevance
- Shape: $(m, d_k)$ where $m$ is the number of key positions

**Value ($V$)**: "Here is my actual information."

- Each row of $V$ contains the actual content that will be aggregated
- Values are weighted by the attention scores and summed
- Shape: $(m, d_v)$ where $m$ is the number of value positions

**The analogy of a library search:**

- You walk into a library with a question (query)
- Each book has a title on its spine (key)
- You compare your question to each title to find the most relevant books (dot product)
- You read the content of those books (values), paying more attention to the most relevant ones (weighted sum)

---

## Step 1: Computing Compatibility Scores

The first operation computes how well each query matches each key:

$$
S = QK^T
$$

This is a matrix multiplication that produces a **score matrix** $S$ of shape $(n, m)$, where $S_{ij}$ is the dot product between query $i$ and key $j$.

**What the dot product measures:**

The dot product between two vectors measures their alignment:

$$
\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta
$$

- High positive value: vectors point in the same direction (high compatibility)
- Near zero: vectors are orthogonal (no particular relationship)
- High negative value: vectors point in opposite directions (low compatibility)

So $S_{ij}$ measures "how relevant is key $j$ to query $i$?"

---

## Step 2: Scaling

The raw dot products are divided by $\sqrt{d_k}$:

$$
S_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}}
$$

**Why scale?**

This is not cosmetic. Without scaling, the dot products grow with the dimension $d_k$. To see why, consider the statistical argument from the original paper.

Assume the components of $Q$ and $K$ are independent random variables with mean 0 and variance 1. The dot product of a query vector $\mathbf{q}$ and a key vector $\mathbf{k}$, each of dimension $d_k$, is:

$$
\mathbf{q} \cdot \mathbf{k} = \sum_{j=1}^{d_k} q_j k_j
$$

Each term $q_j k_j$ has:

- Mean: $E[q_j k_j] = E[q_j]E[k_j] = 0$
- Variance: $\text{Var}(q_j k_j) = E[q_j^2 k_j^2] - (E[q_j k_j])^2 = 1 \cdot 1 - 0 = 1$

Since the $d_k$ terms are independent, the total dot product has:

$$
E[\mathbf{q} \cdot \mathbf{k}] = 0, \quad \text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k
$$

So the standard deviation is $\sqrt{d_k}$. For $d_k = 64$, dot products typically range from about $-16$ to $+16$.

**The softmax saturation problem:**

When softmax receives inputs with large magnitude, it produces outputs that are very close to 0 or 1:

$$
\text{softmax}([10, 1, 1]) \approx [0.9999, 0.00005, 0.00005]
$$

In this regime, the gradients of softmax become extremely small, which slows or stops learning. This is the "vanishing gradient in softmax" problem.

**The fix:**

Dividing by $\sqrt{d_k}$ rescales the dot products to have variance 1, regardless of $d_k$:

$$
\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

This keeps the softmax inputs in a moderate range where gradients are healthy.

---

## Step 3: Softmax

The scaled scores are passed through softmax along the key dimension:

$$
\alpha_{ij} = \frac{\exp(S_{\text{scaled}, ij})}{\sum_{k=1}^{m} \exp(S_{\text{scaled}, ik})}
$$

This converts the raw scores into a **probability distribution** over keys for each query.

**Properties of attention weights $\alpha$:**

- $\alpha_{ij} \geq 0$ for all $i, j$ (non-negative)
- $\sum_{j=1}^{m} \alpha_{ij} = 1$ for each $i$ (rows sum to 1)
- $\alpha_{ij}$ is the "attention" that query $i$ pays to key $j$

The attention weight matrix $\alpha$ has shape $(n, m)$. Each row is a probability distribution that says "how much should I attend to each position?"

**Sharpness vs. diffuseness:**

- If one score is much larger than the others, softmax produces a near-one-hot distribution (sharp attention, focusing on one position)
- If all scores are similar, softmax produces a near-uniform distribution (diffuse attention, attending equally to all positions)
- The scaling by $\sqrt{d_k}$ controls this trade-off

---

## Step 4: Weighted Aggregation

The final step multiplies the attention weights by the values:

$$
\text{Output} = \alpha V
$$

For query $i$, the output is:

$$
\text{output}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j
$$

This is a **weighted average** of all value vectors, where the weights are the attention probabilities. Each output vector is a blend of the values from all positions, with the most relevant positions contributing the most.

**The full picture:**

$$
\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)}_{\text{attention weights}} \underbrace{V}_{\text{values to aggregate}}
$$

---

## Worked Example

Consider 3 tokens with $d_k = 2$:

**Inputs:**

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{pmatrix}, \quad
V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \\ 5 & 5 \end{pmatrix}
$$

**Step 1: Compute $QK^T$**

$$
QK^T = \begin{pmatrix} 1 \cdot 1 + 0 \cdot 0 & 1 \cdot 0 + 0 \cdot 1 & 1 \cdot 0.5 + 0 \cdot 0.5 \\ 0 \cdot 1 + 1 \cdot 0 & 0 \cdot 0 + 1 \cdot 1 & 0 \cdot 0.5 + 1 \cdot 0.5 \\ 1 \cdot 1 + 1 \cdot 0 & 1 \cdot 0 + 1 \cdot 1 & 1 \cdot 0.5 + 1 \cdot 0.5 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{pmatrix}
$$

**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$**

$$
S_{\text{scaled}} = \begin{pmatrix} 0.707 & 0 & 0.354 \\ 0 & 0.707 & 0.354 \\ 0.707 & 0.707 & 0.707 \end{pmatrix}
$$

**Step 3: Apply softmax (row-wise)**

Row 1: $\text{softmax}([0.707, 0, 0.354]) \approx [0.430, 0.212, 0.302]$

- Query 1 attends mostly to Key 1 (both have strong first component)

Row 2: $\text{softmax}([0, 0.707, 0.354]) \approx [0.212, 0.430, 0.302]$

- Query 2 attends mostly to Key 2 (both have strong second component)

Row 3: $\text{softmax}([0.707, 0.707, 0.707]) \approx [0.333, 0.333, 0.333]$

- Query 3 attends equally to all keys (equal compatibility with all)

**Step 4: Multiply by V**

$$
\text{Output}_1 = 0.430 \cdot [10, 0] + 0.212 \cdot [0, 10] + 0.302 \cdot [5, 5] \approx [5.81, 3.63]
$$

$$
\text{Output}_2 = 0.212 \cdot [10, 0] + 0.430 \cdot [0, 10] + 0.302 \cdot [5, 5] \approx [3.63, 5.81]
$$

$$
\text{Output}_3 = 0.333 \cdot [10, 0] + 0.333 \cdot [0, 10] + 0.333 \cdot [5, 5] = [5.0, 5.0]
$$

Query 1 produces an output closer to Value 1. Query 2 produces an output closer to Value 2. Query 3, which attends equally, produces the average of all values.

---

## Self-Attention vs. Cross-Attention

The Transformer uses attention in two distinct ways:

**Self-attention:**

$Q$, $K$, and $V$ all come from the same sequence. Each token attends to all other tokens in the same sequence (including itself).

$$
\text{SelfAttention}(X) = \text{Attention}(XW^Q, XW^K, XW^V)
$$

where $X$ is the input sequence and $W^Q, W^K, W^V$ are learned projection matrices.

This is used in:
- Encoder self-attention (each token attends to all tokens)
- Decoder self-attention (each token attends to previous tokens only)

**Cross-attention:**

$Q$ comes from one sequence, while $K$ and $V$ come from a different sequence. This allows one sequence to gather information from another.

$$
\text{CrossAttention}(X, Y) = \text{Attention}(XW^Q, YW^K, YW^V)
$$

This is used in:
- Decoder cross-attention (decoder tokens attend to encoder output)
- Image captioning (text tokens attend to image patches)

---

## Masking: Preventing Future Information Leakage

In autoregressive models (like GPT), the model generates tokens one at a time, left to right. During training, we process the entire sequence at once for efficiency, but we must ensure that position $i$ cannot attend to positions $j > i$ (future positions).

**How masking works:**

Before applying softmax, we set the scores for future positions to $-\infty$:

$$
S_{ij}^{\text{masked}} = \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

When softmax processes $-\infty$, it produces 0:

$$
\text{softmax}(-\infty) = \frac{e^{-\infty}}{\sum} = \frac{0}{\sum} = 0
$$

So future positions receive zero attention weight, effectively making them invisible to the current position.

**The mask matrix** is a lower-triangular matrix of ones:

$$
M = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{pmatrix}
$$

Position 1 can only see position 1. Position 2 can see positions 1-2. Position 4 can see all four positions.

---

## Attention as a Soft Database Query

A powerful analogy for understanding attention is the database query:

**Hard lookup (traditional database):**

- Query: "Find the row where key = X"
- Returns exactly one row (or none)
- Binary: either a match or not

**Soft lookup (attention):**

- Query: "Find the rows most similar to X"
- Returns a weighted combination of all rows
- Continuous: every row contributes, with the most similar rows contributing the most

This "soft" nature is what makes attention differentiable and trainable. A hard lookup has zero gradients almost everywhere (the output does not change smoothly with the query), but a soft lookup changes smoothly, allowing backpropagation to optimize the queries and keys.

---

## Why Dot-Product Attention?

The Transformer paper specifically chooses dot-product attention over alternatives:

**Additive attention:**

$$
\text{score}(q, k) = v^T \tanh(W_q q + W_k k)
$$

- Uses a small neural network to compute compatibility
- More expressive in theory
- Slower in practice because it cannot be parallelized as matrix multiplications

**Dot-product attention:**

$$
\text{score}(q, k) = q^T k
$$

- Simpler: just a dot product
- Much faster: can be computed as a single matrix multiplication $QK^T$
- Benefits from highly optimized matrix multiplication hardware (GPUs, TPUs)

**Multiplicative attention:**

$$
\text{score}(q, k) = q^T W k
$$

- Adds a learned weight matrix $W$ between query and key
- More expressive than plain dot product
- In multi-head attention, the projection matrices $W^Q$ and $W^K$ effectively provide this learned transformation

The paper notes that dot-product and additive attention have similar theoretical performance, but dot-product attention is "much faster and more space-efficient in practice" due to optimized matrix multiplication.

---

## Computational Complexity

**Time complexity:**

The dominant operation is the matrix multiplication $QK^T$:

$$
O(n \cdot m \cdot d_k)
$$

For self-attention where $n = m = L$ (sequence length):

$$
O(L^2 \cdot d_k)
$$

This **quadratic scaling** with sequence length is the main computational bottleneck of Transformers. Doubling the sequence length quadruples the computation.

**Memory complexity:**

The attention weight matrix $\alpha$ has shape $(L, L)$, requiring $O(L^2)$ memory. For $L = 4096$, this is about 16 million entries per attention head.

**Practical implications:**

- Sequence length 512: manageable on most GPUs
- Sequence length 2048: requires careful memory management
- Sequence length 8192+: requires techniques like Flash Attention, sliding window, or sparse attention

---

## Attention Patterns in Practice

After training, different attention heads develop specialized behaviors:

- **Local attention**: Strongly attends to neighboring positions (captures local syntax)
- **Strided attention**: Attends to every $k$-th position (captures periodic patterns)
- **Global attention**: Attends primarily to the first or last token (captures global context)
- **Syntactic attention**: Follows grammatical structure (e.g., verb attends to its subject)
- **Coreference attention**: Connects pronouns to their antecedents

These patterns emerge naturally from training on language data. The model discovers that different types of attention are useful for different aspects of language understanding.

---

## The Power of Attention

Attention fundamentally changed deep learning because it provides three properties simultaneously:

- **Long-range connections**: Any token can directly attend to any other token, regardless of distance. In RNNs, distant tokens must pass information through many intermediate steps, causing information loss.

- **Parallelism**: All attention operations can be computed simultaneously, unlike RNNs which must process tokens sequentially. This enables efficient GPU utilization.

- **Interpretability**: The attention weights $\alpha$ provide a human-readable map of which tokens the model considers important for each prediction. While imperfect as an explanation tool, this is more interpretable than the hidden states of an RNN.

These properties made Transformers the dominant architecture across NLP, computer vision, speech processing, and beyond.

---

## Dimensional Analysis

Tracking shapes through the attention computation:

**Inputs:**

- $Q$: shape $(B, n, d_k)$ - $B$ is batch size, $n$ is number of queries, $d_k$ is key dimension
- $K$: shape $(B, m, d_k)$ - $m$ is number of keys
- $V$: shape $(B, m, d_v)$ - $d_v$ is value dimension (can differ from $d_k$)

**Step-by-step shapes:**

- $QK^T$: $(B, n, d_k) \times (B, d_k, m) = (B, n, m)$ - one score per query-key pair
- After scaling: $(B, n, m)$ - same shape, values divided by $\sqrt{d_k}$
- After softmax: $(B, n, m)$ - same shape, each row sums to 1
- $\alpha V$: $(B, n, m) \times (B, m, d_v) = (B, n, d_v)$ - one output vector per query

**Key observation:** The output has the same number of vectors as there are queries ($n$), and each output vector has dimension $d_v$. In the standard Transformer, $d_k = d_v = d_{model} / h$ where $h$ is the number of attention heads, so inputs and outputs have the same shape.

**Parameter count:**

Scaled dot-product attention itself has **zero learnable parameters**. It is a pure function of its inputs $Q$, $K$, and $V$. All the learnable parameters come from the linear projections that create $Q$, $K$, and $V$ from the input embeddings, which are part of the multi-head attention wrapper.
