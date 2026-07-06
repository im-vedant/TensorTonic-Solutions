## The Position Problem

A Transformer processes all tokens in a sequence simultaneously through parallel matrix operations. Unlike a recurrent neural network, which reads tokens one by one and naturally knows "this is the 5th word I have seen," the Transformer has no inherent notion of order.

This creates a serious problem. Consider two sentences:

- "The dog bit the man"
- "The man bit the dog"

Both contain the exact same words. Without position information, the Transformer would produce identical representations for both sentences, because the set of embeddings is identical. The attention mechanism computes dot products between all pairs of tokens, and dot products are symmetric with respect to ordering.

Formally, the attention operation is **permutation equivariant**: if you shuffle the input tokens, the output tokens shuffle in the same way but their values do not change. The model cannot distinguish position 1 from position 7.

Positional encoding solves this by injecting position information directly into the input representations.

---

## The Idea: Add Position to Meaning

The solution is simple in concept: create a unique vector for each position, and add it to the token embedding at that position.

$$
\mathbf{x}_i = \mathbf{e}_i + \text{PE}(i)
$$

where $\mathbf{e}_i$ is the token embedding at position $i$ and $\text{PE}(i) \in \mathbb{R}^{d_{model}}$ is the positional encoding vector for position $i$.

After this addition, the vector $\mathbf{x}_i$ carries both **what** the token is (from the embedding) and **where** it is (from the positional encoding). The Transformer can then use attention to compute relationships that depend on both identity and position.

The key design question is: how should we construct $\text{PE}(i)$?

---

## Sinusoidal Positional Encoding

The original Transformer paper uses a deterministic, formula-based encoding using sine and cosine functions at different frequencies:

$$
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)
$$

$$
\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
$$

where:

- $pos$ is the position in the sequence (0, 1, 2, ...)
- $i$ is the dimension index (0, 1, 2, ..., $d_{model}/2 - 1$)
- Even dimensions ($2i$) use sine, odd dimensions ($2i+1$) use cosine
- The denominator $10000^{2i/d_{model}}$ controls the frequency

Each pair of dimensions $(2i, 2i+1)$ forms a sine-cosine pair at a specific frequency, creating a unique "fingerprint" for each position.

---

## Understanding the Frequencies

The key to understanding sinusoidal encoding is the frequency term:

$$
\omega_i = \frac{1}{10000^{2i / d_{model}}}
$$

This creates a geometric progression of frequencies across dimensions:

- **Dimension 0-1** ($i = 0$): $\omega_0 = 1$. The sine and cosine complete a full cycle every $2\pi \approx 6.28$ positions. This encodes very fine-grained position differences.

- **Dimension 2-3** ($i = 1$): $\omega_1 = 1/10000^{2/d_{model}}$. Slightly lower frequency, slightly longer wavelength.

- **Last dimensions** ($i = d_{model}/2 - 1$): $\omega_{d/2-1} = 1/10000$. The sine and cosine have a period of $10000 \times 2\pi \approx 62{,}832$ positions. This encodes very coarse position information.

**Analogy to binary counting:**

Think of each sine-cosine pair as a "clock hand" rotating at a different speed:

- The fastest hand (low dimensions) ticks rapidly, distinguishing neighboring positions
- The slowest hand (high dimensions) rotates slowly, distinguishing positions that are far apart
- Together, the combination of all hands uniquely identifies every position

This is analogous to how binary numbers work: the least significant bit alternates every number (0, 1, 0, 1, ...), the next bit alternates every 2 numbers (0, 0, 1, 1, ...), and so on. Each bit (dimension) captures position information at a different scale.

---

## Why Sine and Cosine Together?

Using both sine and cosine for each frequency is not arbitrary. It enables the model to learn **relative position** relationships through simple linear operations.

**The linear relationship property:**

For any fixed offset $k$, the positional encoding at position $pos + k$ can be expressed as a linear transformation of the encoding at position $pos$:

$$
\begin{pmatrix} \text{PE}(pos+k, 2i) \\ \text{PE}(pos+k, 2i+1) \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} \text{PE}(pos, 2i) \\ \text{PE}(pos, 2i+1) \end{pmatrix}
$$

This is a **rotation matrix**. Moving $k$ positions forward is equivalent to rotating each sine-cosine pair by an angle proportional to $k$.

**Why this matters:**

The attention mechanism computes dot products between positions. If the model can represent relative offsets as linear transformations of absolute positions, then the attention weights can learn to be sensitive to relative distances.

For example, the model might learn that "the word 3 positions to the left is important for this word" without needing to know the absolute positions. The rotation property makes this possible.

If we used only sine (or only cosine), this rotation property would break. We need both components to represent a point on a circle, which can then be rotated.

---

## Worked Example

Consider $d_{model} = 4$ and a sequence of length 5.

**Frequencies:**

- Dimensions 0-1 ($i = 0$): $\omega_0 = 1/10000^{0/4} = 1/1 = 1$
- Dimensions 2-3 ($i = 1$): $\omega_1 = 1/10000^{2/4} = 1/100 = 0.01$

**Position 0:**

- $\text{PE}(0, 0) = \sin(0 \times 1) = \sin(0) = 0$
- $\text{PE}(0, 1) = \cos(0 \times 1) = \cos(0) = 1$
- $\text{PE}(0, 2) = \sin(0 \times 0.01) = \sin(0) = 0$
- $\text{PE}(0, 3) = \cos(0 \times 0.01) = \cos(0) = 1$
- Result: $[0, 1, 0, 1]$

**Position 1:**

- $\text{PE}(1, 0) = \sin(1 \times 1) = \sin(1) \approx 0.841$
- $\text{PE}(1, 1) = \cos(1 \times 1) = \cos(1) \approx 0.540$
- $\text{PE}(1, 2) = \sin(1 \times 0.01) = \sin(0.01) \approx 0.010$
- $\text{PE}(1, 3) = \cos(1 \times 0.01) = \cos(0.01) \approx 1.000$
- Result: $[0.841, 0.540, 0.010, 1.000]$

**Position 2:**

- $\text{PE}(2, 0) = \sin(2) \approx 0.909$
- $\text{PE}(2, 1) = \cos(2) \approx -0.416$
- $\text{PE}(2, 2) = \sin(0.02) \approx 0.020$
- $\text{PE}(2, 3) = \cos(0.02) \approx 1.000$
- Result: $[0.909, -0.416, 0.020, 1.000]$

Notice how the low-frequency dimensions (2-3) change very slowly between positions, while the high-frequency dimensions (0-1) change rapidly. Each position gets a unique combination.

---

## Uniqueness of Position Encodings

Each position receives a unique encoding vector. To see why, note that the encoding for position $pos$ is:

$$
\text{PE}(pos) = \left[\sin(pos \cdot \omega_0), \cos(pos \cdot \omega_0), \sin(pos \cdot \omega_1), \cos(pos \cdot \omega_1), \ldots\right]
$$

The frequencies $\omega_0, \omega_1, \ldots$ are incommensurable (their ratios are irrational), which guarantees that no two positions can produce the same encoding vector, even for arbitrarily long sequences.

**Dot product between positions:**

The dot product between the positional encodings at positions $pos_1$ and $pos_2$ depends only on their difference $\Delta = pos_1 - pos_2$:

$$
\text{PE}(pos_1)^T \text{PE}(pos_2) = \sum_{i=0}^{d_{model}/2 - 1} \cos(\Delta \cdot \omega_i)
$$

This means the "similarity" between two position encodings is a function of their relative distance, not their absolute positions. Nearby positions have higher dot products (more similar encodings), and distant positions have lower dot products.

---

## Why Not Just Use Integers?

A naive approach would be to encode position as a single number: position 0 gets value 0, position 1 gets value 1, position 100 gets value 100.

This fails for several reasons:

- **Unbounded magnitude**: Position 1000 would have a magnitude 1000 times larger than position 1, creating numerical instability
- **Single dimension**: One number cannot capture the rich structure needed for the model to reason about multiple position-dependent patterns simultaneously
- **No periodicity**: The model cannot easily learn patterns like "every 3rd word" or "the previous word"

The sinusoidal encoding solves all of these:

- **Bounded values**: Sine and cosine are always between $-1$ and $+1$, regardless of position
- **Multi-dimensional**: $d_{model}$ dimensions capture position at many scales
- **Periodic structure**: Different frequencies naturally encode patterns at different scales

---

## Addition vs. Concatenation

The Transformer adds positional encodings to token embeddings rather than concatenating them. This is a design choice with important implications.

**Addition** ($d_{model}$ total dimensions):

$$
\mathbf{x}_i = \mathbf{e}_i + \text{PE}(i)
$$

- Keeps the dimensionality unchanged
- Token and position information share the same vector space
- The model must learn to disentangle the two in downstream layers
- More parameter-efficient

**Concatenation** ($2 \times d_{model}$ total dimensions):

$$
\mathbf{x}_i = [\mathbf{e}_i ; \text{PE}(i)]
$$

- Doubles the dimensionality
- Token and position information are in separate subspaces
- No interference between the two
- But doubles the cost of all subsequent operations

The Transformer uses addition. The scaling factor $\sqrt{d_{model}}$ applied to embeddings ensures that the embedding magnitudes are comparable to the positional encoding magnitudes, so neither dominates after addition.

---

## Learned vs. Fixed Positional Encodings

The Transformer paper proposes sinusoidal (fixed) encodings, but also tested learned positional embeddings:

**Sinusoidal (fixed):**

- No additional parameters
- Deterministic: same positions always get the same encoding
- Can theoretically extrapolate to sequence lengths not seen during training
- The geometric frequency structure provides useful inductive biases

**Learned:**

- One learnable vector per position, stored in an embedding table of shape $(L_{max}, d_{model})$
- More flexible: can learn any pattern the data requires
- Cannot extrapolate beyond $L_{max}$ (the maximum training sequence length)
- Adds $L_{max} \times d_{model}$ parameters

**The paper's finding:**

The original Transformer paper found that both approaches produced "nearly identical results." Despite this, the two approaches diverged in practice:

- BERT uses learned positional embeddings (max length 512)
- GPT-2 uses learned positional embeddings (max length 1024)
- The original Transformer uses sinusoidal encodings
- Most modern large language models use learned positions or more advanced schemes

---

## Beyond Sinusoidal: Modern Positional Encodings

The sinusoidal approach from the original paper has been largely superseded by more sophisticated methods in modern Transformers:

**Rotary Position Embeddings (RoPE):**

Used in LLaMA, PaLM, and many modern LLMs. Instead of adding position information, RoPE rotates the query and key vectors based on their position:

$$
\text{RoPE}(q, pos) = R_{pos} \cdot q
$$

where $R_{pos}$ is a rotation matrix. The dot product between rotated queries and keys naturally depends on relative position.

**ALiBi (Attention with Linear Biases):**

Instead of modifying embeddings, ALiBi adds a position-dependent bias directly to the attention scores:

$$
\text{score}(q_i, k_j) = q_i^T k_j - m \cdot |i - j|
$$

where $m$ is a head-specific slope. This penalizes long-range attention and supports length extrapolation.

**Relative Position Encodings:**

Instead of encoding absolute positions, encode the relative distance between every pair of tokens. Used in Transformer-XL and T5.

Each of these approaches addresses limitations of the original sinusoidal scheme, particularly the challenge of generalizing to sequence lengths longer than those seen during training.

---

## The 10,000 Base

The constant $10{,}000$ in the denominator is a design choice, not a mathematical necessity:

$$
\omega_i = \frac{1}{10000^{2i / d_{model}}}
$$

**Why 10,000?**

- It sets the longest wavelength to $10{,}000 \times 2\pi \approx 62{,}832$ positions
- This means the encoding can distinguish positions up to roughly 62,000 apart
- For the typical training sequences of the time (a few hundred tokens), this provided ample range
- The specific value was likely chosen empirically

**What if we change it?**

- A smaller base (e.g., 100) would compress the frequency range, with the slowest dimensions cycling faster. This could lose the ability to distinguish very distant positions.
- A larger base (e.g., 1,000,000) would spread the frequencies further apart, providing finer-grained position information but potentially making it harder for the model to learn position-dependent patterns.

In practice, the base of 10,000 has proven robust across many applications and model sizes.

---

## Dimensional Analysis

The positional encoding matrix has a clear geometric structure:

**Input**: Two integers, $\text{seq\_length}$ and $d_{model}$

**Output**: A matrix of shape $(\text{seq\_length}, d_{model})$

- Each row is the encoding for one position
- Each column pair $(2i, 2i+1)$ is a sine-cosine pair at frequency $\omega_i$

**The division term** is often computed as:

$$
\text{div\_term}_i = e^{-2i \cdot \ln(10000) / d_{model}} = 10000^{-2i/d_{model}} = \frac{1}{10000^{2i/d_{model}}}
$$

The exponential form is used in implementations for numerical stability, avoiding computing $10000^{2i/d_{model}}$ directly (which could overflow for large $d_{model}$).

**Constructing the full matrix:**

1. Create a column vector of positions: $[0, 1, 2, \ldots, L-1]^T$, shape $(L, 1)$
2. Create a row vector of division terms: $[\omega_0, \omega_1, \ldots]$, shape $(1, d_{model}/2)$
3. Compute the outer product: positions $\times$ frequencies, shape $(L, d_{model}/2)$
4. Apply sine to get even columns, cosine to get odd columns
5. Interleave to get the final $(L, d_{model})$ matrix

---

## Positional Encoding and Attention Patterns

The positional encoding influences attention patterns in important ways.

**Local attention bias:**

Because nearby positions have similar encodings (high dot product), the attention mechanism naturally tends to attend more to nearby tokens. This is a useful inductive bias for language, where nearby words are often more relevant than distant ones.

**Long-range connections:**

The low-frequency dimensions (slow-changing sine and cosine) allow the model to detect relationships between distant positions. Different attention heads can specialize: some may focus on local patterns using high-frequency position information, while others may capture long-range dependencies using low-frequency information.

**Position-dependent attention:**

After training, different attention heads learn to use position information differently:

- Some heads learn a "previous word" attention pattern
- Some learn to attend to specific relative positions (e.g., position $-3$ for a particular syntactic pattern)
- Some learn to attend to a fixed absolute position (e.g., the first token)

These diverse attention patterns emerge from the rich multi-scale position information provided by the sinusoidal encoding.

---

## Why Positional Encoding Works

At first glance, it might seem problematic to add two unrelated signals (token identity and position) into the same vector. Won't the position information corrupt the token information, or vice versa?

In practice, the high-dimensional space provides enough room for both. With $d_{model} = 512$, the model has 512 dimensions to represent both what a token is and where it is. The Transformer layers learn to use some dimensions primarily for position and others primarily for content.

**Information-theoretic perspective:**

- Token embeddings typically occupy a low-dimensional subspace of $\mathbb{R}^{d_{model}}$
- Positional encodings occupy a different low-dimensional subspace
- As long as these subspaces do not overlap too much, the model can distinguish the two signals

This is why the scaling factor on embeddings matters: it ensures the two signals have comparable magnitudes, so neither overwhelms the other.
