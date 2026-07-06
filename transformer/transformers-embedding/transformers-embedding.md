## From Token IDs to Vectors

After tokenization converts raw text into a sequence of integer IDs, the model faces a fundamental challenge: integers are discrete, but neural networks operate on continuous vectors. The embedding layer bridges this gap by mapping each token ID to a dense, learnable vector representation.

This seemingly simple operation, a table lookup, is one of the most important components in any language model. The quality of these learned embeddings determines how well the model can represent the meaning, grammar, and relationships between words.

---

## The Embedding Matrix

The embedding layer is defined by a single weight matrix:

$$
W_E \in \mathbb{R}^{V \times d_{model}}
$$

where $V$ is the vocabulary size and $d_{model}$ is the embedding dimension (also called the model dimension).

Each row of this matrix is the learned vector representation for one token:

- Row 0 is the embedding for token ID 0
- Row 1 is the embedding for token ID 1
- Row $i$ is the embedding for token ID $i$

For a vocabulary of $V = 30{,}000$ tokens and $d_{model} = 512$, the embedding matrix contains $30{,}000 \times 512 = 15{,}360{,}000$ parameters. This is often one of the largest parameter blocks in the model.

---

## The Lookup Operation

Given a token ID $z$, the embedding operation simply selects the corresponding row from $W_E$:

$$
\mathbf{e} = W_E[z]
$$

For a sequence of token IDs $[z_1, z_2, \ldots, z_n]$, we look up each one:

$$
[\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n] = [W_E[z_1], W_E[z_2], \ldots, W_E[z_n]]
$$

The result is a matrix of shape $(n, d_{model})$ where each row is the embedding vector for the corresponding token.

**Equivalence to one-hot multiplication:**

The lookup operation is mathematically equivalent to multiplying a one-hot encoded vector by the embedding matrix:

$$
\mathbf{e} = \mathbf{1}_z^T \cdot W_E
$$

where $\mathbf{1}_z \in \mathbb{R}^V$ is a one-hot vector with a 1 at position $z$ and zeros elsewhere.

This equivalence is important conceptually (it shows that embedding lookup is a special case of a linear layer), but in practice, direct indexing is used because it is far more efficient than constructing and multiplying large one-hot vectors.

---

## Why Dense Vectors?

One-hot representations are sparse and high-dimensional: a vocabulary of 30,000 tokens would require 30,000-dimensional vectors where exactly one element is 1 and the rest are 0.

Dense embeddings are preferable for several reasons:

- **Compactness**: A 512-dimensional vector carries richer information than a 30,000-dimensional one-hot vector
- **Similarity**: Dense vectors can capture semantic similarity. Words with related meanings will have embeddings that point in similar directions.
- **Generalization**: If "cat" and "dog" have similar embeddings, what the model learns about "cat" partially transfers to "dog"
- **Composability**: Dense vectors can be added, averaged, and transformed by linear layers, enabling complex reasoning

**The geometry of meaning:**

In a well-trained embedding space, semantic relationships manifest as geometric relationships:

- Similar words cluster together (e.g., "cat", "dog", "hamster")
- Analogical relationships form parallelograms (e.g., "king" - "man" + "woman" $\approx$ "queen")
- Different semantic dimensions correspond to different directions in the vector space

---

## The Scaling Factor

The Transformer paper introduces a crucial detail that is easy to overlook: the embeddings are multiplied by $\sqrt{d_{model}}$ before being added to positional encodings:

$$
E(x) = W_E[x] \cdot \sqrt{d_{model}}
$$

This scaling factor serves a specific mathematical purpose.

**Why scale up?**

Embedding vectors are typically initialized with small random values. A common initialization draws each component independently from:

$$
W_{E_{ij}} \sim \mathcal{N}\left(0, \frac{1}{d_{model}}\right) \quad \text{or} \quad \mathcal{U}\left(-\frac{1}{\sqrt{d_{model}}}, \frac{1}{\sqrt{d_{model}}}\right)
$$

With this initialization, the expected magnitude of an embedding vector is approximately 1, regardless of $d_{model}$.

Meanwhile, the sinusoidal positional encodings (which are added to the embeddings) have components that are sine and cosine values, bounded between $-1$ and $+1$. For a vector of dimension $d_{model}$, the positional encoding has magnitude approximately $\sqrt{d_{model}/2}$.

Without scaling, the positional encodings would dominate the embeddings, especially for large $d_{model}$. Multiplying the embeddings by $\sqrt{d_{model}}$ brings their magnitude to roughly $\sqrt{d_{model}}$, comparable to the positional encoding magnitude.

**The balance:**

$$
\|\mathbf{e}\| \approx \sqrt{d_{model}} \quad \text{(after scaling)}
$$

$$
\|\text{PE}\| \approx \sqrt{d_{model}/2} \quad \text{(positional encoding)}
$$

This ensures that neither the token identity nor the position information overwhelms the other when they are summed.

---

## Worked Example

Consider a tiny vocabulary with $V = 5$ and $d_{model} = 4$.

**Embedding matrix** (randomly initialized):

$$
W_E = \begin{pmatrix} 0.1 & -0.2 & 0.3 & -0.1 \\ -0.3 & 0.4 & 0.1 & 0.2 \\ 0.2 & 0.1 & -0.4 & 0.3 \\ -0.1 & -0.3 & 0.2 & 0.4 \\ 0.4 & 0.2 & -0.1 & -0.2 \end{pmatrix}
$$

**Input tokens**: $[2, 0, 4]$ (a sequence of 3 tokens)

**Step 1: Look up embeddings**

- Token 2: $W_E[2] = [0.2, 0.1, -0.4, 0.3]$
- Token 0: $W_E[0] = [0.1, -0.2, 0.3, -0.1]$
- Token 4: $W_E[4] = [0.4, 0.2, -0.1, -0.2]$

**Result** (shape $3 \times 4$):

$$
\begin{pmatrix} 0.2 & 0.1 & -0.4 & 0.3 \\ 0.1 & -0.2 & 0.3 & -0.1 \\ 0.4 & 0.2 & -0.1 & -0.2 \end{pmatrix}
$$

**Step 2: Apply scaling**

With $d_{model} = 4$, the scaling factor is $\sqrt{4} = 2$:

$$
\begin{pmatrix} 0.4 & 0.2 & -0.8 & 0.6 \\ 0.2 & -0.4 & 0.6 & -0.2 \\ 0.8 & 0.4 & -0.2 & -0.4 \end{pmatrix}
$$

These scaled embeddings will then be added to positional encodings before entering the Transformer.

---

## Embedding Dimension

The embedding dimension $d_{model}$ is a key architectural choice. It determines:

- **Representational capacity**: Larger dimensions can encode more nuanced distinctions between tokens
- **Computational cost**: Larger dimensions increase the cost of every subsequent operation (attention, FFN)
- **Memory footprint**: The embedding matrix size scales linearly with $d_{model}$

**Common choices:**

- $d_{model} = 256$: Small models, fast experimentation
- $d_{model} = 512$: The original Transformer ("base" model)
- $d_{model} = 768$: BERT-Base, GPT-2 Small
- $d_{model} = 1024$: Transformer Big, BERT-Large
- $d_{model} = 4096$: GPT-3, LLaMA-7B
- $d_{model} = 12288$: GPT-3 175B

The embedding dimension is the same throughout the entire Transformer. Every layer, every attention head, and every feed-forward network operates on vectors of size $d_{model}$ (or subdivisions of it).

---

## Weight Tying

The original Transformer paper shares the embedding matrix between three places:

1. **Input embedding**: Maps token IDs to vectors for the encoder
2. **Output embedding**: Maps token IDs to vectors for the decoder
3. **Output projection**: Maps the final hidden state back to vocabulary-size logits

**How weight tying works:**

Without weight tying, the output projection uses a separate matrix $W_{out} \in \mathbb{R}^{d_{model} \times V}$:

$$
\text{logits} = h \cdot W_{out}
$$

With weight tying:

$$
W_{out} = W_E^T
$$

So the output logit for token $j$ is simply the dot product between the hidden state $h$ and the embedding for token $j$:

$$
\text{logits}_j = h \cdot W_E[j]^T
$$

**Why this works:**

Intuitively, the model is asking: "which token's embedding is most similar to the hidden state?" The token with the highest dot product (most similar embedding) gets the highest logit and highest probability after softmax.

**Benefits:**

- Reduces parameters by $V \times d_{model}$ (millions of parameters for typical vocabularies)
- Improves generalization by forcing consistency between input and output representations
- Empirically improves performance, especially for smaller models

---

## Learning Embeddings

Embedding vectors are initialized randomly and learned through backpropagation alongside all other model parameters.

**Gradient flow:**

During training, the loss function produces gradients that flow back through the model. When these gradients reach the embedding layer, they update only the rows that were looked up during the forward pass:

$$
\frac{\partial \mathcal{L}}{\partial W_E[z]} = \frac{\partial \mathcal{L}}{\partial \mathbf{e}} \quad \text{(only for tokens used in the current batch)}
$$

Rows corresponding to tokens not present in the current batch receive zero gradients and are not updated.

**Implication for rare tokens:**

Rare tokens appear in few training batches, so their embeddings receive few gradient updates and may remain poorly trained. This is one reason why subword tokenization is preferred: by decomposing rare words into common subwords, every piece gets frequent updates.

**Training dynamics:**

- Early in training, embeddings are essentially random and carry no semantic information
- As training progresses, the model gradually organizes the embedding space so that semantically related tokens have similar vectors
- Embeddings continue to improve throughout training, even in late stages

---

## Pre-trained vs. Learned From Scratch

There are two approaches to initializing embeddings:

**Learned from scratch:**

- Initialize randomly (uniform or normal distribution)
- Train jointly with the rest of the model
- This is the standard approach for large-scale language models (GPT, BERT, LLaMA)
- Requires large datasets and long training to produce good embeddings

**Pre-trained embeddings (Word2Vec, GloVe, FastText):**

- Initialize the embedding matrix with vectors pre-trained on a large corpus using a simpler objective
- Optionally fine-tune them during training, or freeze them
- Useful when training data is limited (the pre-trained embeddings provide a good starting point)

**The Transformer paper** uses learned embeddings trained from scratch, and this has become the dominant approach for Transformer models. With sufficient training data, jointly-learned embeddings outperform pre-trained ones because they are optimized specifically for the task at hand.

---

## Embeddings vs. Traditional Representations

Before neural embeddings, NLP relied on hand-crafted feature representations:

**Bag of Words:**

Represents text as a vector of word counts. Ignores word order entirely.

$$
\text{BoW}(\text{"the cat sat on the mat"}) = \{the: 2, cat: 1, sat: 1, on: 1, mat: 1\}
$$

**TF-IDF:**

Weights words by their frequency in the document relative to their frequency across all documents:

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \log\frac{N}{\text{DF}(w)}
$$

**Neural embeddings** differ fundamentally:

- They are dense (hundreds of dimensions) rather than sparse (thousands)
- They are learned from data rather than hand-crafted
- They capture distributional semantics: "you shall know a word by the company it keeps" (Firth, 1957)
- They can be updated during training to optimize for the specific task

---

## Dimensional Analysis

Understanding the shapes of tensors as they flow through the embedding layer:

**Input**: A tensor of token IDs with shape $(B, L)$

- $B$ is the batch size
- $L$ is the sequence length
- Each element is an integer in $\{0, 1, \ldots, V-1\}$

**Embedding matrix**: Shape $(V, d_{model})$

**Output (before scaling)**: Shape $(B, L, d_{model})$

- Each token ID has been replaced by its $d_{model}$-dimensional embedding vector

**Output (after scaling)**: Shape $(B, L, d_{model})$

- Same shape, but each value multiplied by $\sqrt{d_{model}}$

**Memory analysis:**

For BERT-Base with $V = 30{,}522$ and $d_{model} = 768$:

- Embedding matrix: $30{,}522 \times 768 = 23{,}440{,}896$ parameters
- At 32-bit float: approximately 89 MB
- This is about 21% of BERT-Base's total 110M parameters

---

## Practical Considerations

**Embedding initialization:**

- Normal distribution: $\mathcal{N}(0, 1/\sqrt{d_{model}})$ or $\mathcal{N}(0, 0.02)$
- Uniform distribution: $\mathcal{U}(-1/\sqrt{d_{model}}, 1/\sqrt{d_{model}})$
- The exact initialization matters less as training proceeds, but poor initialization can slow early training

**Padding token embedding:**

- The PAD token's embedding should not influence computation
- Some implementations initialize PAD's embedding to all zeros and freeze it
- Others rely on attention masks to exclude padding from computations

**Numerical precision:**

- Embedding lookup does not involve floating point arithmetic (it is just indexing), so it is numerically exact
- The scaling multiplication introduces floating point considerations, but is straightforward

---

## The Distributional Hypothesis

The theoretical foundation for why embeddings work comes from the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings.

$$
\text{meaning}(w) \approx f(\text{contexts in which } w \text{ appears})
$$

Consider the word "cat" and the word "dog". Both tend to appear in contexts like:

- "The ___ sat on the mat"
- "She petted the ___"
- "The ___ chased the ball"

Because they share so many contexts, a model trained to predict words from their context (or vice versa) will naturally assign similar embeddings to "cat" and "dog".

This is why embedding lookup, despite being a simple table lookup, produces rich semantic representations. The table entries are shaped by the global patterns of word co-occurrence across the entire training corpus.

**Semantic directions in embedding space:**

Research has shown that consistent semantic directions emerge in embedding spaces:

- A "gender" direction: $\mathbf{e}_{queen} - \mathbf{e}_{king} \approx \mathbf{e}_{woman} - \mathbf{e}_{man}$
- A "tense" direction: $\mathbf{e}_{walked} - \mathbf{e}_{walk} \approx \mathbf{e}_{ran} - \mathbf{e}_{run}$
- A "country-capital" direction: $\mathbf{e}_{Paris} - \mathbf{e}_{France} \approx \mathbf{e}_{Berlin} - \mathbf{e}_{Germany}$

These directions are not engineered; they emerge spontaneously from training on large text corpora.

---

## Embedding Spaces Across Languages

In multilingual models, embeddings from different languages can share the same vector space:

- Words with similar meanings across languages end up near each other
- This enables **zero-shot cross-lingual transfer**: train a model on English data, and it can work on French data because the French embeddings occupy similar regions of the space

The key requirement is that the model sees enough parallel or comparable text across languages during training to align the embedding spaces.

---

## Contextual vs. Static Embeddings

The embedding lookup layer produces **static embeddings**: each token always maps to the same vector, regardless of context.

- The word "bank" gets the same embedding whether it means a financial institution or a river bank
- The word "play" gets the same embedding whether it is a noun or a verb

However, once these static embeddings pass through the Transformer layers (attention + feed-forward), they become **contextual embeddings**. The attention mechanism allows each token's representation to incorporate information from surrounding tokens, resolving ambiguity.

$$
\mathbf{h}_{\text{bank}}^{(0)} = W_E[\text{bank}] \cdot \sqrt{d_{model}} \quad \text{(static, same for all contexts)}
$$

$$
\mathbf{h}_{\text{bank}}^{(L)} = \text{TransformerLayers}(\mathbf{h}^{(0)}) \quad \text{(contextual, different for each context)}
$$

The embedding layer provides the starting point, and the Transformer layers refine it into context-dependent representations. This division of labor is a core principle of the architecture.

---

## Connection to the Transformer Pipeline

The embedding layer sits at the very beginning of the Transformer:

$$
\text{Text} \xrightarrow{\text{tokenize}} \text{Token IDs} \xrightarrow{\text{embed + scale}} \text{Vectors} \xrightarrow{+ \text{PE}} \text{Input to Encoder}
$$

After the embedding layer:

1. Positional encodings are added to give the model position information
2. The result enters the first encoder (or decoder) layer
3. From this point forward, everything operates on continuous vectors of dimension $d_{model}$

The embedding layer is the last place where the discrete nature of language is visible. Once tokens become vectors, the Transformer operates in a purely continuous mathematical space.
