## What Is Tokenization?

Tokenization is the process of converting raw text into a sequence of discrete symbols that a neural network can process. It is the very first step in any natural language processing pipeline, and its design has profound effects on everything downstream: vocabulary size, sequence length, model capacity, and generalization ability.

A neural network cannot operate on raw strings of characters. It requires numerical inputs, typically integer indices that map into an embedding table. Tokenization provides this mapping: from human-readable text to machine-readable integers, and back again.

---

## Why Tokenization Matters

The choice of tokenization strategy affects nearly every aspect of a language model:

- **Vocabulary size** determines the size of the embedding matrix and the output projection layer. A vocabulary of $V$ tokens with embedding dimension $d$ requires $V \times d$ parameters just for the input embeddings alone.

- **Sequence length** determines the computational cost of attention, which scales quadratically with sequence length. Finer-grained tokenization (e.g., character-level) produces longer sequences, increasing both memory and compute.

- **Out-of-vocabulary handling** determines what happens when the model encounters a word it has never seen. A tokenizer that cannot represent new words will map them all to a single unknown token, losing all information about them.

- **Semantic granularity** determines how much meaning each token carries. Whole words carry rich semantics, individual characters carry almost none, and subwords sit in between.

The tension between these factors is the central design challenge of tokenization.

---

## Word-Level Tokenization

The simplest approach splits text on whitespace (and optionally punctuation) to produce one token per word.

**Building the vocabulary:**

Given a collection of training texts, word-level tokenization:

1. Splits each text into individual words
2. Collects all unique words across the entire corpus
3. Assigns each unique word a unique integer ID
4. Adds special tokens for structural purposes

**The vocabulary** is a bidirectional mapping:
- A **word-to-ID** dictionary that converts words into integers during encoding
- An **ID-to-word** dictionary that converts integers back to words during decoding

$$
\text{word\_to\_id}: \text{String} \rightarrow \mathbb{Z}
$$

$$
\text{id\_to\_word}: \mathbb{Z} \rightarrow \text{String}
$$

These two mappings must be consistent inverses of each other: for every word $w$ in the vocabulary:

$$
\text{id\_to\_word}(\text{word\_to\_id}(w)) = w
$$

---

## Special Tokens

Every tokenizer needs a set of special tokens that serve structural roles beyond representing words. These are typically assigned the lowest integer IDs in the vocabulary:

**Padding token (PAD):**

- Purpose: Makes all sequences in a batch the same length
- Typically assigned ID 0
- When sequences have different lengths, shorter ones are padded with this token
- The model learns to ignore padding positions (via attention masks or by learning that PAD carries no information)

**Unknown token (UNK):**

- Purpose: Represents any word not found in the vocabulary
- Assigned during vocabulary construction, typically ID 1
- During encoding, any word that does not appear in the word-to-ID mapping is replaced with the UNK token
- This is the "safety net" that prevents errors when encountering unseen words, but it loses all information about what the original word was

**Beginning-of-sequence token (BOS):**

- Purpose: Marks the start of a sequence
- Gives the model an explicit signal that a new sequence has begun
- In language generation, the model is often given just the BOS token and asked to generate the rest

**End-of-sequence token (EOS):**

- Purpose: Marks the end of a sequence
- Tells the model that the sequence is complete
- In generation, the model produces tokens until it generates the EOS token
- In encoder models, it can serve as a summary position

Special tokens are added to the vocabulary before any real words, ensuring they always have the same IDs regardless of the training corpus.

---

## The Encoding Process

Encoding converts a string of text into a list of integer IDs:

$$
\text{encode}: \text{String} \rightarrow [z_1, z_2, \ldots, z_n]
$$

where each $z_i \in \{0, 1, \ldots, V-1\}$ is a valid token ID.

The process involves:

1. **Normalization**: Convert the text to a standard form (e.g., lowercasing)
2. **Splitting**: Break the text into individual words
3. **Lookup**: Map each word to its ID using the word-to-ID dictionary
4. **Unknown handling**: If a word is not in the vocabulary, replace it with the UNK token ID

**Example:**

Given vocabulary: $\{\text{PAD}: 0, \text{UNK}: 1, \text{BOS}: 2, \text{EOS}: 3, \text{cat}: 4, \text{sat}: 5, \text{the}: 6\}$

- "the cat sat" encodes to $[6, 4, 5]$
- "the dog sat" encodes to $[6, 1, 5]$ (dog is unknown, maps to UNK)

---

## The Decoding Process

Decoding is the inverse of encoding: it converts a list of integer IDs back into human-readable text:

$$
\text{decode}: [z_1, z_2, \ldots, z_n] \rightarrow \text{String}
$$

The process involves:

1. **Lookup**: Map each integer ID to its corresponding word using the ID-to-word dictionary
2. **Joining**: Concatenate the words with spaces between them

**The roundtrip property:**

For any text $t$ that contains only known vocabulary words:

$$
\text{decode}(\text{encode}(t)) = t
$$

However, this property breaks when unknown words are present. If the original text contained "the dog sat" and "dog" is not in the vocabulary, encoding produces $[6, 1, 5]$, and decoding produces "the UNK sat", losing the original word.

---

## Vocabulary Construction

Building the vocabulary is a critical design decision. The key questions are:

**How to split text into words?**

- **Whitespace splitting**: The simplest approach. Split on spaces, tabs, and newlines. Words like "don't" stay as one token.
- **Punctuation splitting**: Separate punctuation from words. "hello!" becomes ["hello", "!"].
- **Lowercasing**: Convert everything to lowercase. Reduces vocabulary size but loses case information (e.g., "Apple" the company vs "apple" the fruit).

**What order for the vocabulary?**

- Special tokens always come first (PAD, UNK, BOS, EOS)
- Remaining words can be sorted alphabetically (deterministic ordering) or by frequency (most common words get smaller IDs)

**What vocabulary size?**

- A corpus of English text might contain 100,000+ unique words
- Many of these are rare: misspellings, proper nouns, technical jargon
- Including all of them creates a very large embedding matrix
- Common practice is to keep only the top $V$ most frequent words and map everything else to UNK

---

## Worked Example: Building a Complete Tokenizer

**Training corpus:**

- "the cat sat on the mat"
- "the dog chased the cat"

**Step 1: Add special tokens**

- PAD $\rightarrow$ 0
- UNK $\rightarrow$ 1
- BOS $\rightarrow$ 2
- EOS $\rightarrow$ 3

**Step 2: Split and collect unique words**

All words (lowercased): the, cat, sat, on, the, mat, the, dog, chased, the, cat

Unique words (sorted alphabetically): cat, chased, dog, mat, on, sat, the

**Step 3: Assign IDs to words**

- PAD $\rightarrow$ 0
- UNK $\rightarrow$ 1
- BOS $\rightarrow$ 2
- EOS $\rightarrow$ 3
- cat $\rightarrow$ 4
- chased $\rightarrow$ 5
- dog $\rightarrow$ 6
- mat $\rightarrow$ 7
- on $\rightarrow$ 8
- sat $\rightarrow$ 9
- the $\rightarrow$ 10

Vocabulary size $V = 11$.

**Step 4: Encode a sentence**

"the cat sat on the mat" $\rightarrow$ [10, 4, 9, 8, 10, 7]

**Step 5: Encode with unknown word**

"the bird sat" $\rightarrow$ [10, 1, 9] (bird is not in vocabulary, maps to UNK=1)

**Step 6: Decode**

$[10, 4, 9] \rightarrow$ "the cat sat"

---

## The Out-of-Vocabulary Problem

Word-level tokenization has a fundamental limitation: it cannot represent words it has not seen during training. This is known as the **out-of-vocabulary (OOV) problem**.

Consider a tokenizer trained on news articles encountering medical text for the first time. Words like "immunoglobulin" or "thrombocytopenia" would all become UNK, losing crucial information.

The severity of this problem depends on the application:

- **Closed-domain** applications (e.g., customer service chatbots with a fixed set of topics) have fewer OOV words
- **Open-domain** applications (e.g., general-purpose language models) encounter OOV words constantly
- **Multilingual** applications face an explosion of vocabulary across languages
- **Code** and **scientific text** contain enormous numbers of rare tokens (variable names, chemical formulas, mathematical notation)

**Quantifying the problem:**

In English, a vocabulary of 30,000 words covers approximately 95% of typical text. But the remaining 5% often carries the most important information: proper nouns, technical terms, and novel words.

---

## Beyond Word-Level: Subword Tokenization

The OOV problem motivated the development of subword tokenization methods, which split rare words into smaller, more common pieces while keeping frequent words intact.

**Byte-Pair Encoding (BPE):**

BPE starts with a character-level vocabulary and iteratively merges the most frequent adjacent pairs:

1. Initialize vocabulary with all individual characters
2. Count all adjacent character pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 for a predetermined number of merges

**Example of BPE merges:**

Starting with characters: l, o, w, e, r, s, t, n

- Most frequent pair: (l, o) $\rightarrow$ merge into "lo"
- Next: (lo, w) $\rightarrow$ merge into "low"
- Next: (low, e) $\rightarrow$ merge into "lowe"
- Next: (lowe, r) $\rightarrow$ merge into "lower"

Common words like "lower" become single tokens, while rare words like "lowest" split into "lower" + "st".

**WordPiece:**

Similar to BPE but uses a slightly different merging criterion. Instead of frequency, WordPiece merges the pair that maximizes the likelihood of the training data:

$$
\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}
$$

This tends to merge pairs where the combination is more informative than its individual parts.

WordPiece marks continuation tokens with a special prefix (e.g., "##" in BERT):

- "playing" might tokenize as ["play", "##ing"]
- "unhappiness" might tokenize as ["un", "##happy", "##ness"]

**SentencePiece:**

Treats the input as a raw character stream (including spaces) and applies BPE or unigram language model segmentation. This makes it language-agnostic and handles languages without clear word boundaries (like Chinese and Japanese).

---

## Vocabulary Size Trade-offs

The vocabulary size $V$ is one of the most important hyperparameters in a language model:

**Small vocabulary** ($V \approx 8{,}000 - 16{,}000$):

- More subword splits per word, producing longer sequences
- Better handling of rare and unseen words (they decompose into known subwords)
- Smaller embedding matrix, fewer parameters
- But: longer sequences mean higher computational cost in attention

**Large vocabulary** ($V \approx 50{,}000 - 100{,}000+$):

- Most words are single tokens, producing shorter sequences
- Faster inference due to shorter sequences
- But: larger embedding matrix, more parameters to learn
- Rare tokens may have poor embeddings due to insufficient training examples

**The sweet spot** in practice:

- Original Transformer: ~37,000 tokens
- BERT: 30,522 tokens (WordPiece)
- GPT-2: 50,257 tokens (BPE)
- GPT-4: ~100,000 tokens (BPE)
- LLaMA: 32,000 tokens (SentencePiece)

The trend in modern models is toward larger vocabularies, enabled by more training data that provides sufficient examples for each token.

---

## The Embedding Connection

Tokenization feeds directly into the embedding layer. After tokenization converts text to integer IDs, the embedding layer converts those IDs into dense vectors:

$$
\text{Text} \xrightarrow{\text{tokenize}} [z_1, z_2, \ldots, z_n] \xrightarrow{\text{embed}} [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n]
$$

where each $\mathbf{e}_i \in \mathbb{R}^{d_{model}}$.

The vocabulary size determines the number of rows in the embedding matrix $W_E \in \mathbb{R}^{V \times d_{model}}$:

- Token ID $z_i$ selects row $z_i$ from $W_E$
- This is equivalent to multiplying a one-hot vector by the embedding matrix

$$
\mathbf{e}_i = W_E[z_i] = \mathbf{1}_{z_i}^T W_E
$$

where $\mathbf{1}_{z_i}$ is a one-hot vector with a 1 at position $z_i$.

---

## Weight Tying

The original Transformer paper introduced an important trick: **sharing weights between the embedding layer and the output projection layer**.

In a language model, the output layer maps from hidden representations back to vocabulary-size logits:

$$
\text{logits} = hW_{out} + b
$$

where $W_{out} \in \mathbb{R}^{d_{model} \times V}$.

With weight tying:

$$
W_{out} = W_E^T
$$

This reduces parameters by $V \times d_{model}$ and forces the model to use consistent representations: words that are close in embedding space also compete for similar output probabilities.

---

## Normalization and Preprocessing

Before splitting text into tokens, most tokenizers apply normalization:

- **Lowercasing**: "The" and "the" become the same token. Reduces vocabulary size but loses case information.
- **Unicode normalization**: Ensures consistent encoding of characters across different systems (NFC, NFKC forms)
- **Whitespace normalization**: Collapse multiple spaces into one, trim leading/trailing spaces
- **Special character handling**: Decide whether to keep or remove punctuation, emojis, and special characters

These choices are permanent and cannot be reversed. A tokenizer that lowercases cannot distinguish "apple" (fruit) from "Apple" (company).

---

## Tokenization for Different Modalities

While text tokenization is the most common, the concept extends to other domains:

**Vision Transformers (ViT):**

Images are "tokenized" by splitting them into fixed-size patches (e.g., $16 \times 16$ pixels). Each patch becomes a token, and a linear projection serves as the embedding.

$$
\text{Image} \xrightarrow{\text{patch}} [p_1, p_2, \ldots, p_N] \xrightarrow{\text{project}} [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_N]
$$

**Audio Transformers:**

Audio waveforms are split into short frames (e.g., 25ms windows) and converted to spectral features. Each frame becomes a token.

**Protein Language Models:**

Amino acid sequences are tokenized character-by-character, with each of the 20 standard amino acids becoming a token.

The Transformer architecture is agnostic to the modality. All it needs is a sequence of discrete tokens with embeddings.

---

## Handling Batches and Padding

Real training happens in batches, and sequences within a batch often have different lengths. Padding resolves this:

**Example:**

Sequences: "the cat" (2 tokens), "the dog sat" (3 tokens)

After padding to the maximum length:

$$
\begin{aligned}
\text{Sequence 1}: &\quad [10, 4, 0] \\
\text{Sequence 2}: &\quad [10, 6, 9]
\end{aligned}
$$

The PAD token (ID 0) fills the gap.

**Attention masks** indicate which positions contain real tokens (1) and which are padding (0):

$$
\begin{aligned}
\text{Mask 1}: &\quad [1, 1, 0] \\
\text{Mask 2}: &\quad [1, 1, 1]
\end{aligned}
$$

The model uses these masks to prevent attention from attending to padding positions, ensuring padding tokens do not influence the computation.

---

## Determinism and Reproducibility

A well-designed tokenizer must be deterministic:

- The same input text must always produce the same token IDs
- The same token IDs must always decode to the same text
- The vocabulary must be built in a deterministic order (e.g., sorting words alphabetically)

Without determinism, models trained with one tokenizer cannot be evaluated with another, and saved models become unusable if the tokenizer changes.

This is why vocabulary files are saved alongside model weights: the tokenizer and the model are inseparable.

---

## Historical Context

The evolution of tokenization in NLP reflects the broader evolution of the field:

- **Early NLP (1990s-2000s)**: Word-level tokenization with large vocabularies. Stemming and lemmatization were used to reduce vocabulary size.
- **Word2Vec era (2013)**: Word-level tokenization with fixed vocabularies. OOV words handled by ignoring them or using random vectors.
- **BPE adoption (2016)**: Sennrich et al. applied BPE to neural machine translation, dramatically improving handling of rare words.
- **Transformer era (2017)**: The original Transformer used BPE with ~37K tokens.
- **BERT (2018)**: Introduced WordPiece tokenization with 30K tokens.
- **GPT-2 (2019)**: Used byte-level BPE, ensuring every possible byte sequence can be encoded (no UNK token needed).
- **Modern LLMs (2023+)**: Vocabularies of 100K+ tokens with byte-level BPE, optimized for multilingual coverage and efficiency.

The trend is clear: from simple word splitting to sophisticated subword algorithms that balance vocabulary size, sequence length, and language coverage.

---

## Summary of Key Concepts

- Tokenization maps text to integers and back, bridging human language and neural computation
- The vocabulary is a bidirectional mapping between tokens and integer IDs
- Special tokens (PAD, UNK, BOS, EOS) serve structural roles
- Word-level tokenization is simple but suffers from the OOV problem
- Subword methods (BPE, WordPiece, SentencePiece) address OOV by splitting rare words into common pieces
- Vocabulary size trades off between sequence length and parameter count
- The tokenizer and model are inseparable: changing one invalidates the other
