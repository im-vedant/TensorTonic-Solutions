import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vocab_idx = {word: i for i, word in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in vocab_idx:
            vec[vocab_idx[token]] += 1
    return vec