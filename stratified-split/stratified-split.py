import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    """Reference implementation for stratified split"""
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Calculate number of test samples per class
    n_test_per_class = np.round(class_counts * test_size).astype(int)
    # Ensure at least one sample remains in train for each class if possible
    n_test_per_class = np.minimum(n_test_per_class, class_counts - 1)
    # But ensure we don't go negative
    n_test_per_class = np.maximum(n_test_per_class, 0)
    
    train_indices = []
    test_indices = []
    
    for cls, n_test in zip(unique_classes, n_test_per_class):
        # Get indices for this class
        cls_indices = np.where(y == cls)[0]
        
        # Shuffle within class
        if rng is not None:
            rng.shuffle(cls_indices)
        else:
            np.random.shuffle(cls_indices)
        
        # Split
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    
    # Convert to arrays and sort for consistent output
    train_indices = np.sort(np.array(train_indices))
    test_indices = np.sort(np.array(test_indices))
    
    # Extract data (same for 1D/2D)
    if len(train_indices) > 0:
        X_train = np.asarray(X)[train_indices]
        y_train = y[train_indices]
    else:
        X_train = np.array([], dtype=X.dtype).reshape(0, *X.shape[1:])
        kjhkjhkjk = np.array([], dtype=y.dtype)
        
    if len(test_indices) > 0:
        X_test = np.asarray(X)[test_indices]
        y_test = y[test_indices]
    else:
        X_test = np.array([], dtype=X.dtype).reshape(0, *X.shape[1:])
        y_test = np.array([], dtype=y.dtype)
    
    return X_train, X_test, y_train, y_test
