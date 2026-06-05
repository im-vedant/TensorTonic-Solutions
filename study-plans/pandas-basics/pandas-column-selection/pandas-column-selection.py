import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    col = df[column]
    return {
        "values": col.tolist(),
        "length": len(col)
    }
