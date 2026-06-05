import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    filtered = df[df[column] > threshold]
    return {
        "filtered_data": filtered.to_dict("list"),
        "count": len(filtered)
    }
