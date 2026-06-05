import pandas as pd

def handle_missing(data, fill_value):
    """
    Returns: dict with 'null_counts' (dict) and 'cleaned_data' (dict)
    """
    df = pd.DataFrame(data)
    null_counts = {k: int(v) for k, v in df.isnull().sum().items()}
    df = df.fillna(fill_value)
    return {
        "null_counts": null_counts,
        "cleaned_data": df.to_dict("list")
    }
