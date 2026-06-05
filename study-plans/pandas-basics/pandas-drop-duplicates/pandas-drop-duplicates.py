import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)
    return [rows_before, rows_after, df.to_dict("list")]
