import pandas as pd

def multi_agg(data, group_col, value_col, funcs):
    """
    Returns: dict mapping function name to {group: value} dict
    """
    df = pd.DataFrame(data)
    result = df.groupby(group_col)[value_col].agg(funcs)
    return result.to_dict()
