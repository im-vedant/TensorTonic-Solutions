import pandas as pd

def multi_groupby(data, group_cols, value_col, aggfunc):
    """
    Returns: dict of lists (flat table with group columns + value column)
    """
    df = pd.DataFrame(data)
    result = df.groupby(group_cols)[value_col].agg(aggfunc).reset_index()
    return result.to_dict("list")
