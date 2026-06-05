import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    frames = [pd.DataFrame(d) for d in dfs]
    result = pd.concat(frames, ignore_index=True)
    return [list(result.shape), result.to_dict("list")]
