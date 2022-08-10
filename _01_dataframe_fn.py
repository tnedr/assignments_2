import pandas as pd


DF = pd.DataFrame(
    {'name': ['foo', 'bar', 'foo', 'foo', 'bar', 'foo', 'bar', 'bar'],
        'measure_1': [5, 35, 10, 15, 20, 25, 30, 12],
        'measure_2': [100, 500, 150, 25, 250, 300, 400, 200]})

# Write a function that outputs another dataframe with rows corresponding to
# the minimum value of measure_1 for each unique value of ‘name’.
def rows_for_minvalue_for_each_unique_name(df):
    df_g = df.groupby('name')
    s_idx_of_mins = df_g['measure_1'].idxmin()
    df_result = df.loc[s_idx_of_mins]
    return df_result
# print(rows_for_minvalue_for_each_unique_name(DF))
