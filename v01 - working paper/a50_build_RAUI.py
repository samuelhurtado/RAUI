import os
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from functools import reduce

# Load the dataframes and rename the variables adding a suffix

for i in range(1, 11):
    file_name = f"RAUI_mle5l_topic_{i}_gpt_montlyseries.xlsx"
    mycode = f"df{i} = pd.read_excel('{file_name}')"
    exec(mycode)
    if i>1:
        mycode = f"df{i}.rename(columns={{'sentimiento': 'sentiment', 'incertidumbre': 'uncertainty'}}, inplace=True)"
        exec(mycode)
    mycode = f"df{i} = df{i}[['year_month', 'item_count', 'sentiment', 'uncertainty']].copy()"
    exec(mycode)
    mycode = f"df{i}.rename(columns={{'item_count': 'item_count_{i}', 'sentiment': 'sentiment_{i}', 'uncertainty': 'uncertainty_{i}'}}, inplace=True)"
    exec(mycode)

# merge dataframes

dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

df = reduce(lambda left, right: pd.merge(left, right, on='year_month', how='outer'), dfs)

# create total count
item_count_columns = [f'item_count_{i}' for i in range(1, 11)]
df[item_count_columns] = df[item_count_columns].fillna(0)
df['total_count'] = df[item_count_columns].sum(axis=1)

# create shares (amplifier)
for i in range(1, 11):
    mycode = f"df['share_{i}'] = df['item_count_{i}'] / df['total_count']"
    exec(mycode)

# standardize sentiment and uncertainty indicators
for i in range(1, 11):

    sentiment_col = f"sentiment_{i}"
    df[sentiment_col] = (df[sentiment_col] - df[sentiment_col].mean()) / df[sentiment_col].std()
    
    uncertainty_col = f"uncertainty_{i}"
    df[uncertainty_col] = (df[uncertainty_col] - df[uncertainty_col].mean()) / df[uncertainty_col].std()

# calculate contributions (will normalize later)
for i in range(1, 11):

    sentiment_col = f"sentiment_{i}"
    share_col = f"share_{i}"
    RASI_col = f"RASIcontr_{i}"
    df[RASI_col] = df[sentiment_col] * df[share_col]
    
    uncertainty_col = f"uncertainty_{i}"
    RAUI_col = f"RAUIcontr_{i}"
    df[RAUI_col] = df[uncertainty_col] * df[share_col]

# calculate aggregate RASI and RAUI (will normalize later)

df['RASI'] = df[[f'RASIcontr_{i}' for i in range(1, 11)]].sum(axis=1)
df['RAUI'] = df[[f'RAUIcontr_{i}' for i in range(1, 11)]].sum(axis=1)

# normalize aggregate RASI and RAUI, and contributions (just divide by std so units can be interpreted easily)

for i in range(1, 11):
    RASI_col = f"RASIcontr_{i}"
    df[RASI_col] = df[RASI_col] / df['RASI'].std()
    df[RAUI_col] = df[RAUI_col] / df['RAUI'].std()

df['RASI'] = df['RASI'] / df['RASI'].std()
df['RAUI'] = df['RAUI'] / df['RAUI'].std()

# calculate topic RASI and RAUI

for i in range(1, 11):
    RASI_colc = f"RASIcontr_{i}"
    RASI_col = f"RASI_{i}"
    RAUI_colc = f"RAUIcontr_{i}"
    RAUI_col = f"RAUI_{i}"
    df[RASI_col] = df[RASI_colc] / df[RASI_colc].std()
    df[RAUI_col] = df[RAUI_colc] / df[RAUI_colc].std()


df.to_excel("RAUI_results.xlsx", index=False)

