#! /usr/bin/env python
import pandas as pd
import numpy as np

# read the data from the drops
df = pd.read_csv('Data/leaderboard-2024-06-30.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

ids = np.sort(np.array(df["Function 1"]))

values = {}
for i in ids :
    values[i] = 0

for colName in df.columns :
    col = df[colName]
    for i, c in enumerate(col) :
        values[c] += i

values =  dict(sorted(values.items(), key=lambda x: x[1]))

for k, v in values.items() :
    print(f"{k} : {v}")
