import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/datasetWithoutClassPrice.csv')

min = 32743
max = 556637
inc = (max-min)/10
col = 'SalePrice'
conditions = [(df[col] >= min) & (df[col] <= min+(inc*1)),
              (df[col] > min+(inc*1)) & (df[col] <= min+(inc*2)),
              (df[col] > min+(inc*2)) & (df[col] <= min+(inc*3)),
              (df[col] > min+(inc*3)) & (df[col] <= min+(inc*4)),
              (df[col] > min+(inc*4)) & (df[col] <= min+(inc*5)),
              (df[col] > min+(inc*5)) & (df[col] <= min+(inc*6)),
              (df[col] > min+(inc*6)) & (df[col] <= min+(inc*7)),
              (df[col] > min+(inc*7)) & (df[col] <= min+(inc*8)),
              (df[col] > min+(inc*8)) & (df[col] <= min+(inc*9)),
              (df[col] > min+(inc*9)) & (df[col] <= min+(inc*10)),
              (df[col] > min+(inc*10))]
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "X"]
df["class_price"] = np.select(conditions, choices, default=np.nan)

df.to_csv('../dataset/dataset.csv', index=False, float_format='%.2f')


