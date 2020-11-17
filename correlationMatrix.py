import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("dataset/dataset.csv")

modeling_data = data.copy()
le = LabelEncoder()
modeling_data['class_price'] = le.fit(modeling_data['class_price']).transform(modeling_data['class_price'])


corr = modeling_data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig = plt.figure(figsize=(50, 50))
axes = fig.add_axes([0, 0.39, 0.8, 0.6])
ax = sns.heatmap(corr, cmap="Accent_r", ax=axes, mask=mask, center=0, vmin=-1,
                 vmax=1, square=True, linewidths=.2, cbar_kws={"shrink": .7})
plt.show()
sns.despine()
