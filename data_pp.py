#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from utils import corr_plot
import missingno as mno

#%%

data = pd.read_csv('./data/HWW.csv', delimiter=';')
data.drop(columns=['Date'], inplace=True)
# for i, val in enumerate(data.isna().sum()):
#     print(data.columns[i], ':',round(val/len(data), 2) * 100 , '% is Nan val')

corr_plot(data, title="Original HWW Data")

nan_indices_per_column={}
for col in data.columns:
    nan_indices_per_column[col] = set(data[data[col].isna()].index.tolist())

common_nan_indices = set.intersection(*nan_indices_per_column.values())
common_nan_indices = sorted(list(common_nan_indices))

mask = ~data.index.isin(common_nan_indices)
data_masked = data[mask]

print('++++++++++++++++++++++++++++++++++++++++++++++++++', '\n',
      'After removing common Nan indices over all columns', '\n',
      '++++++++++++++++++++++++++++++++++++++++++++++++++')

for i, val in enumerate(data_masked.isna().sum()):
    print(data_masked.columns[i], ':',round(val/len(data_masked), 2) * 100 , '% is Nan val')

# %%

data_masked_wn = data_masked[data_masked.columns[~data_masked.isna().any()]]

corr_plot(data_masked_wn, 'columns w/o nan values')
plt.figure(figsize=(20, 10))
# plt.plot(data_masked['MW437'])
# plt.plot(data_masked[['MW438', 'MW437']][::1000])
corr_matrix = data_masked.corr()
mno.matrix(data, figsize=(20,8))
#%%

imputer = IterativeImputer(estimator=BayesianRidge())
imputed_data = imputer.fit_transform(data_masked[list(data_masked_wn.columns) + list(data_masked.columns[data_masked.isna().any()])])
#%%
data_masked[list(data_masked.columns[data_masked.isna().any()])] = imputed_data[:, -10:]
# %%
corr_plot(data_masked, "Corr After RidgeRegression")

# %%
data_masked.to_csv("./data/HWW_imputed.csv")
# %%
