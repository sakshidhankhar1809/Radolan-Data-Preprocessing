#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:09:14 2024

@author: mahyar
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# d = 25
PATH = f'./data/goslar/total.csv'
PATH_arman = '../Downloads/Data.csv'
#%%
# total = []
# for i in os.listdir(PATH):
#     data = np.load(os.path.join(PATH, i))['data']
#     total.append(data)

# total = np.array(total)
# # total.sum(axis=(1,2))

data = pd.read_csv(PATH, index_col='Date-Time')

data.drop(columns=['Unnamed: 0'], axis=0,  inplace=True)
data.index.name = None

# flood = data[(data.index > '2017-07-25') & (data.index < '2017-07-27')]
# f_sum = flood.sum(axis=1)


# fig, ax = plt.subplots(1,1, figsize=(20, 8))
# ax.plot(f_sum[::2])
# plt.xticks(rotation=90)

#%%
# data_a = pd.read_csv(PATH_arman, index_col='Zeit')
# data_a.index = pd.date_range('2003-11-01', '2018-06-30', freq='15min')
# flood_a = data_a[(data_a.index > '2017-07-25') & (data_a.index < '2017-07-27')]

#%%
data['sum'] = data.sum(axis=1)
# data['Image'] = data.iloc[:, :-1].to_numpy().reshape(-1, 7, 9)
# data.drop(data.columns.difference(['sum','Image']), 1, inplace=True)
# data['Image'] = data['Image'].to_numpy().reshape((7, 9))