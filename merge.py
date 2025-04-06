from openai import OpenAI
import numpy as np
import json
import pandas
import time
import os
from dotenv import load_dotenv
df1 = pandas.read_csv('ruling.csv')
print(df1)
df2 = pandas.read_csv('new_ruling.csv')
df2 = df2.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0.2'])
print(df2)
frames = [df1, df2]
df3 = pandas.concat(frames)
df3.drop(columns=['Unnamed: 0',], inplace=True)
# df3.drop(df3.columns[0], inplace=True, axis=1)
print(df3)
# print(len(df3))
# print(df)
# print(len(df))
# df['Ruling'].replace('', np.nan, inplace=True)
# df.dropna(subset=['Ruling'], inplace=True)
# df['Application Type'].replace('', np.nan, inplace=True)
# df.dropna(subset=['Application Type'], inplace=True)
# df['File Number'].replace('', np.nan, inplace=True)
# df.dropna(subset=['File Number'], inplace=True)
# df['Tenant-Caused Damage'].replace('', np.nan, inplace=True)
# df.dropna(subset=['Tenant-Caused Damage'], inplace=True)
# print(len(df))
df3.to_csv('final_ruling.csv')
# print(len(res))
