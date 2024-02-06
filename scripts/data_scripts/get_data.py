#!/usr/bin/python3

import pandas as pd


url_train = 'https://raw.githubusercontent.com/Tatiana302/mlops_2-1_data/main/train.csv'
url_test = 'https://raw.githubusercontent.com/Tatiana302/mlops_2-1_data/main/test.csv'
train_df = pd.read_csv(url_train)
test_df = pd.read_csv(url_test)


train_df.to_csv('~/mlops_2-1/data/raw/train.csv', sep=',', index = False)
test_df.to_csv('~/mlops_2-1/data/raw/test.csv', sep=',', index = False)