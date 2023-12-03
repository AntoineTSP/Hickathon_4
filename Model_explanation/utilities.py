import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def get_gscpi(filepath_gscpi):
    gscpi = pd.read_csv(filepath_gscpi, sep=",")
    gscpi['Year'] = gscpi['Year-Month'].apply(lambda x: x.split('-')[0]).astype(int)
    gscpi['Month'] = gscpi['Year-Month'].apply(lambda x: x.split('-')[1]).astype(int)
    gscpi.drop(columns=['Year-Month'], inplace=True)
    return gscpi

def get_mean_gscpi(gscpi, date):
    months, year = date.split(" ")
    year = int(year)
    gscpi_months = {
        'may-aug': [5, 6, 7, 8],
        'sep-dec': [9, 10, 11, 12],
        'jan-apr': [1, 2, 3, 4],
        'may-jul': [5, 6, 7]
    }
    return gscpi.loc[(gscpi.Year == year) & (gscpi.Month.isin(gscpi_months[months]))].GSCPI.mean()

def add_gscpi_to_df(df, gscpi):
    df['gscpi'] = df['Date'].apply(lambda date: get_mean_gscpi(gscpi, date))

def to_int(x):
    if pd.isna(x):
        return x
    if isinstance(x, int):
        return x
    return int(x.replace(" ", ""))


FEATURES_TO_KEEP = ['Site', 'Reference proxy', 'Customer Persona proxy',
                    'Strategic Product Family proxy', 'Date', 'Month 1',
                    'Month 2', 'Month 3', 'gscpi']


def wrangle_test_data(X_test, gscpi, features_to_keep=FEATURES_TO_KEEP):
    X_test.fillna(0, inplace=True)
    add_gscpi_to_df(X_test, gscpi)
    
    for j in range(1, 5):
        X_test[f'Month {j}'] = X_test[f'Month {j}'].apply(lambda x: to_int(x))
    
    X_test = X_test[features_to_keep]
    
    index_fst_not_encoded = list(X_test.columns).index('Month 1')
    real_test_not_encoded = real_test[real_test.columns[index_fst_not_encoded:]]
    X_test = enc.transform(X_test[X_test.columns[:index_fst_not_encoded]])
    X_test = np.hstack((X_test, real_test_not_encoded))
    
    return X_test

# Example usage:
# gscpi = get_gscpi("../datasets/extra-dataset/GSCPI_data.csv")
# X_test = pd.read_csv("../datasets/X_test_working.csv")
# real_test = # Replace this line with your actual real_test data
# # Assuming 'enc' is defined somewhere in your code
# result = wrangle_test_data(X_test, gscpi)
