# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def generate_data():
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    return X,y,X_test

def fit(X,y):
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_1.fit(X, y)
    return regr_1

def predict(regr_1, X_test):
    # Predict
    y_1 = regr_1.predict(X_test)
    return y_1

def plot(X,y,X_test,y_1):
    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.savefig("./standard_plot.png", format="png")
    plt.show()

@st.cache_data
def get_chart(df):
    #fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
    fig = px.line(df, x="index", y="Month 4", title='Life expectancy in Canada')

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

@st.cache_data
def get_bar_chart(df):
    fig = px.bar(df, x="index", y="Month 4", title='Predicted sales next month')

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

@st.cache_data
def get_time_series(df):
    #fig = px.line(df, x='date', y="GOOG")
    fig = px.line(df)

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

def get_mean_gspci(date, gscpi):
    months, year = date.split(" ")
    year = int(year)
    gscpi_months = []
    if months == 'may-aug':
        gscpi_months = [5, 6, 7, 8]
    elif months == 'sep-dec':
        gscpi_months = [9, 10, 11, 12]
    elif months == 'jan-apr':
        gscpi_months = [1, 2, 3, 4]
    elif months == 'may-jul':
        gscpi_months = [5, 6, 7]
        
    return gscpi.loc[(gscpi.Year == year) & (gscpi.Month.isin(gscpi_months))].GSCPI.mean()

def add_gscpi_to_df(df, gscpi):
    df['gscpi'] = [0]*len(df)
    for date in df.Date.unique():
        df.loc[df.Date == date, 'gscpi'] = get_mean_gspci(date, gscpi)

def add_lpi_to_df(df):
    lpi_col_to_add = ['Customs Score', 'Logistics Competence and Quality Score', 'International Shipments Score']
    for col in lpi_col_to_add:
        df[col] = ['']*len(df)
    for country_code in df.Country.unique():
        lpi_country = lpi.loc[lpi.Country_code == country_code]
        try:
            for col in lpi_col_to_add:
                df.loc[df.Country == country_code, col] = lpi_country[col].iloc[0]
        except:
            print(country_code)

def scores(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    pearson, _ = pearsonr(y, y_pred)
    return mse, np.sqrt(mse), mae, r2, pearson

def print_scores(model):
    train_scores = scores(model, X_train, y_train)
    test_scores = scores(model, X_test, y_test)
    print(f"Train scores: MSE={train_scores[0]}, RMSE={train_scores[1]}, MAE={train_scores[2]}, R2={train_scores[3]}, Pearson={train_scores[4]}")
    print(f"Test scores: MSE={test_scores[0]}, RMSE={test_scores[1]}, MAE={test_scores[2]}, R2={test_scores[3]}, Pearson={test_scores[4]}")

def estim_score_hfactory(model):
    r0 = np.sqrt(mean_squared_error(y_test, [0]*len(y_test)))
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return (r0 - 0.8*rmse) / r0

def to_int(x):
    if pd.isna(x):
        return x
    if isinstance(x, int) or isinstance(x, float):
        return x
    return int(x.replace(" ", ""))