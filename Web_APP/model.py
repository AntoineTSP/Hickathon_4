# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px

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
    fig = px.line(df, title='Life expectancy in Canada')

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

