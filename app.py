
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price EDA Dashboard",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price EDA Dashboard")
st.subheader("Exploratory Analysis of the Ames Housing Dataset")

st.write(
    "Explore house prices, quality, neighborhoods, "
    "correlations and outliers using interactive visualizations."
)

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload train.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Houses", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric(
            "Average Sale Price",
            f"${df['SalePrice'].mean():,.0f}"
        )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.header("2. House Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["SalePrice"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of House Sale Prices")
    ax.set_xlabel("Sale Price")
    st.pyplot(fig)
    plt.close(fig)

    st.header("3. Sale Price by Overall Quality")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, ax=ax)
    ax.set_title("House Price by Overall Quality")
    st.pyplot(fig)
    plt.close(fig)

    st.header("4. Average Price by Neighborhood")

    neighborhood = (
        df.groupby("Neighborhood")["SalePrice"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(neighborhood)

    st.header("5. Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    top_features = (
        corr["SalePrice"]
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr.loc[top_features, top_features],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Top 10 Features Correlated with SalePrice")
    st.pyplot(fig)
    plt.close(fig)

    st.header("6. Outlier Analysis")

    q1 = df["SalePrice"].quantile(0.25)
    q3 = df["SalePrice"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df["SalePrice"] < lower) |
        (df["SalePrice"] > upper)
    ]

    st.metric("Potential Sale Price Outliers", len(outliers))

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=df["SalePrice"], ax=ax)
    ax.set_title("Sale Price Outlier Detection")
    st.pyplot(fig)
    plt.close(fig)

    st.header("7. Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col2:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.subheader("Missing Values by Column")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.dataframe(missing.rename("Missing Count"))

    st.header("8. Final Insights")

    st.write("""
    - House prices are positively skewed.
    - Overall quality has a strong positive relationship with sale price.
    - Living area is another important feature related to house prices.
    - Average house prices vary across neighborhoods.
    - Some properties have unusually high sale prices.
    """)

    st.success("Thank you for exploring the House Price EDA Dashboard!")

else:
    st.info("Please upload the Kaggle train.csv file from the sidebar to explore the dashboard.")

st.sidebar.markdown("---")
st.sidebar.caption("House Price EDA Project | Python & Streamlit")
