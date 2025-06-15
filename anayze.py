import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Adult Income Dataset Analysis")

# Load the Adult Income dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)
    return df

df = load_data()

# Display dataset overview
st.header("Dataset Overview")
st.write("First 5 rows of the Adult Income dataset:")
st.dataframe(df.head())

# Basic statistics
st.header("Basic Statistics")
st.write("Summary statistics of numerical columns:")
st.dataframe(df.describe())

# Income distribution
st.header("Income Distribution")
income_counts = df["income"].value_counts()
st.write("Number of individuals by income level (<=50K, >50K):")
st.bar_chart(income_counts)

# Feature analysis
st.header("Feature Analysis")
numeric_features = ["age", "education-num", "hours-per-week"]
feature = st.selectbox("Select a numerical feature to analyze:", numeric_features)
fig, ax = plt.subplots()
df[feature].hist(ax=ax, bins=20)
ax.set_title(f"Distribution of {feature}")
ax.set_xlabel(feature)
ax.set_ylabel("Count")
st.pyplot(fig)

# Categorical feature analysis
st.header("Categorical Feature Analysis")
categorical_features = ["workclass", "education", "marital-status", "occupation", "sex", "race"]
cat_feature = st.selectbox("Select a categorical feature to analyze:", categorical_features)
fig, ax = plt.subplots()
df[cat_feature].value_counts().plot(kind="bar", ax=ax)
ax.set_title(f"Distribution of {cat_feature}")
ax.set_xlabel(cat_feature)
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# Income vs Feature
st.header("Income vs Feature")
feature = st.selectbox("Select a feature for income comparison:", numeric_features, key="income_feature")
fig, ax = plt.subplots()
for income in df["income"].unique():
    subset = df[df["income"] == income]
    ax.hist(subset[feature], bins=20, alpha=0.5, label=income)
ax.set_title(f"{feature} by Income Level")
ax.set_xlabel(feature)
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Correlation matrix
st.header("Correlation Matrix")
numeric_df = df[numeric_features]
corr_matrix = numeric_df.corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr_matrix, cmap="coolwarm")
fig.colorbar(cax)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticklabels(corr_matrix.columns)
st.pyplot(fig)
st.write("Correlation matrix of numerical features:")
st.dataframe(corr_matrix)