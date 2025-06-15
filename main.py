import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(layout="centered")
st.title("🌍 COVID-19 Global Tracker")

@st.cache_data
def load_data():
    df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

countries = df['location'].unique()
selected_country = st.selectbox("Choose a Country", countries)

country_data = df[df['location'] == selected_country]


latest = country_data.sort_values("date", ascending=False).iloc[0]

st.subheader(f"📊 Latest COVID-19 Stats for {selected_country}")
st.write(f"**Date:** {latest['date'].date()}")
st.write(f"**Total Cases:** {int(latest['total_cases']) if pd.notnull(latest['total_cases']) else 'N/A'}")
st.write(f"**Total Deaths:** {int(latest['total_deaths']) if pd.notnull(latest['total_deaths']) else 'N/A'}")


fig, ax = plt.subplots()
country_data = country_data[country_data['total_cases'].notnull()]
ax.plot(country_data['date'], country_data['total_cases'], label='Total Cases')
ax.plot(country_data['date'], country_data['total_deaths'], label='Total Deaths', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title(f'COVID-19 Trend in {selected_country}')
ax.legend()
st.pyplot(fig)
