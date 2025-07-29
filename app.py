# app.py

import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="VOC-Aware Reasoning Explorer", layout="centered")

st.title("VOC-Aware Reasoning")
st.markdown("Explore when reasoning helps and when it's just token waste.")

with open("outputs/voc_scored.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Sidebar: pick a question
selected_id = st.sidebar.selectbox("Select a question:", df["id"])
entry = df[df["id"] == selected_id].iloc[0]

st.subheader("Question")
st.write(entry["question"])

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Direct Answer**")
    st.code(entry["direct"])

with col2:
    st.markdown("**Chain-of-Thought Answer**")
    st.code(entry["cot"])

st.markdown("---")

# VOC stats
st.subheader("VOC Analysis")

col3, col4, col5 = st.columns(3)
col3.metric("Utility", round(entry["utility"], 3))
col4.metric("Cost", round(entry["cost"], 3))
col5.metric("VOC", round(entry["voc"], 3))

st.markdown(
    f" **Recommended Strategy:** `{entry['strategy'].upper()}`"
)

# Optional: show raw data
with st.expander("See raw JSON"):
    st.json(entry)
