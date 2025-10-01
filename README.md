import pandas as pd

# Load metadata.csv
df = pd.read_csv("metadata.csv")

# Peek at first few rows
print(df.head())
print(df.info())

# DataFrame dimensions
print("Shape:", df.shape)

# Data types
print(df.dtypes)

# Missing values per column
print(df.isnull().sum())

# Summary statistics for numerical columns
print(df.describe())


missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(10))

df_clean = df.dropna(subset=["title"])

df_clean["journal"].fillna("Unknown", inplace=True)

# Convert publish_time to datetime
df_clean["publish_time"] = pd.to_datetime(df_clean["publish_time"], errors="coerce")

# Extract year
df_clean["year"] = df_clean["publish_time"].dt.year

# Abstract word count
df_clean["abstract_word_count"] = df_clean["abstract"].fillna("").apply(lambda x: len(x.split()))
# Count papers by year
papers_per_year = df_clean["year"].value_counts().sort_index()

# Top journals
top_journals = df_clean["journal"].value_counts().head(10)

# Frequent words in titles
from collections import Counter
import re

titles = " ".join(df_clean["title"].dropna()).lower()
words = re.findall(r"\b\w+\b", titles)
common_words = Counter(words).most_common(20)
print(common_words)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Publications over time
papers_per_year.plot(kind="bar", figsize=(8,4), title="Publications by Year")
plt.show()

# Top journals
top_journals.plot(kind="bar", figsize=(8,4), title="Top Journals")
plt.show()

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Distribution by source_x
df_clean["source_x"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(6,6), title="Papers by Source")
plt.show()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Load data
df = pd.read_csv("metadata.csv")
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["year"] = df["publish_time"].dt.year

# Interactive filter
year_range = st.slider("Select year range", int(df["year"].min()), int(df["year"].max()), (2020, 2021))
filtered = df[df["year"].between(year_range[0], year_range[1])]

st.write(f"Showing {filtered.shape[0]} papers")

# Plot
papers_per_year = filtered["year"].value_counts().sort_index()
fig, ax = plt.subplots()
papers_per_year.plot(kind="bar", ax=ax)
st.pyplot(fig)

# Show sample
st.write(filtered.head())
streamlit run app.py
