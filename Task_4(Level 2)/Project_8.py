# UNVEILING THE ANDROID APP MARKET: GOOGLE PLAY STORE ANALYSIS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# LOAD DATASETS
app_df = pd.read_csv("app_data(8).csv")
user_df = pd.read_csv("user_data.csv")
print("App Data Shape:", app_df.shape)
print("User Review Shape:", user_df.shape)
print("\nApp Columns:", app_df.columns)
print("\nUser Columns:", user_df.columns)

# DATA CLEANING & PREPARATION
if "Unnamed: 0" in app_df.columns:
    app_df.drop(columns=["Unnamed: 0"], inplace=True)
app_df["Rating"] = pd.to_numeric(app_df["Rating"], errors="coerce")

app_df["Reviews"] = pd.to_numeric(app_df["Reviews"], errors="coerce")

app_df["Installs"] = app_df["Installs"].astype(str).str.replace("[+,]", "", regex=True)
app_df["Installs"] = pd.to_numeric(app_df["Installs"], errors="coerce")

app_df["Price"] = app_df["Price"].astype(str).str.replace("$", "", regex=False)
app_df["Price"] = pd.to_numeric(app_df["Price"], errors="coerce")

def convert_size(size):
    if "M" in size:
        return float(size.replace("M", ""))
    elif "k" in size:
        return float(size.replace("k", "")) / 1024
    else:
        return np.nan

app_df["Size_MB"] = app_df["Size"].astype(str).apply(convert_size)

app_df["Rating"].fillna(app_df["Rating"].mean(), inplace=True)
app_df.fillna("Unknown", inplace=True)

user_df["Sentiment"].fillna("Unknown", inplace=True)
user_df["Translated_Review"].fillna("No Review", inplace=True)
user_df["Sentiment_Polarity"].fillna(0, inplace=True)
user_df["Sentiment_Subjectivity"].fillna(0, inplace=True)

print("\n    AFTER CLEANING APP DATA:     ")
print(app_df.head())
print("\n    AFTER CLEANING USER DATA:     ")
print(user_df.head())

app_df.to_csv("cleaned_app_data.csv", index=False)
user_df.to_csv("cleaned_user_data.csv", index=False)
print("\nFiles Saved:")
print("✔ cleaned_app_data.csv")
print("✔ cleaned_user_data.csv")

# CATEGORY EXPLORATION
print("\nTop 10 Categories by App Count:")
cat_count = app_df["Category"].value_counts().head(10)
print(cat_count)

plt.figure(figsize=(12,6))
sns.barplot(x=cat_count.values, y=cat_count.index)
plt.title("Top App Categories")
plt.xlabel("Number of Apps")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# 4. METRICS ANALYSIS (Ratings, Installs, Price, Size)
plt.figure(figsize=(10,5))
sns.histplot(app_df["Rating"], kde=True, bins=40)
plt.title("Ratings Distribution")
plt.xlabel("Ratings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x=app_df["Installs"])
plt.title("App Install Distribution")
plt.tight_layout()
plt.show()
paid_apps = app_df[app_df["Price"] > 0]

plt.figure(figsize=(10,5))
sns.scatterplot(data=paid_apps, x="Price", y="Rating")
plt.title("Price vs Rating (Paid Apps)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(data=app_df, x="Size_MB", y="Rating")
plt.title("Size vs Rating")
plt.tight_layout()
plt.show()

# 5. SENTIMENT ANALYSIS
sent_count = user_df["Sentiment"].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(x=sent_count.index, y=sent_count.values)
plt.title("Sentiment Distribution of User Reviews")
plt.xlabel("Sentiment Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(user_df["Sentiment_Polarity"], kde=True)
plt.title("Sentiment Polarity Distribution")
plt.tight_layout()
plt.show()

# INTERACTIVE VISUALIZATIONS
fig = px.bar(
    cat_count,
    title="Interactive: Top App Categories",
    labels={"value": "Number of Apps", "index": "Category"}
)
fig.show()

fig2 = px.scatter(
    app_df,
    x="Installs",
    y="Rating",
    size="Reviews",
    color="Category",
    title="Interactive: Installs vs Rating"
)
fig2.show()
print("\n ANALYSIS COMPLETED SUCCESSFULLY!")
