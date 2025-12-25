import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD AND CLEAN DATASET
df=pd.read_csv("Retail_Data(1).csv")
print(" ------FIRST 10 ROWS -----------")
print(df.head())

df["Order Date"]=pd.to_datetime(df["Order Date"],format="%d-%m-%Y")

print("\n ------MISSING VALUES--------")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# DESCRIPTIVE STATISTICS
print("\n ------SUMMARY STATISTICS--------")
print(df.describe())
print("\n ------MODE-----")
print(df.mode().iloc[0])

# TIME SERIES ANALYSIS
df["Month"]=df["Order Date"].dt.to_period("M")
monthly_sales=df.groupby("Month")["Total Sales"].sum()

plt.figure(figsize=(12,5))
monthly_sales.plot(kind="line",marker="o")
plt.title("MONTHLY TOTAL SALES TREND")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# CUSTOMER & PRODUCT ANALYSIS
category_sales=df.groupby("Product Category")["Total Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
category_sales.plot(kind="bar")
plt.title("SALES BY PRODUCT CATEGORY")
plt.ylabel("Total Sales")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(data=df,x="Buyer Gender")
plt.title("BUYER GENDER DISTRIBUTION")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Buyer Age"], bins=20,kde=True)
plt.title("BUYER AGE DISTRIBUTION")
plt.show()

location_sales=df.groupby("Order Location")["Total Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
location_sales.head(10).plot(kind="bar")
plt.title("TOP 10 LOCATIONS BY SALES")
plt.ylabel("Total Sales")
plt.show()

# CORRELATION HEATMAP
numeric_cols=["Sales Price","Shipping Charges","Sales per Unit","Quantity","Total Sales","Buyer Age","Rating"]
plt.figure(figsize=(10,7))
sns.heatmap(df[numeric_cols].corr(),annot=True,cmap="coolwarm")
plt.title("CORRELATION HEATMAP")
plt.show()

# INSIGHTS
print("\n ============= INSIGHTS & RECOMMENDATIONS =============")

print("\n 1. *SALES TREND ANALYSIS:*")
print(" -PEAK SALES MONTHS:",monthly_sales.idxmax())
print("-LOWEST SALES MONTHS:", monthly_sales.idxmin())

print("\n 2. *TOP PRODUCT CATEGORIES:*")
print(category_sales.head())

print("\n 3. *CUSTOMER SEGMENT INSIGHTS:*")
print(" -DOMINANT BUYER GENDER:", df["Buyer Gender"].mode()[0])
print(" -AVERAGE BUYER AGE:", df["Buyer Age"].mean())

print("\n 4. *HIGH PERFORMING LOCATIONS:*")
print(location_sales.head(5))

print("\n 5. *RECOMMENDATIONS:*")
print(" Improve inventory of top-selling categories.")
print("Offer promotions in low-sales months.")
print("Target dominant age group with personalized ads.")
print("Focus marketing on top-performing cities.")
print("Use correlation insights to adjust pricing/discounting.")