# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATASET
df = pd.read_csv("New_york_data.csv")
print("ORIGINAL DATASET PREVIEW:")
print(df.head())
print("\nDataset Shape:", df.shape)

# DATA INTEGRITY CHECK
print("\nDATASET INFORMATION:")
print(df.info())
print("\nSTATISTICAL SUMMARY:")
print(df.describe(include='all'))


# MISSING DATA HANDLING
print("\n      MISSING VALUES BEFORE CLEANING:       ")
print(df.isnull().sum())

df['name'].fillna("Unknown", inplace=True)
df['host_name'].fillna("Unknown", inplace=True)

df['reviews_per_month'].fillna(0, inplace=True)

df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['last_review'].fillna(pd.Timestamp("1970-01-01"), inplace=True)

print("\n       MISSING VALUES AFTER CLEANING:       ")
print(df.isnull().sum())

# DUPLICATE REMOVAL
print("\nDuplicate Records Before Removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicate Records After Removal:", df.duplicated().sum())
print("Dataset Shape After Duplicate Removal:", df.shape)

# 6. STANDARDIZATION
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
text_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in text_columns:
    df[col] = df[col].str.lower().str.strip()

print("\nCOLUMN NAMES AFTER STANDARDIZATION:")
print(df.columns)

# OUTLIER DETECTION & REMOVAL (IQR METHOD)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

numeric_columns = ['price', 'minimum_nights', 'number_of_reviews', 'availability_365']

for col in numeric_columns:
    df = remove_outliers_iqr(df, col)

print("\nDataset Shape After Outlier Removal:", df.shape)

#VISUALIZATION
avg_price = df.groupby('room_type')['price'].mean()
plt.figure()
avg_price.plot(kind='bar')
plt.title("    AVERAGE PRICE BY ROOM TYPE     ")
plt.xlabel("Room Type")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# FINAL CLEANED DATASET
print("\nCLEANED DATASET PREVIEW")
print(df.head())

# SAVE CLEANED DATA
df.to_csv("cleaned_New_york.csv", index=False)
print("\nData Cleaning Completed Successfully!")
print("Cleaned file saved as 'cleaned New_york.csv'")




