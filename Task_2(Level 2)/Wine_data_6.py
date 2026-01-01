# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# LOAD DATASET
df = pd.read_csv("Wine_data(6).csv")
print("-----FIRST ROWS -----")
print(df.head())
print("\nDATASET SHAPE:", df.shape)
print("\nMISSING VALUES:\n", df.isnull().sum())

# VISUALIZATION
plt.figure(figsize=(8,5))
sns.countplot(x=df["quality"])
plt.title("WINE QUALITY COUNT")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("CORRELATION HEATMAP")
plt.show()

sns.scatterplot(x="density", y="alcohol", hue="quality", data=df)
plt.title("DENSITY VS ALCOHOL BY QUALITY")
plt.show()

# FEATURE ENGINEERING
df = df.drop("Id", axis=1)
df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]

df.to_csv("Cleaned_Wine_data(6).csv", index=False)
print("\nCleaned dataset saved as: cleaned_Wine_data.csv")

# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
sgd = SGDClassifier()
sgd.fit(X_train_scaled, y_train)
sgd_pred = sgd.predict(X_test_scaled)

svc = SVC()
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print("\n--- SUPPORT VECTOR CLASSIFIER ---")
print("Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, svc_pred),
            annot=True, fmt='d', cmap='Greens')
plt.title("CONFUSION MATRIX - SVC")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# MODEL EVALUATION
print("\n--------------------------")
print("     MODEL PERFORMANCE      ")
print("----------------------------")

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\nSGD Accuracy:", accuracy_score(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred))

print("\nSVC Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

# FEATURE IMPORTANCE 
plt.figure(figsize=(10,5))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh")
plt.title("FEATURE IMPORTANCE (RANDOM FOREST)")
plt.show()

# MODEL COMPARISON
models = ['Random Forest', 'SGD', 'SVC']
accuracies = [
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, sgd_pred),
    accuracy_score(y_test, svc_pred)]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies)
plt.title("MODEL ACCURACY COMPARISON")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
print("\n       FINAL ACCURACY COMPARISON:       ")
for model, acc in zip(models, accuracies):
    print(f"{model}: {acc:.2f}")