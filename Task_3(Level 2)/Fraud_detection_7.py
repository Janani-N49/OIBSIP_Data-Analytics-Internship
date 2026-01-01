# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest

# LOAD DATASET
df = pd.read_csv("fraud_data(7).csv")
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nFraud Distribution:")
print(df['Class'].value_counts())

df.to_csv("Cleaned_fraud_data(7).csv", index=False)
print("\nCleaned dataset saved as: cleaned_Wine_data(7).csv")

# DATA VISUALIZATION
plt.figure(figsize=(5,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# FEATURE ENGINEERING
df['Amount_log'] = np.log1p(df['Amount'])
df.drop(['Amount', 'Time'], axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANOMALY DETECTION – ISOLATION FOREST
iso = IsolationForest(contamination=0.0017, random_state=42)
iso.fit(X_train)
iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

print("\n--- Anomaly Detection (Isolation Forest) ---")
print(classification_report(y_test, iso_pred))

# LOGISTIC REGRESSION
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

sns.heatmap(confusion_matrix(y_test, lr_pred),
            annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# DECISION TREE
dt = DecisionTreeClassifier(
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

#  NEURAL NETWORK (MLP)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=42
)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

print("\n--- Neural Network (MLP) ---")
print("Accuracy:", accuracy_score(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))

# MODEL COMPARISON
models = ['Logistic Regression', 'Decision Tree', 'Neural Network']
accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, mlp_pred)
]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies)
plt.title("Fraud Detection Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# REAL-TIME FRAUD DETECTION FUNCTION
rt_model = LogisticRegression(n_jobs=-1)
rt_model.fit(X_train, y_train)
print("\n       REAL-TIME FRAUD DETECTION     ")
stream_data = X_test[:100]   
preds = rt_model.predict(stream_data)

for i, pred in enumerate(preds):
    status = "🔴 FRAUD" if pred == 1 else "🟢 Normal"
    print(f"Transactions {i+1}: {status}")

# SCALABILITY NOTES
print("\nSystem is scalable using:")
print("- Batch processing")
print("- Real-time Kafka / Spark streaming")
print("- Cloud deployment (AWS / GCP / Azure)")