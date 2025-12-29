# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# LOAD DATASET
df = pd.read_csv("Housing_data(5).csv")
X = df[['area']]      
y = df['price']       

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# MODEL EVALUATION
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

df.to_csv("Cleaned_Housing_data(4).csv", index=False)
print("\nCleaned dataset saved as: cleaned_Housing_data.csv")

# VISUALIZATION (LINEAR REGRESSION LINE)
plt.figure()
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, color='red', label="Regression Line")
plt.xlabel("Area (sqft)")
plt.ylabel("House Price")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.show()

# BAR CHART VISUALIZATION
comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

comparison.plot(kind='bar')
plt.title("Actual vs Predicted House Prices (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.show()
