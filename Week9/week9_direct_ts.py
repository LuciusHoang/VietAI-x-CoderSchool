import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# === 1. Load and preprocess data ===
data = pd.read_csv("electricity.csv")
# data = pd.read_csv("Week9/electricity.csv")
data["Time"] = pd.to_datetime(data["Time"])
data["Date"] = pd.to_datetime(data["Date"])
data["Demand"] = data["Demand"].interpolate()

# Label encode 'Holiday': True → 1, False → 0
data["Holiday"] = data["Holiday"].astype(int)

# === 2. Create time series features ===
def create_ts_data(data, window_size):
    for i in range(1, window_size):
        data[f"Demand_{i}"] = data["Demand"].shift(i)
    data["TargetDemand"] = data["Demand"].shift(-1)  # Predict next time step
    data = data.dropna(axis=0)
    return data

window_size = 10
data = create_ts_data(data, window_size)

# === 3. Split features and target ===
x = data.drop(["Date", "Time", "TargetDemand"], axis=1)
y = data[["TargetDemand"]]

# === 4. Train/test split ===
train_ratio = 0.8
num_samples = len(data)
x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

# === 5. Model training with normalization ===
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)

# === 6. Prediction & Evaluation ===
y_predict = model.predict(x_test)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# === 7. Optional: Plot prediction vs actual ===
plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label="Actual")
plt.plot(y_predict, label="Predicted")
plt.title("Electricity Demand Forecast")
plt.xlabel("Time Step")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()
plt.show()
