import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Week6_7/csgo.csv")
# df = pd.read_csv("csgo.csv")

# profile = ProfileReport(df, title="CSGO Report")
# profile.to_file("CSGO Report.html")

# Drop unnecessary columns
columns_to_drop = ["date", "team_a_rounds", "team_b_rounds"]
df_cleaned = df.drop(columns=columns_to_drop)

# Remove rows where the result is "Tie"
df_classification = df_cleaned[df_cleaned["result"] != "Tie"].copy()

# Encode the 'map' column
map_encoder = LabelEncoder()
df_classification['map_encoded'] = map_encoder.fit_transform(df_classification['map'])

# Encode the 'result' column: Win -> 1, Lost -> 0
result_encoder = LabelEncoder()
df_classification['result_encoded'] = result_encoder.fit_transform(df_classification['result'])

# Define feature columns (excluding 'map', 'result')
feature_columns = df_classification.columns.difference(['result_encoded','map', 'result'])

# Define input features (X) and target labels (y)
X = df_classification[feature_columns]
y = df_classification['result_encoded']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
