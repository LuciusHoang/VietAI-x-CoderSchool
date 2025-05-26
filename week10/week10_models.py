import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

# ---------- Load label names from batches.meta ----------
with open('cifar-10-batches-py/batches.meta', 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
label_names = [label.decode('utf-8') for label in meta[b'label_names']]
print("Label names:", label_names)

# ---------- Helper function to load a batch ----------
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = batch[b'labels']
        return X, y

# ---------- Load all training data ----------
X_train = []
y_train = []

for i in range(1, 6):
    X, y = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
    X_train.append(X)
    y_train.extend(y)

X_train = np.vstack(X_train)
y_train = np.array(y_train)

# ---------- Load test data ----------
X_test, y_test = load_cifar10_batch('cifar-10-batches-py/test_batch')
X_test = np.array(X_test)
y_test = np.array(y_test)

# ---------- Normalize pixel values ----------
X_train = X_train / 255.0
X_test = X_test / 255.0

# ---------- Optional: Reduce dataset size for quicker testing ----------
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

# ---------- Train Random Forest ----------
print("\nTraining Random Forest...")
start = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f} (Time: {time.time() - start:.2f}s)")

# ---------- Train XGBoost ----------
print("\nTraining XGBoost...")
start = time.time()
xgb = XGBClassifier(n_estimators=100, tree_method='hist', max_depth=6, verbosity=0, use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.4f} (Time: {time.time() - start:.2f}s)")
