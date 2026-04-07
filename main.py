from src.data_loader import load_data
from src.preprocessing import calculate_rul, preprocess_data
from src.sequence import create_sequences
from src.model import build_model, train_model

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

print("STARTED 🚀")

# -------------------------------
# LOAD TRAIN DATA
# -------------------------------
train_path = "predictive_maintenance_dataset/train_FD001.txt"
df = load_data(train_path)

print("Dataset Loaded ✅")

# -------------------------------
# PREPROCESS TRAIN
# -------------------------------
df = calculate_rul(df)

X_scaled, y, scaler = preprocess_data(df)

# 🔥 CREATE CLEAN DATAFRAME (NO BUG)
df_processed = pd.DataFrame(X_scaled)
df_processed['RUL'] = y.values
df_processed['id'] = df['id'].values

print("Preprocessing Done ✅")

# -------------------------------
# CREATE SEQUENCES
# -------------------------------
time_steps = 30
X_seq, y_seq = create_sequences(df_processed, time_steps)

print("Sequences Created ✅")

# -------------------------------
# BUILD + TRAIN MODEL
# -------------------------------
model = build_model(time_steps, X_seq.shape[2])

print("Model Built ✅")

train_model(model, X_seq, y_seq)

print("Training Complete ✅")

# -------------------------------
# TEST DATA
# -------------------------------
test_path = "predictive_maintenance_dataset/test_FD001.txt"
rul_path = "predictive_maintenance_dataset/RUL_FD001.txt"

test_df = load_data(test_path)

# APPLY SAME SCALER
X_test = test_df.drop(columns=[
    'id', 'cycle',
    'op3', 's1', 's5', 's10', 's16', 's18', 's19'
])
X_test_scaled = scaler.transform(X_test)

test_scaled_df = pd.DataFrame(X_test_scaled)
test_scaled_df['id'] = test_df['id'].values

# CREATE LAST SEQUENCE PER ENGINE
X_test_seq = []

for engine_id in test_scaled_df['id'].unique():
    engine_data = test_scaled_df[test_scaled_df['id'] == engine_id]

    if len(engine_data) >= time_steps:
        seq = engine_data.drop(columns=['id']).iloc[-time_steps:].values
        X_test_seq.append(seq)

X_test_seq = np.array(X_test_seq)

# TRUE RUL
y_true = pd.read_csv(rul_path, header=None)[0].values

# PREDICT
y_pred = model.predict(X_test_seq)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\nFINAL RMSE:", rmse)

# -------------------------------
# MAINTENANCE DECISIONS
# -------------------------------
print("\nMaintenance Decisions:")

for i, val in enumerate(y_pred):

    rul = val[0]

    if rul < 25:
        status = "🔴 Critical"
    elif rul < 50:
        status = "🟡 Warning"
    else:
        status = "🟢 Healthy"

    print(f"Engine {i+1}: {status} (RUL={rul:.2f})")