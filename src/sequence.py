import numpy as np
import pandas as pd

def create_sequences(df, time_steps=30):
    X_seq = []
    y_seq = []

    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]

        X = engine_data.drop(columns=['RUL', 'id'])
        y = engine_data['RUL']

        for i in range(len(X) - time_steps):
            X_seq.append(X.iloc[i:i+time_steps].values)
            y_seq.append(y.iloc[i+time_steps])

    return np.array(X_seq), np.array(y_seq)