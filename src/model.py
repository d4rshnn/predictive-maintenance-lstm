from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------
# Build Model (Improved)
# -------------------------------
def build_model(time_steps, num_features):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, num_features)))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(25))
    model.add(Dropout(0.2))

    # Output layer (important: relu)
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mse')

    return model


# -------------------------------
# Train Model
# -------------------------------
def train_model(model, X, y):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    # -------------------------------
    # Graph (Loss vs Epoch)
    # -------------------------------
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("output/plots/loss.png")
    plt.show()

    return model