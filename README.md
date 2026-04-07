# 🔧 Predictive Maintenance using LSTM

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using the NASA CMAPSS dataset and applies **maintenance decision classification**.

---

## 📌 Project Overview

* Dataset: NASA CMAPSS (FD001)
* Model: LSTM (Deep Learning)
* Goal:

  * Predict Remaining Useful Life (RUL)
  * Classify engine health into:

    * 🟢 Healthy
    * 🟡 Warning
    * 🔴 Critical

---

## 📁 Project Structure

```
predictive-maintenance-project/
│
├── src/
│   ├── data_loader.py        # Load dataset
│   ├── preprocessing.py      # RUL + scaling
│   ├── sequence.py           # Sequence creation (LSTM input)
│   ├── model.py              # Model + training
│
├── predictive_maintenance_dataset/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│
├── docs/
│   └── Predictive_Maintenance_Working_Doc.pdf
│
├── output/plots/
│   └── loss.png              # Training vs Validation graph
│
├── main.py                  # Main execution file
├── .gitignore
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/d4rshnn/predictive-maintenance-lstm.git
cd predictive-maintenance-lstm
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Output You Will See

### ✔ Training

* Loss decreasing over epochs
* Graph will:

  * Open automatically
  * Save at:

    ```
    output/plots/loss.png
    ```

---

### ✔ Final RMSE

Example:

```
FINAL RMSE: ~15
```

---

### ✔ Maintenance Decisions

Example:

```
Engine 1: 🟢 Healthy (RUL=118.51)
Engine 18: 🟡 Warning (RUL=30.11)
Engine 20: 🔴 Critical (RUL=14.75)
```

---

## 🧠 Key Concepts Used

* LSTM (Long Short-Term Memory)
* Time-series sequence modeling
* Piecewise RUL (clipped at 125)
* MinMax Scaling
* EarlyStopping (prevents overfitting)

---

## 🚀 Features

* Engine-wise sequence creation (no data leakage)
* Real-world maintenance classification
* Training vs validation visualization
* Clean modular code (industry style)

---

## ⚠️ Important Notes

* Do NOT upload `venv/` (already ignored)
* Dataset paths must remain unchanged
* Requires Python 3.10+

---

## 👨‍💻 Author

Darshan Mahale

---

## 📌 Future Improvements

* Add Streamlit dashboard
* Deploy model
* Hyperparameter tuning
* Add more CMAPSS datasets (FD002–FD004)

---
