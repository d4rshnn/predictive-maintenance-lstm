# Predictive Maintenance using LSTM

This project predicts the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS FD001 dataset. It is a small practical ML pipeline: load the engine run-to-failure data, create fixed-length time-series windows, train an LSTM model, evaluate RMSE on the test engines, and print simple maintenance status labels.

The goal is not deployment or a full monitoring system. It is mainly a clean academic/project implementation of RUL prediction with an LSTM.

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- .gitignore
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- sequence.py
|   `-- model.py
|-- predictive_maintenance_dataset/
|   |-- train_FD001.txt
|   |-- test_FD001.txt
|   |-- RUL_FD001.txt
|   |-- readme.txt
|   `-- Damage Propagation Modeling.pdf
|-- docs/
|   `-- research-paper.pdf
`-- output/
    `-- plots/
        `-- loss.png
```

`venv/`, `__pycache__/`, and temporary render files are local/generated files and are not part of the project structure above.

## Setup

Run the project from the repository root because the dataset and output paths are relative.

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn tensorflow
```

On Linux/macOS, activate the environment with:

```bash
source venv/bin/activate
```

There is no `requirements.txt` in the current repo, so dependencies are installed directly.

## How to Run

```bash
python main.py
```

The script trains the model, predicts RUL for the FD001 test engines, prints the final RMSE, and shows basic maintenance decisions.

## Dataset

The project uses the FD001 subset from NASA CMAPSS, already placed in `predictive_maintenance_dataset/`.

FD001 has:

- 100 training engine trajectories
- 100 test engine trajectories
- one operating condition
- one fault mode, listed in the dataset notes as HPC degradation

The main files used by the code are:

- `train_FD001.txt` for training
- `test_FD001.txt` for test engine sequences
- `RUL_FD001.txt` for the true test RUL values

The loader reads the space-separated files, removes empty columns, and assigns column names:

```text
id, cycle, op1, op2, op3, s1 ... s21
```

The dataset folder also includes the original `Damage Propagation Modeling.pdf` reference paper.

## RUL Prediction Approach

For the training data, RUL is calculated engine-wise:

```text
RUL = max cycle for that engine - current cycle
```

The RUL value is clipped at 125 cycles. This is a common piecewise RUL setup for CMAPSS-style experiments, where early-life cycles are treated as having the same capped health target instead of very large RUL values.

For test data, the script takes the last 30 cycles from each engine and predicts one RUL value per engine. These predictions are compared with `RUL_FD001.txt` using RMSE.

## Preprocessing

Current preprocessing is simple and fixed in code:

- drops `cycle` before scaling
- drops `op3`, `s1`, `s5`, `s10`, `s16`, `s18`, and `s19`
- separates features from `RUL` and `id`
- fits `MinMaxScaler` on training features
- uses the same scaler for test features

The code does not save the scaler separately. It is used only during the current run.

## LSTM Workflow

The sequence length is set to 30 cycles in `main.py`.

`src/sequence.py` creates rolling sequences per engine so windows do not mix data from different engines. Each training sample has:

```text
30 time steps x selected sensor/operation features
```

The model in `src/model.py` is:

- LSTM with 50 units, returning sequences
- Dropout 0.2
- LSTM with 25 units
- Dropout 0.2
- Dense output layer with ReLU activation

Training uses:

- Adam optimizer
- MSE loss
- batch size 32
- up to 50 epochs
- validation split 0.2
- early stopping on validation loss with patience 5

No trained model file is saved by the current implementation.

## Outputs

During a run, the terminal prints progress messages, the final RMSE, and maintenance decisions for each test engine.

The maintenance status is based on simple thresholds:

```text
RUL < 25   -> Critical
RUL < 50   -> Warning
otherwise  -> Healthy
```

This is only a rule-based status label on top of the regression output, not a separate classification model.

The training loss graph is saved automatically at:

```text
output/plots/loss.png
```

The graph compares training loss and validation loss across epochs. The file is already present in the repo and gets overwritten when the script runs. Depending on the environment, Matplotlib may also open the plot window because the script calls `plt.show()`.

## Research Paper

A detailed project research paper/write-up is included here:

```text
docs/research-paper.pdf
```

It documents the project work alongside the code. The dataset's original reference paper is kept separately inside `predictive_maintenance_dataset/`.

## Dependencies

Main libraries used:

- NumPy
- pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib

The code is written as plain Python scripts, with no web app, API, notebook workflow, or deployment setup in the current version.

## Future Improvements

Some realistic next steps:

- add `requirements.txt` with tested package versions
- save the trained model and scaler for reuse
- move paths and sequence length into a small config
- add a baseline model for comparison
- tune LSTM parameters more systematically
- add support for FD002, FD003, and FD004 if those datasets are added
