import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.dropna(axis=1)

    columns = ["id", "cycle"]
    columns += [f"op{i}" for i in range(1, 4)]
    columns += [f"s{i}" for i in range(1, 22)]

    df.columns = columns

    return df