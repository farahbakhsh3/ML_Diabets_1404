import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y
