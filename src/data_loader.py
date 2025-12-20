import pandas as pd
import numpy as np
from visualization import visualize_data


ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def load_data(path):
    df = pd.read_csv(path)

    visualize_data(df=df, save_dir="figures/plots", save=True)
    
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    return X, y


if __name__ == "__main__":
    load_data("data/diabetes.csv")