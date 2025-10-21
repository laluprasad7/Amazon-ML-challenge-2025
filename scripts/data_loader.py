import pandas as pd
from config import project_root, train_csv, test_csv


def load_data():
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        print("Train data loaded:", train_df.shape)
        print("Test data loaded:", test_df.shape)
        return train_df, test_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {e}. Ensure {train_csv} and {test_csv} exist in {project_root}")