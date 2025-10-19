import pandas as pd
import pytest

def test_data_columns():
    data_path = '../data/data.csv'
    df = pd.read_csv(data_path)

    expected_columns = [
        "sepal_length", 
        "sepal_width", 
        "petal_length", 
        "petal_width",
        "species" 
    ]
    assert all([col in df.columns for col in expected_columns])
    