import pandas as pd
import dvc.api
import pytest

def test_data_columns():
    data_path = '../data/data.csv'
    try:
        data_content = dvc.api.read(data_path, mode='r')
        from io import StringIO
        df = pd.read_csv(StringIO(data_content))

        expected_columns = [
            "sepal_length", 
            "sepal_width", 
            "petal_length", 
            "petal_width",
            "species" 
        ]

        assert all([col in df.columns for col in expected_columns])

    except Exception as e:
        pytest.fail(f"Failed to read or validate data from DVC: {e}")