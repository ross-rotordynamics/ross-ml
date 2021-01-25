"""pipeline.py test file"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential

from rossml.pipeline import Model, Pipeline

file = Path(__file__).parent / r"data\seal_data.csv"
df = pd.read_csv(file)


@pytest.fixture
def model():
    return Pipeline(df)


def test_pipeline(model):
    # setting features and labels
    x = model.set_features(0, 20)
    y = model.set_labels(20, 28)

    assert all(col1 == col2 for col1, col2 in zip(x.columns, df.columns[0:20]))
    assert all(col1 == col2 for col1, col2 in zip(y.columns, df.columns[20:28]))

    # without feature reduction and not scaling
    z = model.feature_reduction(20)
    x_train, x_test, y_train, y_test = model.data_scaling(0.1, scaling=False)
    assert len(list(z.columns)) == 20
    assert x_train.shape[1] == 20
    assert x_test.shape[1] == 20
    assert y_train.shape[1] == 8
    assert y_test.shape[1] == 8

    # with feature reduction and scaling
    z = model.feature_reduction(10)
    x_train, x_test, y_train, y_test = model.data_scaling(
        0.1, scalers=[RobustScaler(), RobustScaler()], scaling=True
    )
    assert len(list(z.columns)) == 10
    assert x_train.shape[1] == 10
    assert x_test.shape[1] == 10
    assert y_train.shape[1] == 8
    assert y_test.shape[1] == 8

    # building the model
    model.build_Sequential_ANN(4, [50, 50, 50, 50])
    ann_model, predictions = model.model_run(batch_size=300, epochs=500)

    assert model.history.params["epochs"] == 500
    assert model.predictions.shape[1] == 8
    assert isinstance(model.model, Sequential)


def test_model():
    model = Model("test_model")
    assert model.name == "test_model"
