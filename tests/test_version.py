from ledger.version import APP_VERSION, DATA_MODEL_VERSION


def test_data_model_version_is_two() -> None:
    assert DATA_MODEL_VERSION == 2


def test_app_version_is_a_string() -> None:
    assert isinstance(APP_VERSION, str)
    assert APP_VERSION == "2.0.0"
