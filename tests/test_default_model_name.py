from ega.verifiers.nli_cross_encoder import DEFAULT_MODEL_NAME


def test_default_model_name() -> None:
    assert DEFAULT_MODEL_NAME == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
