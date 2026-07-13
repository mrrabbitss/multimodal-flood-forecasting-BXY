import torch

from src.batch4_models import MODEL_TYPES, build_batch4_model, count_parameters


def test_all_batch4_models_return_bounded_multi_horizon_maps() -> None:
    x = torch.rand(2, 4, 6, 16, 16)
    for model_type in MODEL_TYPES:
        model = build_batch4_model(
            model_type,
            input_channels=6,
            hidden_channels=4,
            num_horizons=3,
            output_max=1.2,
        )
        output = model(x)
        assert output.shape == (2, 3, 16, 16)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.2)
        assert count_parameters(model) > 0
