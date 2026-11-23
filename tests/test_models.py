import pytest
import torch

from losses import CrossEntropyDiceLoss
from network import ARCHITECTURES, ModelConfig, build_model


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("shape", [(1, 3, 17, 29), (2, 3, 16, 24), (1, 3, 4, 4)])
def test_full_resolution_logits_and_gradients(architecture, shape):
    model = build_model(ModelConfig(architecture=architecture, base_channels=4, depth=2))
    logits = model(torch.randn(shape))
    assert logits.shape == (shape[0], 8, shape[2], shape[3])
    loss = CrossEntropyDiceLoss()(logits, torch.randint(1, 8, (shape[0], shape[2], shape[3])))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_invalid_input_has_actionable_error():
    model = build_model(ModelConfig(base_channels=4, depth=2))
    with pytest.raises(ValueError, match="at least"):
        model(torch.zeros(1, 3, 3, 10))
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(1, 1, 16, 16))


def test_tiny_batch_can_be_overfit():
    torch.manual_seed(7)
    model = build_model(ModelConfig(base_channels=4, depth=2, dropout=0))
    x = torch.randn(2, 3, 16, 16)
    y = torch.ones(2, 16, 16, dtype=torch.long)
    y[:, :, 8:] = 2
    criterion = CrossEntropyDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = criterion(model(x), y).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    assert criterion(model(x), y).item() < initial * 0.25
