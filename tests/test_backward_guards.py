import torch


def test_empty_non_empty_unfiltered_grad_unpack():
    """Verify that when all output gradients are None, zip(*non_empty) does not raise ValueError."""
    outputs = [torch.randn(2, 4), torch.randn(2, 4)]
    output_grads = [None, None]

    non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
    if non_empty:
        filtered_outputs, filtered_grads = list(zip(*non_empty))
    else:
        filtered_outputs, filtered_grads = [], []

    assert filtered_outputs == []
    assert filtered_grads == []


def test_input_grads_with_none_elements():
    """Verify that input_grads extraction handles None entries gracefully without AttributeError."""
    t1 = torch.randn(2, 4, requires_grad=True)
    t1.grad = torch.ones_like(t1)
    inputs = [t1, None]

    input_grads = [t.grad if t is not None else None for t in inputs]
    assert len(input_grads) == 2
    assert torch.equal(input_grads[0], torch.ones_like(t1))
    assert input_grads[1] is None
