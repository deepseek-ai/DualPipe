import torch

from dualpipe.utils import (
    WeightGradStore,
    cat_tensor,
    chunk_tensor,
    gather,
    scatter,
)


def test_chunk_tensor():
    assert chunk_tensor(None, 4, dim=0) == [None, None, None, None]

    x = torch.arange(8)
    chunks = chunk_tensor(x, 4, dim=0)
    assert len(chunks) == 4
    for i, c in enumerate(chunks):
        assert torch.equal(c, torch.tensor([i * 2, i * 2 + 1]))


def test_cat_tensor():
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([3, 4])

    assert torch.equal(cat_tensor([t1], dim=0), t1)
    assert torch.equal(cat_tensor([t1, t2], dim=0), torch.tensor([1, 2, 3, 4]))
    assert cat_tensor([None, None], dim=0) is None
    assert cat_tensor((None,), dim=0) is None


def test_scatter():
    t1 = torch.arange(8)
    t2 = torch.arange(8, 16)

    # Single tensor
    batches = scatter(t1, 4, dim=0)
    assert len(batches) == 4
    assert all(len(b) == 1 for b in batches)

    # Tuple of tensors
    batches = scatter((t1, t2), 4, dim=0)
    assert len(batches) == 4
    assert all(len(b) == 2 for b in batches)

    # Mixed with None
    batches = scatter((t1, None), 4, dim=0)
    assert len(batches) == 4
    assert all(len(b) == 2 and b[1] is None for b in batches)


def test_gather_empty():
    assert gather([], dim=0) == ()


def test_gather_single_tensor():
    chunks = [torch.tensor([i, i + 1]) for i in range(0, 8, 2)]
    res = gather(chunks, dim=0)
    assert len(res) == 1
    assert torch.equal(res[0], torch.arange(8))


def test_gather_tuple_tensors():
    chunks = [(torch.tensor([i]), torch.tensor([i * 10])) for i in range(4)]
    res = gather(chunks, dim=0)
    assert len(res) == 2
    assert torch.equal(res[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(res[1], torch.tensor([0, 10, 20, 30]))


def test_gather_none():
    res = gather([None, None, None], dim=0)
    assert res == (None,)

    res2 = gather([(torch.tensor([1]), None), (torch.tensor([2]), None)], dim=0)
    assert len(res2) == 2
    assert torch.equal(res2[0], torch.tensor([1, 2]))
    assert res2[1] is None


def test_scatter_gather_roundtrip():
    x = torch.randn(12, 16)
    chunks = scatter(x, 4, dim=0)
    gathered = gather(chunks, dim=0)
    assert len(gathered) == 1
    assert torch.equal(gathered[0], x)


def test_weight_grad_store():
    WeightGradStore.clear()
    assert WeightGradStore.cache == []
    assert WeightGradStore.funcs_queue.empty()

    executed = []
    WeightGradStore.put(lambda: executed.append(1))
    WeightGradStore.put(lambda: executed.append(2))
    assert len(WeightGradStore.cache) == 2

    WeightGradStore.flush()
    assert WeightGradStore.cache == []
    assert not WeightGradStore.funcs_queue.empty()

    WeightGradStore.pop()
    assert executed == [1, 2]
    assert WeightGradStore.funcs_queue.empty()

    WeightGradStore.clear()
