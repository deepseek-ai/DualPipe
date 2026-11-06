import queue
from typing import List, Callable

import torch
from torch.autograd import Variable


class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    if not micro_outputs:
        return ()
    assert all(x is None or isinstance(x, (torch.Tensor, tuple, list)) for x in micro_outputs)
    if isinstance(micro_outputs[0], torch.Tensor) or micro_outputs[0] is None:
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs
