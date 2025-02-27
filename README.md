# DualPipe

DualPipe is an innovative bidirectional pipeline parallism algorithm introduced in the [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437). It achieves full overlap of forward and backward computation-communication phases, also reducing pipeline bubbles. For detailed information on computation-communication overlap, please refer to the [profile data](https://github.com/deepseek-ai/profile-data).

### Schedules

![schedules](images/schedules.png)

Example DualPipe scheduling for 8 PP ranks and 20 micro-batches in two directions.
The micro-batches in the reverse direction are symmetric to those in the forward direction, so
we omit their batch ID for illustration simplicity. Two cells enclosed by a shared black border
have mutually overlapped computation and communication

### Pipeline Bubbles and Memory Usage Comparison

| Method      | Bubble                          | Parameter | Activation |
|-------------|---------------------------------|-----------|------------|
| 1F1B        | (PP-1)(𝐹+B)                     | 1×        | PP         |
| ZB1P        | (PP-1)(𝐹+𝐵-2𝑊)                  | 1×        | PP         |
| DualPipe    | (PP/2-1)(𝐹&𝐵+𝐵-3𝑊)             | 2×        | PP+1       |

𝐹 denotes the execution time of a forward chunk, 𝐵 denotes the execution time of a
full backward chunk, 𝑊 denotes the execution time of a "backward for weights" chunk, and 𝐹&𝐵
denotes the execution time of two mutually overlapped forward and backward chunks.

## Quick Start

The usage is shown in the following example:

```bash
python example.py
```

Note: For real-world applications, you will need to implement a custom `overlapped_forward_backward` method tailored to your specific module.

## Requirements

- PyTorch 2.0 and above

## Developers

DualPipe was created and developed by Jiashi Li and Chengqi Deng and Wenfeng Liang.

## Citation

```bibtex
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437}, 
}
```
