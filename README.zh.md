# DualPipe

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

DualPipe 是在 [DeepSeek-V3 技术报告](https://arxiv.org/pdf/2412.19437) 中提出的一种创新的双向流水线并行（Bidirectional Pipeline Parallelism）算法。它实现了前向与反向计算-通信阶段的完全重叠（Full Overlap），并显著降低了流水线气泡（Pipeline Bubbles）。有关计算-通信重叠的详细性能分析信息，请参阅[性能分析数据 (profile data)](https://github.com/deepseek-ai/profile-data)。

### 调度方案

![dualpipe](images/dualpipe.png)

8 个流水线并行（PP）rank 在两个方向上处理 20 个微批次（micro-batch）的 DualPipe 调度示例。反向传播中的微批次与前向传播对称，为简化示意省略了其批次 ID。由同一黑色粗边框包围的两个单元具有相互重叠的计算与通信过程。

## DualPipeV

DualPipeV 是一种利用“对半切分 (cut-in-half)”过程从 DualPipe 衍生而来的精简 V 形调度方案，由 Sea AI Lab 在其[技术博客](https://hackmd.io/@ufotalent/r1lVXsa9Jg)中以 "Cut-in-half" 之名提出。感谢他们提供这一高效的调度算法！

### 调度方案

![dualpipev](images/dualpipev.png)

4 个 PP rank（8 个 PP stage）处理 10 个微批次的 DualPipeV 调度示例。

## 流水线气泡与显存占用对比（基于相同 PP stage 数量）

| 方法 | 气泡时间 | 单卡参数量 | 单卡激活值 | 设备数量 |
|---|---|---|---|---|
| 1F1B | (*PP*-1)(𝐹+𝐵) | 1× | *PP* | *PP* |
| ZB1P | (*PP*-1)(𝐹+𝐵-2𝑊) | 1× | *PP* | *PP* |
| DualPipe | (*PP*/2-1)(𝐹&𝐵+𝐵-3𝑊) | 2× | *PP*+1 | *PP* |
| DualPipeV | (*PP*/2-1)(𝐹&𝐵+𝐵-3𝑊) | 2× | *PP*+1 | *PP*/2 |

*PP* 表示流水线并行阶段数（偶数）。
𝐹 表示前向分块（forward chunk）的执行时间，𝐵 表示完整反向分块（full backward chunk）的执行时间，𝑊 表示“针对权重的反向传播（backward for weights）”分块的执行时间，𝐹&𝐵 表示两个相互重叠的前向和反向分块的执行时间。

## 快速上手

使用示例如下：

```bash
python examples/example_dualpipe.py
python examples/example_dualpipev.py
```

说明：在实际业务场景中，您需要针对具体的模型模块实现自定义的 `overlapped_forward_backward` 方法。

## 环境依赖

- PyTorch 2.0 及更高版本

## 核心开发者

DualPipe 由李家仕（Jiashi Li）、邓成其（Chengqi Deng）和梁文锋（Wenfeng Liang）构思并开发。

## 论文引用

```bibtex
@misc{deepseekai2025deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2025},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月14日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
