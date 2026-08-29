# Pipeline Parallelism Comparison Guide

This document provides a comprehensive comparison of pipeline parallelism methods, addressing questions from issues [#15](https://github.com/deepseek-ai/DualPipe/issues/15) (VPP vs DualPipe) and [#20](https://github.com/deepseek-ai/DualPipe/issues/20) (DualPipeV benefits).

## Overview

| Method | Core Idea | Bubble Ratio | Param Memory | Activation Memory | Devices |
|--------|-----------|--------------|--------------|-------------------|---------|
| **1F1B** | Alternating forward/backward | (PP-1)/acc | 1x | PP | PP |
| **VPP** | Virtual stages interleaving | (1/vpp)×(PP-1)/acc | 1x | Higher | PP |
| **ZB1P** | Weight-gradient scheduling | Near-zero (large acc) | 1x | ~1F1B | PP |
| **DualPipe** | Bidirectional full overlap | ~(1/3)×(PP-2)/acc | 2x | PP+1 | PP |
| **DualPipeV** | V-shape half-device variant | ~DualPipe trend | 2x | PP+1 | PP/2 |

*PP = pipeline stages, acc = gradient accumulation microbatches per optimizer step, vpp = virtual pipeline stages*

**Assumptions behind the bubble formulas.** `acc` counts gradient-accumulation microbatches within one optimizer step. Forward and backward chunk costs are treated as equal; in practice backward is roughly 2x forward, which changes the constants but not the relative trends. Communication time is excluded, i.e. assumed fully overlapped with computation; on interconnects where that overlap fails, the observed bubble is larger. The ratios are scheduling-model estimates for comparing methods, not exact throughput predictions.

## Method Descriptions

### 1F1B (One Forward One Backward)

The traditional pipeline parallelism approach where each device alternates between forward and backward passes after a warmup phase.

**Characteristics:**
- Simple and widely supported
- Bubble from pipeline fill/flush phases
- Minimal complexity and memory overhead
- Best baseline for comparison

### VPP (Virtual Pipeline Parallelism)

Introduced by Megatron-LM, VPP splits each physical pipeline stage into multiple virtual sub-stages and interleaves their execution.

**Characteristics:**
- Reduces bubble by factor of `1/vpp`
- More frequent, smaller communication messages
- Slightly higher activation memory (more in-flight microbatches)
- Good when network latency is low

### ZB1P (Zero Bubble 1P)

From the "Zero Bubble" paper, ZB1P reorders weight gradient computation and communication to hide pipeline bubbles.

**Characteristics:**
- Targets near-zero bubble with large accumulation steps
- Requires strict dependency ordering
- No extra parameter memory
- Complex implementation

### DualPipe (DeepSeek V3)

DeepSeek's bidirectional pipeline where forward passes flow in one direction while backward passes flow in the opposite direction, achieving full overlap.

**Characteristics:**
- Full forward-backward computation-communication overlap
- Requires 2x parameter memory (mirrored weights)
- Concurrent bidirectional traffic
- Best throughput when bandwidth available

### DualPipeV (V-Shape Variant)

A "cut-in-half" variant of DualPipe introduced by Sea AI Lab, using half the devices while retaining bidirectional overlap benefits.

**Characteristics:**
- Half the devices of DualPipe
- Retains overlap behavior
- Lower hardware cost
- Slightly reduced performance vs full DualPipe

## Bubble Ratio Analysis

The bubble ratio determines idle time during training. Lower is better. The formulas below inherit the assumptions stated under the Overview table: equal forward and backward chunk cost, and communication fully hidden by computation.

| Method | Formula | At PP=8, acc=16 |
|--------|---------|-----------------|
| 1F1B | (PP-1)/acc | 43.75% |
| VPP (vpp=2) | (1/2)×(PP-1)/acc | 21.88% |
| VPP (vpp=4) | (1/4)×(PP-1)/acc | 10.94% |
| DualPipe | ~(1/3)×(PP-2)/acc | ~12.5% |
| DualPipeV | Similar to DualPipe | ~12.5% |

**Key Insight:** When `vpp > 3`, VPP's bubble ratio becomes competitive with DualPipe. However, DualPipe achieves its low bubble through *overlap* rather than *interleaving*, which has different memory and bandwidth implications.

## Memory Trade-offs

### Parameter Memory

| Method | Parameter Memory | Why |
|--------|------------------|-----|
| 1F1B, VPP, ZB1P | 1x (baseline) | Single copy of weights per stage |
| DualPipe | 2x | Mirrored weights for bidirectional flow |
| DualPipeV | 2x | Still requires mirrored weights |

### Activation Memory

| Method | Activation Memory | Why |
|--------|-------------------|-----|
| 1F1B | PP microbatches | Standard pipeline depth |
| VPP | Higher than 1F1B | More in-flight microbatches |
| ZB1P | ~1F1B | Similar to baseline |
| DualPipe/V | PP+1 microbatches | Overlap reduces peak, slight overhead |

## Communication Patterns

| Method | Pattern | Bandwidth Requirement |
|--------|---------|----------------------|
| **1F1B** | Sequential forward/backward | Moderate |
| **VPP** | Frequent small messages | Low latency critical |
| **ZB1P** | Ordered gradient comm | Moderate, ordering-sensitive |
| **DualPipe** | Concurrent bidirectional | High bandwidth, both directions |
| **DualPipeV** | V-pattern bidirectional | High bandwidth, half scale |

## When to Use Each Method

### Choose 1F1B when:
- Simplicity and stability are priorities
- Memory budget is tight
- Moderate pipeline depth and accumulation
- First implementation of pipeline parallelism

### Choose VPP when:
- Pipeline bubble is the throughput bottleneck
- Network has low latency for small messages
- Activation memory budget allows more in-flight batches
- Cannot afford 2x parameter memory

### Choose ZB1P when:
- Large accumulation steps are used
- Near-zero bubble is critical
- Can implement strict weight-gradient ordering
- Parameter memory must stay at 1x

### Choose DualPipe when:
- Maximum throughput is the goal
- 2x parameter memory is acceptable
- Network supports concurrent bidirectional traffic
- Deep pipeline with smaller accumulation (bubble-dominated)

### Choose DualPipeV when:
- Want DualPipe-like overlap benefits
- Device budget is limited (half the GPUs)
- Willing to accept some performance trade-off
- Debugging with minimal resources

## MoE with Expert Parallelism Considerations

When using Mixture of Experts (MoE) with Expert Parallelism (EP):

1. **Compare same PP stage counts** - As noted in issue #15, compare PP3VPP4, DualPipe PP12, and DualPipeV PP6 when using 24 devices.

2. **Network pressure** - EP increases per-microbatch activation traffic. Methods with more in-flight microbatches (VPP) or bidirectional comm (DualPipe) need stronger networks.

3. **Parameter memory** - DualPipe's 2x params is significant with large MoE expert weights. DualPipeV may be preferable.

4. **Collective overlap** - Ensure EP's all-to-all dispatch doesn't conflict with pipeline sends. Use separate streams/channels.

5. **Router ordering** - For ZB1P, ensure token routing happens before gradient steps per the schedule.

## Decision Flowchart

```
Start
  │
  ├─► Can you afford 2x parameter memory?
  │     │
  │     ├─► YES: Is bidirectional bandwidth available?
  │     │         │
  │     │         ├─► YES: Use DualPipe (max throughput)
  │     │         │
  │     │         └─► NO: Use DualPipeV or VPP
  │     │
  │     └─► NO: Continue below
  │
  ├─► Is accumulation steps large (>32)?
  │     │
  │     ├─► YES: Can you implement strict ordering?
  │     │         │
  │     │         ├─► YES: Use ZB1P (near-zero bubble)
  │     │         │
  │     │         └─► NO: Use VPP
  │     │
  │     └─► NO: Continue below
  │
  ├─► Is low-latency network available?
  │     │
  │     ├─► YES: Use VPP (vpp=2-4)
  │     │
  │     └─► NO: Use 1F1B (simplest)
  │
  └─► Default: Start with 1F1B, profile, then optimize
```

## Summary

| If you need... | Use... |
|----------------|--------|
| Simplicity | 1F1B |
| Lower bubble, 1x params | VPP or ZB1P |
| Maximum throughput | DualPipe |
| DualPipe on fewer GPUs | DualPipeV |
| MoE + memory constraints | VPP or DualPipeV |

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - DualPipe
- [Sea AI Lab: Cut-in-half](https://hackmd.io/@ufotalent/r1lVXsa9Jg) - DualPipeV
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - VPP
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) - ZB1P
