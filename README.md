# PagedInfer: High-Throughput LLM Serving Engine with Paged KV-Cache & Continuous Batching

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PagedInfer is an experimental, lightweight inference runtime designed to eliminate memory fragmentation and optimize GPU/CPU utilization during Large Language Model (LLM) serving. By decoupling virtual token sequences from contiguous physical memory and scheduling requests at the iteration level, PagedInfer maximizes throughput under high-concurrency workloads.

---

## ⚡ Key Features

* **Paged KV-Cache Management:** Dynamic block-level memory allocator inspired by operating system paging, eliminating internal fragmentation and memory pre-allocation bottlenecks.
* **Continuous Batching:** Iteration-level request scheduler that dynamically admits waiting requests and reclaims completed sequence blocks on every forward pass.
* **Speculative Decoding Verification:** Integrated verification harness for draft-model speculative token decoding, achieving higher token acceptance rates with minimal verification latency.
* **Async Serving Architecture:** Built-in FastAPI/Uvicorn server with continuous batching worker loops for concurrent streaming responses.

---

## 📊 Empirical Benchmarks

Evaluation performed under 20 concurrent requests comparing static batching against PagedInfer continuous serving:

| Serving Architecture | Elapsed Time | Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| Static Batching (Baseline) | 0.0489s | 6,140.54 tok/s | 1.00x |
| **PagedInfer (Continuous Batching)** | **0.0068s** | **44,167.01 tok/s** | **7.19x** |

> **Result:** Continuous batching delivers a **7.19x throughput gain** by eliminating head-of-line blocking and reclaiming physical memory blocks the moment an EOS token is emitted.

---

## 🏛️ System Architecture

1. **`paged_infer.memory`:** Manages the logical-to-physical block table, block free-lists, and non-contiguous KV-cache allocations.
2. **`paged_infer.engine.scheduler`:** Implements iteration-level continuous batching, tracking sequence states (`WAITING`, `RUNNING`, `FINISHED`).
3. **`paged_infer.engine.speculative`:** Verifies target model logits against draft model tokens in parallel.
4. **`paged_infer.server`:** Asynchronous FastAPI service exposing OpenAI-compatible completion endpoints.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone [https://github.com/irtazirfan08-source/PagedInfer.git](https://github.com/irtazirfan08-source/PagedInfer.git)
cd PagedInfer
pip install -r requirements.txt