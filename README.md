# PagedInfer: Continuous Batching & Paged KV-Cache LLM Serving Engine

PagedInfer is a high-throughput, low-latency LLM serving engine built with PyTorch and FastAPI. It implements dynamic physical memory page allocation (inspired by vLLM) and iteration-level continuous scheduling to eliminate GPU memory fragmentation and maximize inference throughput.

## Core Architectural Features

* **Paged KV-Cache Allocator**: Partitions Key-Value cache tensors into fixed-size physical memory pages (16 tokens/block), eliminating internal and external GPU memory fragmentation.
* **Continuous Iteration Scheduler**: Dynamically injects arriving prompt sequences and evicts completed decode steps on every forward pass without stalling active streams.
* **Zero-Waste Memory Reclamation**: Reclaims physical memory blocks immediately upon generation completion.
* **Server-Sent Events (SSE) Streaming Gateway**: Exposes real-time token streaming endpoints with live KV-cache block allocation telemetry.

## System Architecture

```text
Incoming Requests ---> [ Async Ingestion Queue ]
                               |
                               v
                     [ Iteration Scheduler ] <---> [ Paged Block Allocator ]
                               |                          (Physical Page Table)
                               v
                    [ PyTorch Model Runner ]
                               |
                               v
                 [ Real-time SSE Token Stream ]