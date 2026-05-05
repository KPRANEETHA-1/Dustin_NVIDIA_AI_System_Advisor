# NVIDIA TensorRT — RAG Knowledge Base

**Source:** [NVIDIA TensorRT Official Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)  
**Category:** Inference Optimization, GPU Acceleration, Model Compilation  
**Tool Identity:** TensorRT is NVIDIA's inference optimization SDK — it takes a trained model and produces a highly optimized, GPU-specific engine for maximum inference speed.

---

## 1. What is TensorRT?

NVIDIA TensorRT is a **high-performance deep learning inference SDK** that optimizes trained models for deployment on NVIDIA GPUs. It is not a training framework — it is purely an *optimization and runtime* layer.

**What TensorRT does:**
1. Takes a trained model (from PyTorch, TensorFlow, ONNX, etc.)
2. Applies graph optimizations, layer fusions, kernel auto-tuning
3. Produces a serialized "engine" (a `.plan` file) optimized for a specific GPU
4. Runs inference with that engine at maximum throughput and minimum latency

**Current version:** TensorRT 10.x (TensorRT 11.0 coming soon with modernized APIs)

---

## 2. When to Use TensorRT (Decision Guide)

Use TensorRT when you need to:

| Use Case | TensorRT Feature |
|---|---|
| Maximize inference speed on NVIDIA GPU | Kernel auto-tuning, layer fusion, graph optimization |
| Reduce model memory footprint | INT8/FP8/FP4 quantization |
| Deploy production LLMs at scale | TensorRT-LLM (TRT-LLM) |
| Run models with mixed precision | FP16/BF16/INT8 mixed precision |
| Support models with variable input sizes | Dynamic Shapes |
| Partition GPU for multiple isolated workloads | MIG (Multi-Instance GPU) support |

**TensorRT vs Triton vs NeMo — Quick Rule:**
- Need to *optimize* a model for GPU inference → **TensorRT**
- Need to *serve* a model with an API → **Triton** (which uses TensorRT as a backend)
- Need to *train or fine-tune* a model → **NeMo**
- The natural stack: NeMo trains → TensorRT optimizes → Triton serves

---

## 3. TensorRT Architecture: Two-Phase Model

TensorRT operates in **two distinct phases**:

### Phase 1: Build Phase (Offline — done once)
```
Input: Trained model (ONNX, PyTorch, TensorFlow)
       ↓
NetworkDefinition (defines the model graph)
       ↓
BuilderConfig (specifies optimization targets: precision, workspace size, profiles)
       ↓
Builder runs: graph optimization, layer fusion, kernel timing
       ↓
Output: Serialized Engine ("plan" file, e.g., model.plan)
```

**What the Builder does:**
- Eliminates dead computations
- Folds constants
- Fuses layers (e.g., Conv + BN + ReLU → single kernel)
- Selects the fastest CUDA kernel for each operation
- Applies precision reduction (FP16, INT8) where safe
- Generates CUDA kernels tuned for the specific GPU

> ⚠️ **Important:** The build phase can take **minutes to hours** for large models. This is expected — it's a one-time cost.

### Phase 2: Runtime Phase (Online — runs inference)
```
Load serialized engine (model.plan)
       ↓
Create ExecutionContext
       ↓
Populate input buffers (CPU or GPU memory)
       ↓
Call enqueueV3() → CUDA stream execution
       ↓
Synchronize stream → Read output buffers
```

**Key insight:** The engine is GPU-specific and TensorRT-version-specific by default. To share engines across machines, use version compatibility settings.

---

## 4. Precision and Quantization

TensorRT's biggest performance lever is **precision control**. Lower precision = faster computation + less memory = higher throughput.

### Supported Data Types

| Type | Bits | Use Case |
|---|---|---|
| FP32 | 32 | Default, maximum accuracy |
| TF32 | 19 | Ampere GPUs, good accuracy/speed balance |
| FP16 | 16 | Great speedup on tensor cores, minimal accuracy loss |
| BF16 | 16 | Better dynamic range than FP16 for LLMs |
| FP8 | 8 | Hopper GPUs, major speedup for LLMs |
| FP4 | 4 | Blackwell GPUs, extreme compression |
| INT8 | 8 | Legacy quantization, requires calibration |
| INT4 | 4 | Weight-only quantization |
| INT32 | 32 | Integer operations |
| BOOL | 1 | Boolean logic |

### Strong Typing vs Weak Typing

| Mode | Behavior | Recommended? |
|---|---|---|
| **Strong Typing** | TensorRT statically infers types; no automatic precision changes | ✅ Yes (as of TRT 10.x) |
| **Weak Typing** | TensorRT may substitute lower precision for speed | ⚠️ Deprecated in 10.12 |

**Best practice:** Use **strongly typed networks** for better precision control, faster build times, and future compatibility.

### Quantization Workflow (INT8)
```
Option A: Post-Training Quantization (PTQ)
  → Provide calibration dataset → TensorRT calculates dynamic ranges

Option B: Quantization-Aware Training (QAT)
  → Train with quantization simulation → Export to ONNX → TensorRT imports scales

Tool: NVIDIA Model Optimizer (replaces deprecated PyTorch/TF Quantization Toolkits)
```

---

## 5. Dynamic Shapes

By default, TensorRT optimizes for fixed input shapes. **Dynamic Shapes** allow the engine to handle variable input sizes at runtime.

### How it works:
Define one or more `OptimizationProfile` with:
- **Minimum shape** — smallest allowed input
- **Optimal shape** — shape TensorRT optimizes for
- **Maximum shape** — largest allowed input

TensorRT generates kernels valid across the entire [min, max] range, fastest at the optimal point.

```python
profile = builder.create_optimization_profile()
profile.set_shape("input_ids", 
    min=(1, 1),       # batch=1, seq_len=1
    opt=(4, 512),     # batch=4, seq_len=512
    max=(8, 2048))    # batch=8, seq_len=2048
config.add_optimization_profile(profile)
```

---

## 6. Model Import Paths

### Option A: ONNX (Recommended for most frameworks)
```
PyTorch model → torch.onnx.export() → model.onnx
TensorFlow model → tf2onnx → model.onnx
                                    ↓
                          TensorRT ONNX Parser
                                    ↓
                          TensorRT Engine
```

**Post-export tip:** Run constant folding with Polygraphy before parsing:
```bash
polygraphy surgeon sanitize model.onnx --fold-constants -o model_folded.onnx
```

### Option B: PyTorch (Torch-TensorRT)
```
PyTorch model → Torch-TensorRT compiler → TensorRT-accelerated module
```
Subgraphs are compiled by TensorRT; unsupported operations fall back to native PyTorch.

### Option C: Python API (Layer-by-layer construction)
Build the network programmatically using TensorRT's API. Use **TriPy** for a Pythonic interface.

---

## 7. TensorRT-LLM (TRT-LLM)

TensorRT-LLM is a **specialized, open-source library built on top of TensorRT** specifically for large language models.

**What TRT-LLM adds over vanilla TensorRT:**
- LLM-specific optimizations: **in-flight batching**, **paged KV cache**, **speculative decoding**
- Pre-built support for popular LLM architectures (LLaMA, Mistral, GPT, Falcon, Gemma, etc.)
- Easy Python API for building TensorRT engines for LLMs
- Integration with Triton via the TRT-LLM backend

### TRT-LLM Key Features

| Feature | Description |
|---|---|
| **In-Flight Batching** | Process requests mid-generation without waiting for batch completion |
| **Paged KV Cache** | Efficient memory management for attention key-value pairs (like OS paging) |
| **Speculative Decoding** | Small model predicts; large model verifies — faster token generation |
| **Chunked Prefill** | Split long context into chunks for better batching during prefill |
| **Multi-GPU Support** | Tensor parallelism and pipeline parallelism across multiple GPUs |
| **LoRA Support** | Serve multiple fine-tuned adapters from one base model at runtime |
| **Quantization** | INT8, FP8, INT4 with NVIDIA Model Optimizer |

### TRT-LLM Workflow:
```bash
# Step 1: Convert model to TensorRT engine
python convert_checkpoint.py --model_dir llama-3-8b --output_dir ./ckpt

trtllm-build --checkpoint_dir ./ckpt \
             --output_dir ./engine \
             --gemm_plugin float16 \
             --gpt_attention_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 2048 \
             --max_output_len 512

# Step 2: Serve with Triton (see Triton docs)
```

---

## 8. Key Tools in the TensorRT Ecosystem

| Tool | Purpose | Link |
|---|---|---|
| **trtexec** | CLI tool: benchmark, build engines, generate timing caches | Ships with TensorRT |
| **Polygraphy** | Debug, compare outputs across backends, constant folding | [GitHub](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) |
| **ONNX-GraphSurgeon** | Edit ONNX graphs: replace subgraphs, add plugins | [GitHub](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon) |
| **Model Optimizer** | Quantization, pruning, distillation for TensorRT deployment | [Docs](https://docs.nvidia.com/deeplearning/modelopt/index.html) |
| **Torch-TensorRT** | Compile PyTorch models with TensorRT | [GitHub](https://github.com/pytorch/TensorRT) |
| **Nsight Systems** | Profile TensorRT applications end-to-end | [Docs](https://docs.nvidia.com/nsight-systems/) |
| **GenAI-Perf** | Benchmark LLMs served by Triton | Ships with Triton |

---

## 9. Multi-Instance GPU (MIG)

MIG (available on Ampere A100 and later) partitions a single GPU into smaller, isolated GPU instances.

**Use with TensorRT:**
- If your model has **low GPU utilization**, MIG can run multiple smaller models on isolated GPU slices
- Each MIG slice has dedicated compute and memory with quality-of-service guarantees
- No interference between workloads on different MIG slices
- Optimal partitioning is application-specific

---

## 10. Multi-Device Inference (Preview)

TensorRT 10.x introduces multi-GPU inference support:

| Feature | Description | GPU Requirement |
|---|---|---|
| **DistCollective** | Distributed collective ops (AllReduce, AllGather, Broadcast) via NCCL | Ampere (SM80+) |
| **Multi-device attention** | Split KV sequence across GPUs with context parallelism | Blackwell (SM100+) |

This enables serving models too large for a single GPU.

---

## 11. Compatibility and Versioning

TensorRT uses **semantic versioning (MAJOR.MINOR.PATCH)**:
- MAJOR changes → incompatible API/ABI changes
- MINOR changes → backward-compatible additions
- PATCH changes → backward-compatible bug fixes

**Engine compatibility:**
- By default, engines are tied to: OS + CPU architecture + GPU model + TensorRT version
- **For forward compatibility:** Use version compatibility settings in the builder
- **Serialized engines are not compatible** across TensorRT versions by default
- **Calibration caches:** Generally reusable within a major version

**Deprecation policy:**
- 12-month migration period after any API is deprecated
- C++ APIs marked with `TRT_DEPRECATED_API` / `TRT_DEPRECATED` macros
- Python APIs issue `DeprecationWarning` at runtime

---

## 12. Memory Management

### Build Phase Memory
- TensorRT allocates device memory for timing kernel candidates
- Control via `IBuilderConfig::setMemoryPoolLimit()` (workspace size)
- At least two copies of weights in host memory during build

### Runtime Phase Memory
- Engine allocates device memory for model weights upon deserialization
- `ExecutionContext` uses two memory types:
  - **Persistent memory** — per-context state (e.g., convolution edge masks)
  - **Enqueue memory** — intermediate activations + scratch memory during inference
- Use `ICudaEngine::getDeviceMemorySizeV2()` to query required memory
- **Weight streaming** — for models larger than GPU memory, stream weights from host to device at inference time (latency tradeoff)

**Performance tip:** Enable CUDA lazy loading:
```bash
export CUDA_MODULE_LOADING=LAZY
```
This significantly reduces peak GPU and host memory usage and speeds up TensorRT initialization.

---

## 13. Threading Model

TensorRT objects are **not thread-safe** by default. The safe concurrency pattern:

✅ Safe patterns:
- Different threads → different `ExecutionContext` (from the same engine)
- Non-modifying operations on an engine from multiple threads
- Creating contexts from an engine in multiple threads
- Deserializing engines in parallel

❌ Not safe:
- Sharing a single `ExecutionContext` across threads
- Using multiple builders on the same GPU (timing interference)

---

## 14. Runtime Options (Library Selection)

| Library | Size | Use Case |
|---|---|---|
| `libnvinfer.so` | Full | Default; all operator implementations |
| `libnvinfer_lean.so` | Smaller | Run version-compatible engines only |
| `libnvinfer_dispatch.so` | Smallest | Shim that loads lean runtime; max compatibility |

Python equivalents: `tensorrt`, `tensorrt_lean`, `tensorrt_dispatch`

---

## 15. TensorRT in the Broader NVIDIA Stack

```
┌──────────────────────────────────────────────────────┐
│                    NVIDIA AI Stack                   │
│                                                      │
│  NeMo Platform                                       │
│  (Fine-tune model, generate synthetic data, eval)    │
│              ↓ produces weights                      │
│  TensorRT / TRT-LLM  ◄── YOU ARE HERE               │
│  (Optimize: quantize, fuse layers, tune kernels)     │
│  Input:  ONNX / PyTorch / HF model                  │
│  Output: Serialized TensorRT engine (.plan)          │
│              ↓ engine file                           │
│  Triton Inference Server                             │
│  (Load engine, serve via HTTP/gRPC, batch requests)  │
│              ↓ API                                   │
│  Your Application                                    │
└──────────────────────────────────────────────────────┘
```

---

## 16. Common Misconceptions

| Misconception | Correct Understanding |
|---|---|
| "TensorRT works on any GPU" | TensorRT requires NVIDIA GPUs; Turing (SM 7.5+) for best results |
| "TensorRT engines are portable" | By default, engines are GPU + TensorRT version specific |
| "Weak typing is fine for new projects" | Weak typing is deprecated in 10.12; use strong typing |
| "TensorRT can train models" | TensorRT is inference-only; use PyTorch/NeMo for training |
| "Quantization always hurts accuracy" | With proper calibration or QAT, INT8/FP8 accuracy loss is often negligible |
| "TRT-LLM is the same as TensorRT" | TRT-LLM is a higher-level library built on TensorRT specifically for LLMs |

---

*Last updated from official NVIDIA TensorRT documentation. For the latest, see: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html*
