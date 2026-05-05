# NVIDIA Triton Inference Server — RAG Knowledge Base

**Source:** [NVIDIA Triton Inference Server Official Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)  
**Category:** Model Serving, Inference, Deployment  
**Tool Identity:** Triton is NVIDIA's open-source inference serving platform — it takes any trained/optimized model and exposes it as a production-grade HTTP/gRPC API.

---

## 1. What is Triton Inference Server?

NVIDIA Triton Inference Server is an **open-source inference serving software** that streamlines AI model deployment. It acts as the *serving layer* in the NVIDIA AI stack — taking models from frameworks like TensorRT, PyTorch, ONNX, and others, and exposing them as standardized API endpoints.

**Core value proposition:** Deploy *any* model, from *any* framework, to *any* hardware, behind a *single*, standardized API.

**Part of:** NVIDIA AI Enterprise — the production-grade AI software platform.

---

## 2. When to Use Triton (Decision Guide)

Use Triton when you need to:

| Use Case | Triton Capability |
|---|---|
| Serve multiple models simultaneously | Concurrent model execution |
| Handle bursts of inference requests | Dynamic batching |
| Serve models across GPU, CPU, or edge devices | Multi-hardware support |
| Connect to Kubernetes / production infrastructure | Health endpoints + metrics |
| Build model pipelines (chained inference) | Ensembling + Business Logic Scripting |
| Serve stateful models (e.g., speech, video) | Sequence batching |
| Benchmark LLM throughput and latency | GenAI-Perf tool |

**Triton vs NeMo vs TensorRT — Quick Rule:**
- Need to *train or fine-tune* a model → **NeMo**
- Need to *serve/deploy* a model with a standard API → **Triton**
- Need to *optimize GPU inference speed* → **TensorRT** (then serve with Triton)
- Common pattern: TensorRT optimizes the engine → Triton serves it → clients query via HTTP/gRPC

---

## 3. Supported Frameworks (Backends)

Triton supports a wide range of ML/DL backends out of the box:

| Backend | Framework |
|---|---|
| TensorRT | NVIDIA's optimized inference engine |
| PyTorch (TorchScript) | Meta's deep learning framework |
| ONNX Runtime | Cross-framework open standard |
| TensorFlow | Google's ML framework |
| OpenVINO | Intel's inference optimization toolkit |
| Python | Custom Python-based backends |
| RAPIDS FIL | NVIDIA's ML inference for tree models |
| TensorRT-LLM | Optimized LLM inference (via TRT-LLM backend) |

**Key insight:** This multi-backend support is what differentiates Triton. You can serve a TensorRT-optimized LLM and a PyTorch vision model from the *same* Triton instance simultaneously.

---

## 4. Triton Architecture

```
                    ┌──────────────────────────────────┐
                    │      Triton Inference Server      │
                    │                                  │
  HTTP/REST ───────►│                                  │
  gRPC      ───────►│  Per-Model Scheduler             │
  C API     ───────►│    ├─ Static Batching            │
                    │    ├─ Dynamic Batching            │
                    │    └─ Sequence Batching           │
                    │         ↓                        │
                    │  Backend Dispatcher               │
                    │    ├─ TensorRT Backend            │
                    │    ├─ PyTorch Backend             │
                    │    ├─ ONNX Backend                │
                    │    └─ TRT-LLM Backend             │
                    │         ↓                        │
                    │  GPU / CPU / AWS Inferentia       │
                    └──────────────────────────────────┘
                              ↑
                    Model Repository (filesystem)
```

**Model Repository:** A file-system-based directory where all model artifacts are stored. Triton reads from this on startup and can be dynamically updated via the Model Management API.

**Request flow:**
1. Inference request arrives via HTTP/REST, gRPC, or C API
2. Routed to the per-model scheduler
3. Scheduler batches requests (if configured)
4. Backend performs GPU/CPU inference
5. Outputs returned to the client

---

## 5. Key Features In Depth

### 5.1 Dynamic Batching
Triton can automatically group incoming requests into batches to maximize GPU utilization, even when clients send individual requests. This is critical for LLM serving where batching dramatically improves throughput.

### 5.2 Sequence Batching
For stateful models (speech recognition, video, multi-turn conversations), Triton maintains **implicit state** across a sequence of requests from the same client. The model doesn't need to re-receive the full context each time.

### 5.3 Concurrent Model Execution
Multiple instances of the same model (or entirely different models) can run simultaneously on the same GPU or across multiple GPUs. Triton manages resource allocation automatically.

### 5.4 Ensemble Pipelines
Chain multiple models together in a **pipeline**. For example:
```
Input → Preprocessing Model → LLM → Postprocessing Model → Output
```
Triton handles the data flow between models, eliminating the need for client-side orchestration.

### 5.5 Business Logic Scripting (BLS)
For pipelines that require conditional logic (e.g., "if the first model's output exceeds a threshold, call model B"), BLS allows writing Python-based orchestration logic that runs inside Triton.

### 5.6 Backend C API
Triton's extensible architecture allows adding **custom backends** — custom pre/post-processing operations or entirely new inference frameworks — via the Backend C API.

---

## 6. TensorRT-LLM + Triton Integration

This is the **recommended production stack for serving LLMs** with NVIDIA hardware:

### Step-by-Step Integration:
```
Step 1: Convert model to TensorRT engine using TRT-LLM
         (handles quantization, kernel fusion, KV cache optimization)

Step 2: Prepare model repository for Triton
         (directory structure + config.pbtxt)

Step 3: Launch Triton with TRT-LLM backend
         (use NGC Triton TRT-LLM container)

Step 4: Send inference requests via HTTP/gRPC
```

### Advanced LLM Configuration Options (via TRT-LLM backend):

| Feature | Purpose |
|---|---|
| **KV Cache** | Reuse attention key-value pairs across requests to reduce memory bandwidth |
| **Chunked Context** | Split long contexts into chunks; batch during generation for higher throughput |
| **Speculative Decoding** | Use a small draft model to predict tokens; verify with large model for speedup |
| **MIG Support** | Run multiple model instances on GPU partitions for isolation and efficiency |
| **Quantization** | INT8/FP8/INT4 to reduce model size and increase throughput |
| **LoRA** | Serve multiple fine-tuned adapters from a single base model |
| **Beam Search** | For tasks requiring multiple candidate outputs (translation, code gen) |

---

## 7. Supported Hardware

Triton runs on a wide range of hardware:

| Hardware | Support |
|---|---|
| NVIDIA GPUs (all modern) | Full support, best performance |
| x86 CPU | Supported via ONNX Runtime / OpenVINO |
| ARM CPU | Supported (edge deployments) |
| AWS Inferentia | Supported |
| NVIDIA SoCs (Jetson) | Supported via C API and ARM builds |

---

## 8. Protocols and APIs

| Protocol | Use Case |
|---|---|
| **HTTP/REST** | Standard web clients, easy integration |
| **gRPC** | High-performance, low-latency clients |
| **C API** | Embedded/edge deployments, in-process use |
| **Java API** | JVM-based applications |
| **KServe Protocol** | Community standard for interoperability |

---

## 9. Observability and Operations

Triton provides **production-grade observability** out of the box:

- **Readiness endpoint** — Is Triton ready to serve? (for Kubernetes)
- **Liveness endpoint** — Is Triton alive? (for Kubernetes health checks)
- **Metrics endpoint** — Prometheus-compatible metrics:
  - GPU utilization
  - Server throughput (requests/sec)
  - Server latency (p50, p90, p99)
  - Queue time, compute time, transfer time
  - Per-model statistics

**GenAI-Perf:** Command-line tool for benchmarking LLM throughput and latency on Triton. Part of the Triton toolkit.

```bash
# Example: Benchmark a model served by Triton
genai-perf --model-name llama-3 --url localhost:8000 --concurrency 10
```

---

## 10. Deployment Patterns

### 10.1 Single-Model Deployment
One model, one Triton instance. Simple, for focused applications.

### 10.2 Multi-Model Deployment
Multiple models served by one Triton instance. GPU resources shared intelligently.

### 10.3 Kubernetes Deployment
Triton integrates naturally with Kubernetes:
- Readiness/liveness probes → Kubernetes health checks
- Prometheus metrics → Grafana dashboards
- Horizontal Pod Autoscaler support
- Helm charts available via NGC

### 10.4 Edge Deployment
For resource-constrained environments (Jetson, embedded), use the C API or ARM builds. Triton can run in-process with no network overhead.

---

## 11. Model Management API

Triton exposes a dedicated API for managing models at runtime (via HTTP/REST, gRPC, or C API):

- **Load/unload models** without restarting Triton
- **Query model status** (is it ready, loading, unloading?)
- **Get model metadata** (input/output shapes, data types)
- **Get server statistics**

---

## 12. Triton in the Broader NVIDIA Stack

```
┌──────────────────────────────────────────────────┐
│                  NVIDIA AI Stack                 │
│                                                  │
│  NeMo Platform                                   │
│  (Fine-tune, evaluate, generate synthetic data)  │
│              ↓                                   │
│  TensorRT / TRT-LLM                              │
│  (Optimize model: quantize, fuse, build engine)  │
│              ↓                                   │
│  Triton Inference Server  ◄── YOU ARE HERE       │
│  (Serve model at scale: HTTP/gRPC, batching,     │
│   multi-model, Kubernetes-ready)                 │
│              ↓                                   │
│  Your Application / API Gateway                  │
└──────────────────────────────────────────────────┘
```

---

## 13. Common Misconceptions

| Misconception | Correct Understanding |
|---|---|
| "Triton only works with TensorRT" | Triton supports 10+ backends including PyTorch, ONNX, Python, etc. |
| "I need Triton for local development" | Triton is for production serving; for dev, just use the framework's native inference |
| "Triton replaces TensorRT" | Triton *serves* models; TensorRT *optimizes* them — complementary, not competing |
| "Triton only works on GPUs" | Triton supports x86 CPU, ARM, AWS Inferentia, and edge devices |
| "I have to restart Triton to load a new model" | The Model Management API allows hot-loading models at runtime |

---

## 14. Quick Reference: Configuration File (config.pbtxt)

Every model in Triton's repository needs a `config.pbtxt`:

```protobuf
name: "my_llm"
backend: "tensorrtllm"
max_batch_size: 8

input [
  { name: "input_ids", data_type: TYPE_INT32, dims: [-1] }
]
output [
  { name: "output_ids", data_type: TYPE_INT32, dims: [-1] }
]

instance_group [
  { count: 1, kind: KIND_GPU }
]

dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 100
}
```

---

*Last updated from official NVIDIA Triton documentation. For the latest, see: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html*
