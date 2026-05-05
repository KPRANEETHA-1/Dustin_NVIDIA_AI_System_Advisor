# NVIDIA NeMo Platform — RAG Knowledge Base

**Source:** [NVIDIA NeMo Platform Official Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html)  
**Category:** Model Customization, Evaluation, Guardrails, Deployment  
**Tool Identity:** NeMo is NVIDIA's end-to-end platform for building, fine-tuning, evaluating, and deploying AI models with enterprise-grade controls.

---

## 1. What is NeMo Platform?

NeMo Platform is NVIDIA's infrastructure layer for building and deploying **specialized AI agents** using open-source models. It is not a single model — it is a full platform that handles the entire AI development lifecycle:

- **Synthetic data generation** — create training data at scale
- **Model fine-tuning** — customize models for your domain
- **Evaluation** — benchmark and measure model quality
- **Security testing** — scan for vulnerabilities via NeMo Auditor
- **Guardrails** — real-time content moderation and safety
- **Inference** — serve models through a unified gateway

**Deployment options:** Docker (local) or Kubernetes (production)  
**Enterprise features:** RBAC (Role-Based Access Control), observability, audit logging

---

## 2. When to Use NeMo (Decision Guide)

Use NeMo when you need to:

| Use Case | NeMo Feature |
|---|---|
| Fine-tune an LLM on domain-specific data | Customizer + Data Designer |
| Evaluate model quality (accuracy, safety, RAG metrics) | Evaluator |
| Generate synthetic training data | Data Designer Service |
| Add safety filters to an LLM app | Guardrails |
| Fine-tune embedding models for RAG | Customizer + Evaluator |
| Manage AI resources with RBAC across teams | Workspaces + Roles |
| Scan AI agents for vulnerabilities | NeMo Auditor |

**NeMo vs Triton vs TensorRT — Quick Rule:**
- Need to *train or fine-tune* a model → **NeMo**
- Need to *serve/deploy* a model at scale → **Triton**
- Need to *optimize inference speed* on GPU → **TensorRT**
- Full pipeline: NeMo → TensorRT → Triton (this is the standard NVIDIA production stack)

---

## 3. Core Architecture Concepts

### 3.1 Workspaces
Workspaces are the **fundamental organizational boundary** in NeMo. Every resource (models, datasets, jobs, deployments) must belong to a workspace.

**Role levels:**
- `Viewer` — read-only access
- `Editor` — can create and modify resources
- `Admin` — full control including role management

**Workspace naming rules:**
- Must start with a lowercase letter (a–z)
- 2–63 characters long
- Allowed: lowercase letters, digits, hyphens
- No consecutive hyphens (`--`)
- Cannot end with a hyphen
- **Cannot be renamed after creation** — choose carefully

**Built-in workspaces:**
- `default` — general experimentation, editable by all authenticated users
- `system` — platform-provided resources (read-only for regular users)

**Best practice:** Use separate workspaces for true isolation (e.g., `team-ml-research`, `env-production`, `client-acme`). For lightweight grouping within a workspace, use **Projects**.

### 3.2 Entities
Entities are the underlying data objects for all NeMo resources. Every model, dataset, job, and config is stored as an entity.

- All entities share common metadata: `name`, `workspace`, `timestamps`, `custom fields`
- Entity names are unique **within** a workspace (same name can exist in different workspaces)
- Accessed via service-specific APIs (Customizer, Evaluator, Models) — not a generic entity API

### 3.3 Secrets Management
NeMo provides encrypted secret storage for API keys (e.g., Hugging Face tokens, Weights & Biases keys).

- Secrets are **encrypted at rest**
- Once created, the secret value **cannot be retrieved** through the API — not even by Admins
- Only **Platform Administrators** can retrieve secret values
- Referenced in configs as `workspace/secret_name`

### 3.4 File Storage (Filesets)
A **Fileset** is a named container for files, used for datasets, model artifacts, and evaluation results.

**Supported storage backends:**

| Backend | Type | Description |
|---|---|---|
| `local` | Read/Write | Default local filesystem storage |
| `s3` | Read/Write | Amazon S3 or S3-compatible (e.g., MinIO) |
| `ngc` | Read-Only | NVIDIA GPU Cloud storage |
| `huggingface` | Read-Only | HuggingFace Hub repositories |

---

## 4. Model Registry and Inference Gateway

### 4.1 Core Objects

**ModelDeploymentConfig** — A *versioned blueprint* for deploying a NIM (NVIDIA Inference Microservice) container. Specifies:
- GPU count
- Container image
- Model name
- Optional: LoRA support, chat templates, tool calling, custom environment variables
- **Reusable:** multiple deployments from one config; updates create a new version without affecting existing deployments

**ModelDeployment** — A *running instance* of a NIM container.  
Lifecycle states: `CREATED → PENDING → READY | FAILED`  
When `READY`, a `ModelProvider` is automatically created.

**ModelProvider** — A *routable inference host*. All inference requests are routed through a ModelProvider, which can serve one or more Models.

**Model** — A registered model entity (e.g., `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`). Can be:
- Hosted locally via NIM
- External (NVIDIA Build, OpenAI, etc.)

### 4.2 Inference Flow
```
User Request → Unified Gateway → ModelProvider → ModelDeployment (NIM Container) → GPU
```

---

## 5. NeMo Guardrails

**Purpose:** Apply safety checks and content moderation to LLM applications.

**How it works:**
1. User request arrives at the Guardrails OpenAI-compatible endpoint
2. Guardrails evaluates the input against configured policies
3. If safe, request is forwarded to the inference model
4. Model output is evaluated before returning to the user
5. Blocked content is handled according to the guardrail config

**Use cases:**
- Content safety (toxicity, hate speech)
- Topic control (keep models on-topic)
- PII protection
- Jailbreak detection

**Key distinction:** Guardrails is about *protecting* an LLM application in production. NeMo Auditor is about *scanning* AI agents for vulnerabilities before deployment.

---

## 6. NeMo Evaluator

**Purpose:** Measure the quality of LLMs, RAG pipelines, and AI agents at scale.

### 6.1 Core Primitives

| Primitive | Purpose | When to Use |
|---|---|---|
| **Metrics** | Flexible scoring logic for model outputs | Custom datasets, task-specific criteria |
| **Benchmarks** | Metrics + Dataset paired together | Standardized comparisons, regression testing |

### 6.2 Execution Modes

| Mode | API | Best For |
|---|---|---|
| Live (synchronous) | `POST /v2/workspaces/{workspace}/evaluation/metric-evaluate` | Fast iteration, small payloads |
| Jobs (asynchronous) | `/evaluation/metric-jobs` | Production workloads, large datasets |

### 6.3 Evaluation Patterns

- **Offline evaluation** — score pre-existing model outputs (already generated)
- **Online evaluation** — generate outputs from a model and score them in one pipeline

### 6.4 Recommended Evaluation Journey

```
1. Develop metrics → 2. Validate with live eval → 3. Scale to async jobs 
→ 4. Package into benchmarks → 5. Monitor and track regression
```

**Supported out of the box:** 100+ industry benchmarks, LLM-as-a-judge, RAG metrics, agent metrics

---

## 7. Data Designer Service

**Purpose:** Orchestrate complex synthetic data generation workflows at scale.

**Core capability:** Coordinates LLM calls, manages dependencies between data fields, handles batching and parallelization, and validates generated data against specifications.

**Typical workflow:**
```
Define schema → Generate synthetic examples (LLM-powered) → Validate → Export as fileset → Use for fine-tuning
```

---

## 8. NeMo Studio

NeMo Studio is the **web UI** for NeMo Platform — a user-friendly interface that surfaces all NeMo capabilities visually. It is built on top of the same APIs available via CLI and Python SDK.

---

## 9. NeMo in the Broader NVIDIA Stack

```
┌─────────────────────────────────────────┐
│            NVIDIA AI Stack              │
│                                         │
│  NeMo Platform (Training/Fine-tuning)   │
│       ↓ Produces optimized model        │
│  TensorRT / TRT-LLM (Optimization)      │
│       ↓ Creates TensorRT engine         │
│  Triton Inference Server (Serving)      │
│       ↓ Exposes HTTP/gRPC endpoints     │
│  Your Application                       │
└─────────────────────────────────────────┘
```

**NeMo handles the left side of the pipeline** (data → training → evaluation → guardrails).  
**Triton + TensorRT handle the right side** (optimization → serving → monitoring).

---

## 10. Key NeMo API References

| Action | Endpoint / Method |
|---|---|
| List workspaces | `GET /v2/workspaces` |
| Create workspace | `POST /v2/workspaces` |
| Live metric evaluation | `POST /v2/workspaces/{workspace}/evaluation/metric-evaluate` |
| Submit metric job | `POST /v2/workspaces/{workspace}/evaluation/metric-jobs` |
| Submit benchmark job | `POST /v2/workspaces/{workspace}/evaluation/benchmark-jobs` |
| Register model provider | `POST /v2/workspaces/{workspace}/models/providers` |

---

## 11. Common Misconceptions

| Misconception | Correct Understanding |
|---|---|
| "NeMo is just a model" | NeMo is a full platform — it includes data, training, eval, guardrails, and inference |
| "I need NeMo to serve models" | Triton is the primary serving layer; NeMo provides the gateway and management layer |
| "Guardrails is optional for production" | For enterprise deployments with user-facing LLMs, guardrails is a critical safety component |
| "NeMo only works with NVIDIA models" | NeMo supports open-source models from HuggingFace and external providers like OpenAI |

---

*Last updated from official NVIDIA NeMo documentation. For the latest, see: https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html*
