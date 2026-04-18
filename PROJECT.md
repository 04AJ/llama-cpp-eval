==QUESTIONS==
* Is it okay if I create a fork on the github and submit that with all my build instructions
	* yes, but also include instructions in final report
* For the improvement and extension part of the project: could we work on an open issue on the github and submit a PR?
	* yes, that is the goal
* Is the following strategy okay? --> CPU only is fine but need to explain the optimizations (GGML and GGUF)
	* Set up/implementation on NOTS (cpu only)
	* Evaluation on NOTS (cpu only)
	* Deployment on AWS
		*==Docker --> ECS --> Access Via Web==
* **When is the final deadline for the project? Is it okay we we're not done with all the parts before the presentation?**
	* Which parts of the project are required to be complete by the presentation date?
		* Need to have the implementation done
		* Need to show evaluations
		* Need to show how our implementation improved the evaluation performance
* Deployment is NOT necessary for final project
	

# Project Exploration Assignment Guidelines  
COMP 468/568 – Spring 2026

In this assignment, you will explore an existing open-source GitHub project related to Deep Learning Systems (e.g., training systems, inference engines, compilers, distributed ML, or system optimizations). The goal is to understand a real-world ML systems artifact, reproduce its results, and critically analyze its design and performance.

## 1. Project Signup

Each student must sign up for ONE project using the provided Google Sheet. [https://docs.google.com/spreadsheets/d/1A6mRKnstF_ZnnTlxThOGPJXnHrSnx1dldHg4RKAxhNk/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1A6mRKnstF_ZnnTlxThOGPJXnHrSnx1dldHg4RKAxhNk/edit?usp=sharing)

  
You must:  
- Select a GitHub repository related to deep learning systems  
- Enter your name(s) and the project link in the Google Sheet  
- Ensure no duplicate selections unless explicitly allowed  
  
Once selected, you are responsible for exploring and evaluating that artifact.

---

## 2. Background and Artifact Description

Provide background information about the selected project:  
  
### 1. What problem does this system address?

Before `llama.cpp`, running Large Language Models (LLMs) required **massive VRAM** (Video RAM) and specialized enterprise GPUs (like NVIDIA A100s). The "average" developer or enthusiast with a consumer laptop or a Mac was effectively locked out of running these models locally due to:
- **Dependency Bloat:** Standard frameworks (like PyTorch or TensorFlow) have massive installation footprints and Python overhead.
- **Hardware Barriers:** Most LLM implementations were "GPU-only" or extremely slow on CPUs.
- **Memory Footprint:** 16-bit or 32-bit models are too large for consumer RAM.
`llama.cpp` solved this by providing a **zero-dependency, C++ implementation** optimized for **CPU-first inference** and **quantization**, making it possible to run a 7B or 13B parameter model on a standard MacBook or even a Raspberry Pi.
### 2. What type of artifact is it?
`llama.cpp` is primarily a **high-performance inference runtime (or engine)** that runs on GGML (core math and tensor library).
- **Runtime:** It manages the execution of the model's compute graph, handles memory allocation, and interfaces with hardware backends (CPU, CUDA, Metal, etc.).
- **Library:** It is often used as a backend library for other applications (like Ollama or LM Studio).
- **Toolkit:** It includes conversion and quantization tools to transform raw model weights (e.g., PyTorch `.bin` files) into the efficient **GGUF** format.
### 3. Key Ideas and System Contributions
The project introduced several breakthroughs that changed how the community interacts with AI:
- **GGUF Format:** A self-contained file format that stores weights, vocabulary, and all necessary metadata (like chat templates) in a single file, ensuring portability across different systems.
- **Aggressive Quantization:** It pioneered the use of 4-bit, 3-bit, and even 2-bit quantization (K-Quants), allowing models to fit into a fraction of their original size with minimal loss in "intelligence."
- **Unified Memory & Heterogeneous Computing:** It allows for "partial offloading," where some layers of a model run on the GPU while the rest run on the CPU/RAM, maximizing the use of all available hardware.
- **Static Compute Graphs:** Unlike dynamic frameworks, it builds a static graph of the model's math operations, reducing overhead during the actual generation of text.
### 4. Relevance to Deep Learning System Design
`llama.cpp` is a masterclass in **systems-level optimization**. It is relevant to deep learning design for several reasons:
1. **Hardware Abstraction:** It demonstrates how to write a single codebase that can target diverse backends (ARM NEON, x86 AVX, Apple Metal, NVIDIA CUDA, and Vulkan) without sacrificing performance.
2. **Memory Management:** Its custom heap and buffer management show how to handle massive tensors in constrained environments where traditional `malloc` would be too slow or fragmented.
3. **The "Minimalist" Philosophy:** It challenges the "Python-heavy" status quo of deep learning, proving that removing abstraction layers can lead to 10x or 100x improvements in deployment efficiency.
  
Cite the GitHub repository and any associated papers or documentation.
==https://github.com/ggml-org/llama.cpp==

---
## 3. Code Setup and Environment
Describe how you set up the project so others can reproduce your work. 
* Utilized SLURM batch script to automate the build and deployment process. This ensures that the environment is consistent and that the compute resources are properly allocated.
Include:  
- Hardware used (CPU/GPU type, memory) 
	- **CPU:** 80 cores
	- **Memory:** 32GB total RAM (`4G` per CPU core)
	- **Acceleration:** CPU-only mode (No GPU/CUDA used)
	- **Networking:** Port 8080 is exposed on the compute node for local tunneling
- Software environment (OS, Python version, framework versions)  
	- **Operating System:** Linux (RHEL)
	- **Compiler:** `GCCcore/14.3.0`.
	- **Build System:** `CMake` (handles the compilation and dependency resolution for `llama.cpp`)
	- **Version Control:** `git/2.50.1`
	- **Framework:** `llama.cpp` (compiled from source)
	- **Models: Used several to apply memory bandwidth pressure and increase compute time**

| **Model Family**  | **Version**  | **Parameters** | **Quantization** | **File Name**                                 |
| ----------------- | ------------ | -------------- | ---------------- | --------------------------------------------- |
| **Meta Llama**    | 3.2 Instruct | 1B             | `Q4_K_M`         | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`           |
| **Meta Llama**    | 3.2 Instruct | 3B             | `Q8_0`           | `bartowski_Llama-3.2-3B-Instruct...Q8_0.gguf` |
| **Meta Llama**    | 3.1 Instruct | 8B             | `Q8_0`           | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`        |
| **Google Gemma**  | 3-1b-it      | 1B             | `Q4_K_M`         | `ggml-org_gemma-3-1b-it...Q4_K_M.gguf`        |
| **DeepSeek/Qwen** | R1-0528      | 8B             | _(Unspecified)_  | `DeepSeek-R1-0528-Qwen3-8B-GGUF.gguf`         |
|                   |              |                |                  |                                               |

- Installation steps and dependencies  
	- The script creates a persistent cache directory in your project space to store the model and Hugging Face assets.
- Any issues encountered during setup and how you resolved them
	- During initial deployment, the application defaulted to the user's home directory (~/.cache), threatening disk quota limits. 
		- [Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) wasn't clear about setting up llama.cpp cache directory for `llama-server`
		- **Solution**: [`LLAMA_CACHE`](https://github.com/ggml-org/llama.cpp/pull/7826) environment variable 
		
	

---
## 4. Evaluation Platform
Clearly state where and how the project was evaluated:  
- Platform
	- The project was evaluated on the NOTs cluster
- Number of GPUs or nodes used  
	- No GPUs, 80 `Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz` were used
- Any platform-specific constraints or configurations  
	* V100 is an older architecture and doesn't support new cutting edge features in `llama.cpp`
Explain why this platform is suitable for evaluating the artifact.
- `llama-bench` is directly supported by the developers of the `llama.cpp` project
	- Can perform three types of tests: prompt processing, text generation, prompt processing + text generation
- `llama-perplexity` can be used to evaluate how well the model predicts the next token
---
## 5. Implementation and Deployment
==DockerFile --> ECS --> Access Via Web==
Explain how you ran, integrated, or modified the code.  
Include:  
- Which scripts or entry points were used  
- Any configuration changes or code modifications you made  
- How the system was executed (training, inference, benchmarking)  
- Deployment details if applicable (e.g., containers, distributed launch)

---
## 6. Screenshots of Results
Include 2–4 screenshots demonstrating the artifact in action.  
Examples:  
- Execution logs or terminal output  
- Performance metrics (runtime, throughput, memory)  
- Profiler timelines or dashboards  
- Training or inference progress  
  
Each screenshot must be readable, captioned, and referenced in the report.

---
## 7. Results
Summarize the main results you obtained.  
Include:  
- Key performance metrics  
- Comparison to any baseline provided by the project  
- Whether your results match those reported by the authors  
  
Use tables or figures if helpful.

---
## 8. Observations
Discuss what you observed from running and evaluating the system.  
  
Examples:  
- Performance trends or bottlenecks  
- Resource utilization behavior  
- Sensitivity to configuration choices  
  
Focus on explaining why the system behaves the way it does.

---
## 9. Improvements and Extensions
==Cosmetic Refactoring Issue==: https://github.com/ggml-org/llama.cpp/issues/5239

==Better Issue: [https://github.com/ggml-org/llama.cpp/issues/19110](https://github.com/ggml-org/llama.cpp/issues/19110)==

==Even Better Issue: https://github.com/ggml-org/llama.cpp/issues/9086==
## How to Reproduce Error default error
```bash
timeout 10 /projects/comp468/aj162/src/llama.cpp/build/bin/llama-server \

  -m "/projects/comp468/aj162/cache/llama.cpp/Llama-3.2-1B-Instruct-Q4_K_M.gguf" --threads -1 \

  2>&1 | grep -i "n_threads\|threads ="
```
* This will spawn 80 threads instead of the optimal 40

## Solution
	* Solution is simple (one-liner)
* **Deeper Issue**
	* llama.cpp has no awareness of NUMA topology at all.
		When it spawns threads, it has no idea your machine has two sockets with separate memory controllers. The OS scheduler is free to migrate threads across sockets, and model weights loaded into one socket's RAM get accessed by cores on the other socket over QPI — paying the ~1.7× latency penalty on every single weight read, for every single token generated.

Layer 1 — thread count        (trivial, already in source)
           hardware_concurrency() → cpu_get_num_physical_cores()

Layer 2 — thread placement    (medium, ~30-60 lines)
           pthread_setaffinity_np() to pin each worker
           to a specific physical core on one socket

Layer 3 — memory placement    (harder, ~100-200 lines)
           mbind() / numa_alloc_onnode() to ensure
           model weights live in the same socket's RAM
           as the threads consuming them

---
## COMP468 Project: NUMA-Aware Inference Optimization in llama.cpp

### Motivation
- Starting point: llama.cpp [issue #19110](https://github.com/ggml-org/llama.cpp/issues/19110)
    --threads -1 double counts logical cores via hardware_concurrency()
    due to hyper-threading, causing performance regression
- Reproducing on dual-socket Xeon Gold 6230 revealed a deeper problem:
    the thread count bug is a symptom of llama.cpp having no NUMA
    topology awareness at all
- This connects to a broader open request: [issue #9086](https://github.com/ggml-org/llama.cpp/issues/9086)(tensor
    parallelism on multi-socket Xeon CPUs) — our fixes are the
    foundational layer that TP would build on

### 1. Reproducing issue #19110
- Benchmark sweep: llama-bench -t 8,16,32,40,48,64,80
- Show performance cliff when threads cross socket boundary (32→40)
- Identify hardware_concurrency() vs cpu_get_num_math() inconsistency:
    llama-bench uses cpu_get_num_math() correctly
    llama-cli and llama-server use hardware_concurrency() incorrectly

### 2. Fix 1 — thread count default (addresses issue #19110 directly)
- Source: common/arg.cpp lines 1114, 1124, 3219, 3229
- Change: hardware_concurrency() → cpu_get_num_physical_cores()
- Evidence: two-binary llama-bench comparison, n_threads column proves
    different default decisions from identical invocations

### 3. Root cause: NUMA topology blindness
- Fix 1 corrects the thread count but threads still scatter across
    both sockets — OS scheduler migrates workers freely across QPI
- ggml-cpu.c line 1382: existing comment acknowledges NUMA needs
    special work partitioning but it was never implemented
- Even with --numa isolate (thread affinity), weight rows are still
    assigned to threads with no regard for which socket owns that memory

==Background on the topic==
* QPI (Quick Path Interconnect) or UPI is the high-speed link between the two sockets
* NUMA (Non Uniform Memory Access)
	* Not all RAM is equally fast for all cores depending on distance
	* OS groups cores + attached RAM into NUMA nodes 
	* Node 0 = socket 0 + its RAM
	* Node 1 = socket 1 + its RAM
* Why this is a problem
	* when running inference the model weights (giant matrix) gets loaded to RAM and the OS scheduler does whatever it wants (thread assignment is random)
* `--numa isolate` flag across the llama tools that tells OS to **pin each thread to a specific NUMA node** using `mbind` or `set_mempolicy` on linux
	* scheduler is
* ==Interesting note==: Evaluating on large 8B model seemed to resolve the issue vs the 1B model i initially tested on 
	* 1B is compute bound (threads fighting over synchronization)
	* 8B is memory bandwidth bound (model is too large for any cache)
### 4. Fix 2 — NUMA-aware row partitioning in ggml_mul_mat
- Source: ggml/src/ggml-cpu/ggml-cpu.c
- Changes:
    * Add numa_node field to ggml_compute_state struct
    * Set numa_node per thread at spawn in ggml_threadpool_new_impl
    * Modify ir0/ir1 range calculation to partition weight rows by node
- Evidence: before/after llama-bench --numa isolate, same thread count

==Notes==
* `ggml-cpu.c` contains the code for the `ggml_mul_mat`
* `ggml_compute_state` is the per-thread state object
	* stores which thread pool it belongs to, its index, its CPU mask
	* **doesn't have concept of NUMA node**
* `numa_node` field in ggml_compute_state — gives each thread a permanent identity: "I belong to node 0" or "I belong to node 1". Without this, the row-partitioning logic has no way to know which socket a given thread is running on.
* Set `numa_node` per thread in `ggml_threadpool_new_impl` — at spawn time, each thread's node is derived from its CPU affinity. Threads 0–19 get numa_node=0, threads 20–39 get numa_node=1.
	* this makes such each thread is assigned to data on it's node
* Clamp `ir0` range by node — inside ggml_mul_mat, instead of letting work-stealing assign any row chunk to any thread, the ir0_start/ir0_end range is intersected with the node's slice of the weight matrix:
* ![[Pasted image 20260313232844.png]]

### 5. Fix 3 — NUMA-local wdata buffer via mbind() (stretch goal)
> NOTE: this is only meaningful if we utilize `numa distribute`
- Source: ggml/src/ggml-cpu/ggml-cpu.c ~line 3287
- Change: bind wdata scratch buffer to local NUMA node after malloc()
- Evidence: stddev_ns improvement (consistency) on top of Fix 2

==Note: Also stress test with bigger models to show diff==
### 6. Evaluation
- Baseline:      hardware_concurrency() default, no NUMA flags
- Fix 1 only:    correct thread count
- Fix 2 only:    NUMA row partitioning + --numa isolate
- Fix 1 + 2:     combined
- Fix 1 + 2 + 3: full improvement
- Metrics:       avg_ts, avg_ns (ms/token), stddev_ns

### 7. Future work — tensor parallelism (issue #9086)
- Fix 2 + Fix 3 establish the infrastructure TP requires:
    each node owns its compute rows + its scratch memory
- Next: bind model weight tensors to NUMA nodes at load time
- Beyond: split weights across nodes + all-reduce after each layer
- This project is the first step toward closing issue #9086
---
Describe any improvements or optimizations you attempted or proposed.  
You may include:  
- Changes you implemented and their impact  
- Optimizations you attempted but did not finish  
- Ideas for future improvements based on your observations  
  
Clearly state what was implemented versus what is proposed.

---
## 10. Code Submission (Required)
You must submit:  
- A link to the original GitHub project  
- Your modified code or scripts (if any)  
- Clear instructions to reproduce your results  
  
All code must run without errors. Missing or non-runnable code will result in reduced credit.

Note: Failure to sign up in the Google Sheet, missing screenshots, or missing code submission will be considered an incomplete assignment.