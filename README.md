# Deep Learning Systems Artifact Reproduction

## Code Setup and Environment
### Hardware Used

- **CPU:** 8 cores
- **Memory:** 32GB total RAM (`4G` per CPU core)
- **Acceleration:** CPU-only mode (No GPU/CUDA used)
- **Networking:** Port 8080 is exposed on the compute node for local tunneling

### Software Environments

- **Operating System:** Linux (RHEL)
- **Compiler:** `GCCcore/14.3.0`
- **Build System:** `CMake` (handles the compilation and dependency resolution for `llama.cpp`)
- **Version Control:** `git/2.50.1`
- **Framework:** `llama.cpp` (compiled from source)
- **Model:** `gemma-3-1b-it-GGUF` (quantized model from Google via Hugging Face)

### Installation steps and dependencies
- Edit the `BASE_DIR` variable in `scripts/build.sh` to point to your allocated project space. This redirects large model weights and Hugging Face assets away from your home directory to stay within disk quotas.
- Clone [my fork](https://github.com/04AJ/llama.cpp) into `BASE_DIR/src`
- Submit the Job: Launch the build and server process:
```bash
    sbatch scripts/build.sh
```
- Establish connectivity: Check the generated log file (e.g., `logs/build_<jobid>.log`) to retrieve the compute node hostname and the specific SSH tunneling command.
- Access the Interface: Run the tunneling command in a local terminal, then navigate to http://localhost:8080 in your web browser to interact with the model

### Issues encountered
- During initial execution, the application defaulted to the user's home directory (`~/.cache`), threatening disk quota limits. [Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) wasn't clear about setting up llama.cpp cache directory for `llama-server`
    - **Solution**: [`LLAMA_CACHE`](https://github.com/ggml-org/llama.cpp/pull/7826) environment variable 