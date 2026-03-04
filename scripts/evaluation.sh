#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --ntasks=1 
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --cpus-per-task=80   
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# Load modules
module load CUDA
module load GCCcore/14.3.0       
module load git/2.50.1

# Setup environment
BASE_DIR="/projects/comp468/aj162/"
mkdir -p $BASE_DIR/cache/llama.cpp
mkdir -p "${BASE_DIR}/evaluation"

PROJECT_DIR="${BASE_DIR}/src/llama.cpp"
EVAL_PROJECT_DIR="$HOME/llama-cpp-eval"  

# Environment variables
export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# Configure the build
cd $PROJECT_DIR
# rm -rf build

cmake -B build

# Compile using all 8 CPU cores
cmake --build build --config Release -j 8

# Download models for evaluation
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_FILE="${BASE_DIR}/cache/llama.cpp/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Download only if the file doesn't exist
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model..."
    wget -c "$MODEL_URL" -O "$MODEL_FILE"
else
    echo "Model already exists, skipping download."
fi


# llama-bench can perform three types of tests:

# Prompt processing (pp): processing a prompt in batches (-p)
# Text generation (tg): generating a sequence of tokens (-n)
# Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (-pg)

cd build/bin/
./llama-bench -m "$MODEL_FILE" -t 8,16,32,40,48,64,80 -p 512 -n 128 -o csv > "${EVAL_PROJECT_DIR}/evaluation/recreate_bug_${SLURM_JOB_ID}.csv"

echo "Benchmark complete. Results saved to: ${EVAL_PROJECT_DIR}/thread-bug_${SLURM_JOB_ID}.csv"