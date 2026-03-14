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
source "$HOME/llama-cpp-eval/build/config.sh"
mkdir -p $BASE_DIR/cache/llama.cpp

export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# ---------------------------------------------------------------------------
# Model selection — set MODEL to a name or number from build/models.sh
#   1  gemma-1b   ~0.6 GB
#   2  llama-1b   ~0.7 GB
#   3  llama-3b   ~3.3 GB
#   4  llama-8b   ~8.5 GB
# ---------------------------------------------------------------------------
MODEL="llama-1b"
source "$HOME/llama-cpp-eval/build/models.sh"

# Configure and build
cd $PROJECT_DIR
cmake -B build
cmake --build build --config Release -j 8


# llama-bench can perform three types of tests:

# Prompt processing (pp): processing a prompt in batches (-p)
# Text generation (tg): generating a sequence of tokens (-n)
# Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (-pg)

cd build/bin/
./llama-bench -m "$MODEL_FILE" -t 8,16,32,40,48,64,80 -p 512 -n 128 -o csv > "${EVAL_PROJECT_DIR}/evaluation/reproduce_issue_${SLURM_JOB_ID}.csv"

echo "Benchmark complete. Results saved to: ${EVAL_PROJECT_DIR}/thread-bug_${SLURM_JOB_ID}.csv"