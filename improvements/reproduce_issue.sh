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

CSV="${EVAL_PROJECT_DIR}/evaluation/reproduce_issue/reproduce_issue_${SLURM_JOB_ID}.csv"
MD="${EVAL_PROJECT_DIR}/evaluation/reproduce_issue/reproduce_issue_${SLURM_JOB_ID}.md"
mkdir -p "${EVAL_PROJECT_DIR}/evaluation/reproduce_issue"

cd build/bin/
./llama-bench -m "$MODEL_FILE" -t 8,16,32,40,48,64,80 -p 512 -n 128 -o csv > "$CSV"

# Post-process CSV → markdown (model_type, n_threads, n_prompt, n_gen, avg_ts, stddev_ts)
awk -F',' '
NR==1 {
    for (i=1; i<=NF; i++) {
        gsub(/"/, "", $i)
        if ($i=="model_type")  c_model=i
        if ($i=="n_threads")   c_threads=i
        if ($i=="n_prompt")    c_pp=i
        if ($i=="n_gen")       c_tg=i
        if ($i=="avg_ts")      c_avg=i
        if ($i=="stddev_ts")   c_std=i
    }
    printf "| model | threads | n_prompt | n_gen | avg tok/s | stddev tok/s |\n"
    printf "|-------|---------|----------|-------|-----------|--------------|\n"
    next
}
{
    for (i=1; i<=NF; i++) gsub(/"/, "", $i)
    printf "| %s | %s | %s | %s | %s | %s |\n",
        $c_model, $c_threads, $c_pp, $c_tg, $c_avg, $c_std
}' "$CSV" > "$MD"

echo "Benchmark complete."
echo "  CSV: $CSV"
echo "  MD:  $MD"