#!/bin/bash
#SBATCH --job-name=build
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --ntasks=1 
#SBATCH --output=logs/build_%j.log
#SBATCH --error=logs/build_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=00:45:00

# Load modules
module load GCCcore/14.3.0       
module load git/2.50.1

# Setup environment
source "$HOME/llama-cpp-eval/build/config.sh"
mkdir -p "$BASE_DIR/cache/llama.cpp"

export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# ---------------------------------------------------------------------------
# Model selection — set MODEL to a name or number from build/models.sh
#   1  gemma-1b      ~0.6 GB  (VLM)
#   2  llama-1b      ~0.7 GB
#   3  llama-3b      ~3.3 GB
#   4  llama-8b      ~8.5 GB
#   5  deepseek-8b   ~8.5 GB
#   6  qwen3-vl-4b   ~2.5 GB  (VLM)
# ---------------------------------------------------------------------------
# MODEL="llama-1b"
MODEL="qwen3-vl-4b"
source "$HOME/llama-cpp-eval/build/models.sh"

# Identify Connection Info
NODE_HOSTNAME=$(hostname)
MY_USER=$(whoami)
PORT=8080

# Build archive (unpatched) binary — wipe stale cache first to avoid
# inheriting CMakeCache.txt paths from the project build directory
cd "$ARCHIVE_DIR"
# rm -rf build
cmake -B build -DGGML_CUDA=OFF > /dev/null 2>&1
cmake --build build --config Release -j 8 2>&1 | tail -3

echo "==============================================================="
echo "  LLAMA CPU SERVER IS STARTING!"
echo "==============================================================="
echo "  STEP 1: Open a NEW terminal on your local machine."
echo "  STEP 2: Run this exact command to create the tunnel:"
echo ""
echo "  ssh -L ${PORT}:${NODE_HOSTNAME}:${PORT} ${MY_USER}@nots.crc.rice.edu"
echo ""
echo "  STEP 3: Open your browser and go to: http://localhost:${PORT}"
echo "==============================================================="

MMPROJ_ARG=""
[ -n "$MMPROJ_FILE" ] && MMPROJ_ARG="--mmproj $MMPROJ_FILE"

"$ARCHIVE_DIR/build/bin/llama-server" \
    -m "$MODEL_FILE" \
    $MMPROJ_ARG \
    --host 0.0.0.0 --port "$PORT" \
    -ngl 0 -t 20 --log-disable