#!/bin/bash
#SBATCH --job-name=build
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --ntasks=1 
#SBATCH --output=logs/build_%j.log
#SBATCH --error=logs/build_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:45:00

# Load modules
module load GCCcore/14.3.0       
module load git/2.50.1

# Setup environment
source "$HOME/llama-cpp-eval/build/config.sh"
mkdir -p $BASE_DIR/cache/llama.cpp

# Environment variables
export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# Identify Connection Info
NODE_HOSTNAME=$(hostname)
MY_USER=$(whoami)
PORT=8080

# Configure the build
cd $PROJECT_DIR
cmake -B build

# Compile using all 8 CPU cores
cmake --build build --config Release -j 8

# Obtaining and quantizing the model
cd build/bin/

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

# Start the server
# Will download quantized model
./llama-server -hf ggml-org/gemma-3-1b-it-GGUF --host 0.0.0.0 --port 8080 -ngl 0 -t 8 --log-disable