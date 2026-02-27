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
#SBATCH --gres=gpu:volta:1
#SBATCH --time=00:45:00

# Load modules
module load GCCcore/14.3.0       
module load git/2.50.1

# Setup environment
BASE_DIR="/projects/comp468/aj162/"
mkdir -p $BASE_DIR/cache/llama.cpp
mkdir -p "${BASE_DIR}/evaluation"

PROJECT_DIR="${BASE_DIR}/src/llama.cpp"


# Environment variables
export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# Configure the build
cd $PROJECT_DIR
cmake -B build

# Compile using all 8 CPU cores
cmake --build build --config Release -j 8

# Obtaining and quantizing the model
cd build/bin/


# Start bench mark
./llama-bench -n 0 -p 1024 -b 128,256,512,1024 -o csv > "${BASE_DIR}/evaluation/results_${SLURM_JOB_ID}.csv"