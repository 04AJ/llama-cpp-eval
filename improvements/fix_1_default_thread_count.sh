#!/bin/bash
#SBATCH --job-name=fix1-thread-compare
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --ntasks=1
#SBATCH --output=logs/fix1_%j.log
#SBATCH --error=logs/fix1_%j.err
#SBATCH --cpus-per-task=80
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# Load modules
module load GCCcore/14.3.0
module load git/2.50.1

# Setup environment
source "$HOME/llama-cpp-eval/build/config.sh"
mkdir -p "$BASE_DIR/cache/llama.cpp"

export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"


# Model selection — set MODEL to a name or number from build/models.sh
#   1  gemma-1b   ~0.6 GB
#   2  llama-1b   ~0.7 GB
#   3  llama-3b   ~3.3 GB
#   4  llama-8b   ~8.5 GB
MODEL="llama-1b"
source "$HOME/llama-cpp-eval/build/models.sh"

ORIG_BIN="${ARCHIVE_DIR}/build/bin/llama-server"
FIXED_BIN="${PROJECT_DIR}/build/bin/llama-server"

# STEP 1 -- Build the original (archive) binary  [hardware_concurrency]
echo ""
echo "[1/4] Building ORIGINAL binary (hardware_concurrency) ..."
cd "$ARCHIVE_DIR"
cmake -B build > /dev/null 2>&1
cmake --build build --config Release -j 8 --target llama-server 2>&1
echo "      Built: $ORIG_BIN"

# STEP 2 -- Build the fixed binary  [cpu_get_num_physical_cores]
echo ""
echo "[2/4] Building FIXED binary (cpu_get_num_physical_cores) ..."
rm -rf "$PROJECT_DIR/build"
cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build" > /dev/null 2>&1
cmake --build "$PROJECT_DIR/build" --config Release -j 8 --target llama-server 2>&1
echo "      Built: $FIXED_BIN"

# STEP 3 -- Probe each binary with --threads -1
echo ""
echo "[3/4] Running both binaries with --threads -1 ..."
echo "  [debug] lscpu topology on this node:"
lscpu | grep -E "Socket|Core|Thread|CPU\(s\)" | sed 's/^/    /'

ORIG_OUTPUT=$(timeout 10 "$ORIG_BIN" \
    -m "$MODEL_FILE" --threads -1 2>&1 \
    | grep -iE "n_threads|threads =" || true)

PHYS_CORES=$(lscpu | awk '/^Core\(s\) per socket:/{cores=$NF} /^Socket\(s\):/{sockets=$NF} END{print cores*sockets}')
echo "  [debug] PHYS_CORES='$PHYS_CORES'"
echo "  [debug] env var visible to subshell: $(LLAMA_PHYSICAL_CORES="$PHYS_CORES" printenv LLAMA_PHYSICAL_CORES)"
echo "  [debug] fixed binary with explicit --threads $PHYS_CORES:"
timeout 10 "$FIXED_BIN" -m "$MODEL_FILE" --threads "$PHYS_CORES" 2>&1 \
    | grep -iE "n_threads|threads =" | sed 's/^/    /'
echo "  [debug] fixed binary with --threads -1 + env var:"
FIXED_OUTPUT=$(LLAMA_PHYSICAL_CORES="$PHYS_CORES" timeout 10 "$FIXED_BIN" \
    -m "$MODEL_FILE" --threads -1 2>&1 \
    | grep -iE "n_threads|threads =" || true)

ORIG_N=$(echo "$ORIG_OUTPUT"  | grep -oP "n_threads\s*=\s*\K[0-9]+" | head -1)
FIXED_N=$(echo "$FIXED_OUTPUT" | grep -oP "n_threads\s*=\s*\K[0-9]+" | head -1)

LOGICAL_CORES=$(nproc)

# STEP 4 -- Print comparison
echo ""
echo "[4/4] Results: default n_threads with --threads -1"
echo "-------------------------------------------------------------------"
printf "  %-28s  %-12s  %-12s\n" "Metric"                  "ORIGINAL"    "FIXED"
echo "-------------------------------------------------------------------"
printf "  %-28s  %-12s  %-12s\n" "Default function"        "hardware_concurrency()" "cpu_get_num_physical_cores()"
printf "  %-28s  %-12s  %-12s\n" "n_threads chosen"        "${ORIG_N:-N/A}"  "${FIXED_N:-N/A}"
echo "-------------------------------------------------------------------"
printf "  %-28s  %s\n" "Physical cores on node:"  "$PHYS_CORES"
printf "  %-28s  %s\n" "Logical cores (w/ HT):"   "$LOGICAL_CORES"
echo "-------------------------------------------------------------------"

echo ""
if [ -n "$ORIG_N" ] && [ -n "$FIXED_N" ]; then
    if [ "$ORIG_N" -ne "$FIXED_N" ]; then
        echo "  Verdict: bug confirmed and fixed."
        echo "    Original chose $ORIG_N threads (logical cores, inflated by HT)"
        echo "    Fixed    chose $FIXED_N threads (physical cores only)"
        echo "    Excess threads eliminated: $(( ORIG_N - FIXED_N ))"
    else
        echo "  Verdict: both binaries chose the same value ($ORIG_N)."
        echo "           HT may not be active on this node."
    fi
fi

echo ""
echo "  Raw output -- ORIGINAL:"
echo "$ORIG_OUTPUT" | sed 's/^/    /'
echo ""
echo "  Raw output -- FIXED:"
echo "$FIXED_OUTPUT" | sed 's/^/    /'
echo ""
