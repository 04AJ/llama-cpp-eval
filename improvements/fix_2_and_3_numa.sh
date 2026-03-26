#!/bin/bash
#SBATCH --job-name=fix2-numa-bench
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --ntasks=1
#SBATCH --output=logs/fix2_and_3_%j.log
#SBATCH --error=logs/fix2_and_3_%j.err
#SBATCH --cpus-per-task=80
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Load modules
module load CUDA
module load GCCcore/14.3.0
module load git/2.50.1

# Setup environment
source "$HOME/llama-cpp-eval/build/config.sh"
mkdir -p "$BASE_DIR/cache/llama.cpp"

export LLAMA_CACHE="$BASE_DIR/cache/llama.cpp"
export HF_HOME="$BASE_DIR/cache/llama.cpp"

# ---------------------------------------------------------------------------
# Model selection — set MODEL to a name or number from build/models.sh
#   1  gemma-1b   ~0.6 GB
#   2  llama-1b   ~0.7 GB
#   3  llama-3b   ~3.3 GB
#   4  llama-8b   ~8.5 GB  (recommended: largest model = most memory pressure)
# ---------------------------------------------------------------------------
MODEL="llama-8b"
source "$HOME/llama-cpp-eval/build/models.sh"

# No --numa flag: use llama.cpp's default (disabled), so all pages land on node 0
# via first-touch. The mbind fix then redistributes them — that's the improvement we measure.

ORIG_BENCH="${ARCHIVE_DIR}/build/bin/llama-bench"
FIXED_BENCH="${PROJECT_DIR}/build/bin/llama-bench"

OUT_DIR="${EVAL_PROJECT_DIR}/evaluation/numa_compare_${MODEL}_three_way"
mkdir -p "$OUT_DIR"
ORIG_CSV="${OUT_DIR}/numa_original_${SLURM_JOB_ID}.csv"
DISTRIBUTE_CSV="${OUT_DIR}/numa_distribute_${SLURM_JOB_ID}.csv"
FIXED_CSV="${OUT_DIR}/numa_fixed_${SLURM_JOB_ID}.csv"
COMBINED_CSV="${OUT_DIR}/numa_compare_${SLURM_JOB_ID}.csv"

# Sweep across the socket boundary (Xeon Gold 6230: 20 cores/socket, 2 sockets = 40 physical).
# The cliff at 32→40 is what issue #19110 / Fix 2 targets — we need counts on both
# sides of 40 to see whether the fix flattens the cross-socket performance drop.
THREADS="8,16,32,40,48,64,80"

# ---------------------------------------------------------------------------
# STEP 1 -- Build both binaries
# ---------------------------------------------------------------------------
# echo ""
echo "[1/5] Building ORIGINAL binary (no NUMA row partition) ..."
cd "$ARCHIVE_DIR"
cmake -B build > /dev/null 2>&1
cmake --build build --config Release -j 8 2>&1 | tail -3

echo ""
echo "[2/5] Building FIXED binary (NUMA-aware ir0 partition) ..."
cd "$PROJECT_DIR"
cmake -B build > /dev/null 2>&1
cmake --build build --config Release -j 8 2>&1 | tail -3

# ---------------------------------------------------------------------------
# STEP 2 -- Download model if needed
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Checking model ..."
if [ ! -f "$MODEL_FILE" ]; then
    echo "      Downloading model (~8.5 GB) ..."
    wget -q --show-progress -c "$MODEL_URL" -O "$MODEL_FILE"
else
    echo "      Model already cached: $MODEL_FILE"
fi

# ---------------------------------------------------------------------------
# STEP 3 -- Run benchmarks (three-way comparison)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Running llama-bench on ORIGINAL binary  (no --numa, -t $THREADS) ..."
unset LLAMA_NUMA_BIND_ROWS
"$ORIG_BENCH" -m "$MODEL_FILE" \
    -t "$THREADS" \
    -p 512 -n 128 -o csv \
    > "$ORIG_CSV"

echo "      Running llama-bench on ORIGINAL binary  (--numa distribute, -t $THREADS) ..."
unset LLAMA_NUMA_BIND_ROWS
"$ORIG_BENCH" -m "$MODEL_FILE" \
    --numa distribute \
    -t "$THREADS" \
    -p 512 -n 128 -o csv \
    > "$DISTRIBUTE_CSV"

echo "      Running llama-bench on FIXED binary  (no --numa, LLAMA_NUMA_BIND_ROWS=1, -t $THREADS) ..."
export LLAMA_NUMA_BIND_ROWS=1
"$FIXED_BENCH" -m "$MODEL_FILE" \
    -t "$THREADS" \
    -p 512 -n 128 -o csv \
    > "$FIXED_CSV"

# ---------------------------------------------------------------------------
# STEP 4 -- Merge into a single CSV with a leading "binary" column
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Merging results into $COMBINED_CSV ..."

head -1 "$ORIG_CSV"  | awk '{print "binary," $0}'         >  "$COMBINED_CSV"
tail -n +2 "$ORIG_CSV"        | awk '{print "original," $0}'   >> "$COMBINED_CSV"
tail -n +2 "$DISTRIBUTE_CSV"  | awk '{print "distribute," $0}' >> "$COMBINED_CSV"
tail -n +2 "$FIXED_CSV"       | awk '{print "fixed," $0}'      >> "$COMBINED_CSV"

echo ""
echo "-------------------------------------------------------------------"
echo "  Results written to: $COMBINED_CSV"
echo ""
echo "  Row counts:"
echo "    original rows    : $(tail -n +2 "$ORIG_CSV"       | wc -l)"
echo "    distribute rows  : $(tail -n +2 "$DISTRIBUTE_CSV" | wc -l)"
echo "    fixed rows       : $(tail -n +2 "$FIXED_CSV"      | wc -l)"
echo "-------------------------------------------------------------------"
echo ""
echo "  Preview (binary / test / avg tokens per second):"
awk -F',' 'NR==1 {
        for (i=1; i<=NF; i++)
            if ($i=="binary" || $i=="test" || $i=="avg_ts" || $i=="t_pp" || $i=="t_tg")
                col[i]=$i
        next
    }
    {
        out=""
        for (i=1; i<=NF; i++)
            if (i in col) out = (out=="" ? $i : out "," $i)
        print out
    }' "$COMBINED_CSV" | column -t -s','
echo "-------------------------------------------------------------------"
