# models.sh — central model registry
# Source this file, then call: load_model <name|number>
# Sets MODEL_FILE and MODEL_URL in the calling script.
#
# Usage examples:
#   MODEL=llama-8b   source "$HOME/llama-cpp-eval/build/models.sh"
#   MODEL=4          source "$HOME/llama-cpp-eval/build/models.sh"
#
# To list all available models:
#   source "$HOME/llama-cpp-eval/build/models.sh" && list_models

# Requires BASE_DIR to be set (done by config.sh)
_CACHE="${BASE_DIR}/cache/llama.cpp"

# Registry: parallel arrays indexed 1..N
# Add a new model by appending one entry to each array.
_MODEL_NAMES=(
    [1]="gemma-1b"
    [2]="llama-1b"
    [3]="llama-3b"
    [4]="llama-8b"
)

_MODEL_FILES=(
    [1]="${_CACHE}/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
    [2]="${_CACHE}/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    [3]="${_CACHE}/bartowski_Llama-3.2-3B-Instruct-GGUF_Llama-3.2-3B-Instruct-Q8_0.gguf"
    [4]="${_CACHE}/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
)

_MODEL_URLS=(
    [1]="https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
    [2]="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    [3]="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
    [4]="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
)

_MODEL_SIZES=(
    [1]="~0.6 GB"
    [2]="~0.7 GB"
    [3]="~3.3 GB"
    [4]="~8.5 GB"
)

list_models() {
    echo ""
    echo "  Available models:"
    echo "  -----------------------------------------------------------"
    printf "  %-4s  %-14s  %-10s  %s\n" "No." "Name" "Size" "Cached"
    echo "  -----------------------------------------------------------"
    for i in "${!_MODEL_NAMES[@]}"; do
        local cached="no"
        [ -f "${_MODEL_FILES[$i]}" ] && cached="yes"
        printf "  %-4s  %-14s  %-10s  %s\n" \
            "$i" "${_MODEL_NAMES[$i]}" "${_MODEL_SIZES[$i]}" "$cached"
    done
    echo "  -----------------------------------------------------------"
    echo ""
}

# load_model <name|number>
# Sets MODEL_FILE, MODEL_URL.  Downloads if not cached.
load_model() {
    local key="$1"
    local idx=""

    # Resolve number directly
    if [[ "$key" =~ ^[0-9]+$ ]]; then
        idx="$key"
    else
        # Resolve name to index
        for i in "${!_MODEL_NAMES[@]}"; do
            if [ "${_MODEL_NAMES[$i]}" = "$key" ]; then
                idx="$i"
                break
            fi
        done
    fi

    if [ -z "$idx" ] || [ -z "${_MODEL_NAMES[$idx]+_}" ]; then
        echo "  ERROR: unknown model '$key'."
        list_models
        exit 1
    fi

    MODEL_FILE="${_MODEL_FILES[$idx]}"
    MODEL_URL="${_MODEL_URLS[$idx]}"

    echo "  Model   : ${_MODEL_NAMES[$idx]} (${_MODEL_SIZES[$idx]})"
    echo "  File    : $MODEL_FILE"

    if [ ! -f "$MODEL_FILE" ]; then
        echo "  Status  : not cached — downloading ..."
        wget -q --show-progress -c "$MODEL_URL" -O "$MODEL_FILE"
    else
        echo "  Status  : cached"
    fi
}

# Auto-load if MODEL is set when this file is sourced
if [ -n "${MODEL:-}" ]; then
    load_model "$MODEL"
fi
