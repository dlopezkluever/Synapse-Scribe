#!/usr/bin/env bash
# download_data.sh — Download datasets for the BCI-2 Brain-Text Decoder project.
#
# Usage:
#   bash scripts/download_data.sh --dataset willett
#   bash scripts/download_data.sh --dataset openneuro
#   bash scripts/download_data.sh --dataset all
#   bash scripts/download_data.sh --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATASET=""
DATA_DIR="./data"

# ---------------------------------------------------------------------------
# Color helpers (degrade gracefully if no tty)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    GREEN='' YELLOW='' RED='' CYAN='' NC=''
fi

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash scripts/download_data.sh [OPTIONS]

Download datasets for the BCI-2 Brain-Text Decoder project.

Options:
  --dataset DATASET   Dataset to download: willett, openneuro, or all
                      (required)
  --data-dir DIR      Base data directory (default: ./data)
  --help              Show this help message and exit

Datasets:
  willett       Willett et al. handwriting BCI dataset (Dryad)
                DOI: https://doi.org/10.5061/dryad.wh70rxwmv
                NOTE: Requires manual download (browser-based).

  openneuro     OpenNeuro dataset ds003688
                Uses openneuro-py CLI for automated download.

  all           Download/setup all supported datasets.

Examples:
  bash scripts/download_data.sh --dataset willett
  bash scripts/download_data.sh --dataset openneuro --data-dir /mnt/data
  bash scripts/download_data.sh --dataset all
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    error "Missing required --dataset flag."
    echo ""
    usage
    exit 1
fi

# Validate dataset argument
case "$DATASET" in
    willett|openneuro|all)
        ;;
    *)
        error "Invalid dataset: '$DATASET'. Choose from: willett, openneuro, all"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Create base data directory
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"
info "Data directory: $DATA_DIR"

# ---------------------------------------------------------------------------
# Willett dataset (manual download)
# ---------------------------------------------------------------------------
download_willett() {
    info "Setting up Willett handwriting BCI dataset..."

    WILLETT_DIR="$DATA_DIR/willett_handwriting"
    mkdir -p "$WILLETT_DIR"

    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  Willett et al. Handwriting BCI Dataset${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo "  This dataset requires manual download from Dryad."
    echo ""
    echo "  Steps:"
    echo "    1. Visit: https://doi.org/10.5061/dryad.wh70rxwmv"
    echo "    2. Click 'Download Dataset' (you may need to create a free account)"
    echo "    3. Extract the downloaded archive"
    echo "    4. Place the contents into: $WILLETT_DIR/"
    echo ""
    echo "  Expected structure after extraction:"
    echo "    $WILLETT_DIR/"
    echo "      Datasets/"
    echo "        t5.2019.05.08/"
    echo "          singleLetters.mat"
    echo "          sentences.mat"
    echo "          ..."
    echo ""
    echo "  DOI: 10.5061/dryad.wh70rxwmv"
    echo "  Paper: Willett et al., 'High-performance brain-to-text communication"
    echo "         via handwriting', Nature 2021"
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    # Create a README in the directory as a reminder
    cat > "$WILLETT_DIR/DOWNLOAD_INSTRUCTIONS.txt" <<HEREDOC
Willett Handwriting BCI Dataset
===============================

DOI: https://doi.org/10.5061/dryad.wh70rxwmv

This dataset requires manual download:
1. Visit the DOI link above
2. Click "Download Dataset"
3. Extract the archive contents into this directory

Expected structure:
  Datasets/
    t5.2019.05.08/
      singleLetters.mat
      sentences.mat
      ...
HEREDOC

    info "Created download instructions at $WILLETT_DIR/DOWNLOAD_INSTRUCTIONS.txt"
    info "Willett dataset setup complete (manual download required)."
}

# ---------------------------------------------------------------------------
# OpenNeuro dataset (automated download)
# ---------------------------------------------------------------------------
download_openneuro() {
    info "Setting up OpenNeuro dataset ds003688..."

    OPENNEURO_DIR="$DATA_DIR/openneuro"
    mkdir -p "$OPENNEURO_DIR"

    # Check if openneuro-py is installed
    if ! python -c "import openneuro" 2>/dev/null; then
        info "Installing openneuro-py..."
        pip install openneuro-py
        if [[ $? -ne 0 ]]; then
            error "Failed to install openneuro-py. Please install manually: pip install openneuro-py"
            exit 1
        fi
        info "openneuro-py installed successfully."
    else
        info "openneuro-py is already installed."
    fi

    info "Downloading OpenNeuro dataset ds003688..."
    info "This may take a while depending on your connection speed."
    echo ""

    # Download using openneuro-py CLI
    openneuro-py download --dataset ds003688 --target-dir "$OPENNEURO_DIR/ds003688"

    if [[ $? -eq 0 ]]; then
        info "OpenNeuro dataset downloaded successfully to $OPENNEURO_DIR/ds003688/"
    else
        error "Download failed. You can retry with:"
        error "  openneuro-py download --dataset ds003688 --target-dir $OPENNEURO_DIR/ds003688"
        exit 1
    fi

    info "OpenNeuro dataset setup complete."
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
echo ""
info "BCI-2 Data Download Script"
info "=========================="
echo ""

case "$DATASET" in
    willett)
        download_willett
        ;;
    openneuro)
        download_openneuro
        ;;
    all)
        download_willett
        echo ""
        download_openneuro
        ;;
esac

echo ""
info "Done. Data directory: $DATA_DIR"
