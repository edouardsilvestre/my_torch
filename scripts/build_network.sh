#!/bin/sh
# Rebuild my_torch_network.nn from scratch, exactly as it was trained.
#
# Run from the root of the repository:  ./scripts/build_network.sh
# The network is trained on the dataset given with the subject, dataset.zip, which must be
# at the root of the repository: it is not pushed, since everyone gets it with the subject.
# SEED, CONFIG, ARCHIVE and OUTPUT can be changed through the environment.

set -eu

SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/my_torch_network.conf}"
ARCHIVE="${ARCHIVE:-dataset.zip}"
OUTPUT="${OUTPUT:-my_torch_network.nn}"
STEM="datasets/official"
BLANK="$(basename "${CONFIG%.*}")_1.nn"

if [ ! -f "$ARCHIVE" ]; then
    echo "build_network: $ARCHIVE not found: put the dataset given with the subject there" >&2
    exit 1
fi

make --no-print-directory

python3 -m scripts.prepare_official_dataset --archive "$ARCHIVE" --output "$STEM.txt"
python3 -m scripts.split_dataset --input "$STEM.txt" --seed "$SEED"

./my_torch_generator "$CONFIG" 1
./my_torch_analyzer --train --save "$OUTPUT" "$BLANK" "${STEM}_train.txt"
rm -f "$BLANK"

echo
echo "=== Validation set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_validation.txt"
echo
echo "=== Test set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_test.txt"
