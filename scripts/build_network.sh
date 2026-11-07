#!/bin/sh
# Rebuild my_torch_network.nn from scratch, exactly as it was trained.
#
# Run from the root of the repository:  ./scripts/build_network.sh
# The dataset is written to datasets/ (ignored by git) and reused if it is already there.
# SIZE, SEED, CONFIG and OUTPUT can be changed through the environment, to try the
# pipeline on a smaller dataset.

set -eu

SIZE="${SIZE:-100000}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/my_torch_network.conf}"
OUTPUT="${OUTPUT:-my_torch_network.nn}"
DATASET="datasets/all_${SIZE}_${SEED}.txt"
STEM="datasets/all_${SIZE}_${SEED}"
BLANK="$(basename "${CONFIG%.*}")_1.nn"

make --no-print-directory

if [ -f "$DATASET" ]; then
    echo "Reusing $DATASET"
else
    python3 -m scripts.generate_dataset --size "$SIZE" --seed "$SEED" --output "$DATASET"
fi
python3 -m scripts.split_dataset --input "$DATASET" --seed "$SEED"

./my_torch_generator "$CONFIG" 1
./my_torch_analyzer --train --save "$OUTPUT" "$BLANK" "${STEM}_train.txt"
rm -f "$BLANK"

echo
echo "=== Validation set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_validation.txt"
echo
echo "=== Test set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_test.txt"
