#!/bin/sh
# Rebuild my_torch_network.nn from scratch, exactly as it was trained.
#
# Run from the root of the repository:  ./scripts/build_network.sh
# The network is trained on the dataset given with the subject, dataset.zip, which must be
# at the root of the repository: it is not pushed, since everyone gets it with the subject.
# The settings are chosen on the validation set (docs/reseau_final.md). The final network is
# then trained with them on every position of the dataset, training, validation and test parts
# together: TRAIN_ON=train trains on the training part only, to measure the recipe.
# SEED, CONFIG, ARCHIVE, OUTPUT and TRAIN_ON can be changed through the environment.

set -eu

SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/my_torch_network.conf}"
ARCHIVE="${ARCHIVE:-dataset.zip}"
OUTPUT="${OUTPUT:-my_torch_network.nn}"
TRAIN_ON="${TRAIN_ON:-all}"
STEM="datasets/official"
BLANK="$(basename "${CONFIG%.*}")_1.nn"

if [ ! -f "$ARCHIVE" ]; then
    echo "build_network: $ARCHIVE not found: put the dataset given with the subject there" >&2
    exit 1
fi

make --no-print-directory

python3 -m scripts.prepare_official_dataset --archive "$ARCHIVE" --output "$STEM.txt"
python3 -m scripts.split_dataset --input "$STEM.txt" --seed "$SEED"

case "$TRAIN_ON" in
    all) cat "${STEM}_train.txt" "${STEM}_validation.txt" "${STEM}_test.txt" > "${STEM}_all.txt" ;;
    train) ;;
    *) echo "build_network: TRAIN_ON must be all or train, got '$TRAIN_ON'" >&2; exit 1 ;;
esac

./my_torch_generator "$CONFIG" 1
./my_torch_analyzer --train --save "$OUTPUT" "$BLANK" "${STEM}_${TRAIN_ON}.txt"
rm -f "$BLANK"

if [ "$TRAIN_ON" = all ]; then
    echo
    echo "The network has seen the validation and test sets: the scores below are not those of"
    echo "unseen positions. Run with TRAIN_ON=train to measure the recipe on them."
fi
echo
echo "=== Validation set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_validation.txt"
echo
echo "=== Test set ==="
python3 -m scripts.evaluate --network "$OUTPUT" --chessboards "${STEM}_test.txt"
