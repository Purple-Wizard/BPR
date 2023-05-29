#!/usr/bin/env sh

ENCODER_MODEL="models/encoder.h5"
DATA_TO_COMPRESS="test/gmaps_test"
NAME_IMAGES="gmaps"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --encoder_model) ENCODER_MODEL="$2"; shift 2;;
        --data_to_compress) DATA_TO_COMPRESS="$2"; shift 2;;
        --name_images) NAME_IMAGES="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

python3 compress.py --data_to_compress "$DATA_TO_COMPRESS" --name_images "$NAME_IMAGES" --encoder_model "$ENCODER_MODEL"
