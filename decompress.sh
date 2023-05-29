#!/usr/bin/env sh

# Set the default values for the flags
DECODER_MODEL="models/decoder.h5"
PATH_TO_COMPRESSED="compressed/gmaps.npy"
IMAGES_FOLDER_NAME="test/gmaps"
SHOW_IMAGES=0
SAVE_LOCAL="True"

# Parse the command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --decoder_model) DECODER_MODEL="$2"; shift 2;;
        --path_to_compressed) PATH_TO_COMPRESSED="$2"; shift 2;;
        --images_folder_name) IMAGES_FOLDER_NAME="$2"; shift 2;;
        --show_images) SHOW_IMAGES="$2"; shift 2;;
        --save_local) SAVE_LOCAL="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

# Run the Python script with the given arguments
python3 decompress.py --decoder_model "$DECODER_MODEL" --path_to_compressed "$PATH_TO_COMPRESSED" --show_images "$SHOW_IMAGES" --save_local "$SAVE_LOCAL" --images_folder_name "$IMAGES_FOLDER_NAME"
