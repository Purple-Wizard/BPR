#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 {compress|decompress} [args]"
    exit 1
fi

COMMAND=$1
shift

if [ "$COMMAND" = 'compress' ]; then
    exec ./compress.sh "$@"
elif [ "$COMMAND" = 'decompress' ]; then
    exec ./decompress.sh "$@"
else
    echo "Invalid command: $COMMAND"
    echo "Usage: $0 {compress|decompress} [args]"
    exit 1
fi
