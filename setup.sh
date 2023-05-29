#!/usr/bin/env sh

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if printf '%s\n%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V -C; then
    echo "Python version is $PYTHON_VERSION, which is sufficient."
    
    # Check operating system
    OS=$(uname)
    if [ "$OS" = "Darwin" ]; then
        echo "This is a macOS system."
        REQUIREMENTS_FILE="requirements/mac_req.txt"
    elif [ "$OS" = "Linux" ]; then
        echo "This is a Linux system."
        REQUIREMENTS_FILE="requirements/linux_req.txt"
    else
        echo "Unknown operating system. Exiting."
        exit 1
    fi
    
    echo "Installing requirements from $REQUIREMENTS_FILE..."
    python3 -m pip install -r "$REQUIREMENTS_FILE"
else
    echo "Python version is $PYTHON_VERSION, which is not sufficient."
    echo "Please upgrade to at least Python $REQUIRED_VERSION."
fi