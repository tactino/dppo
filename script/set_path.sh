#!/bin/bash

##################### Paths #####################

# Set default paths
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"

# Prompt the user for input, allowing overrides
read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
DPPO_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], leave empty to use default: " LOG_DIR
DPPO_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

# Export to current session
export DPPO_DATA_DIR="$DPPO_DATA_DIR"
export DPPO_LOG_DIR="$DPPO_LOG_DIR"

# Confirm the paths with the user
echo "Data directory set to: $DPPO_DATA_DIR"
echo "Log directory set to: $DPPO_LOG_DIR"

# Detect shell config file (Zsh or Bash)
SHELL_CONFIG="$HOME/.zshrc"
if [[ "$SHELL" == *"bash"* ]]; then
  SHELL_CONFIG="$HOME/.bashrc"
fi

# Append environment variables to shell config
echo "export DPPO_DATA_DIR=\"$DPPO_DATA_DIR\"" >> "$SHELL_CONFIG"
echo "export DPPO_LOG_DIR=\"$DPPO_LOG_DIR\"" >> "$SHELL_CONFIG"

echo "Environment variables DPPO_DATA_DIR and DPPO_LOG_DIR added to $SHELL_CONFIG and applied to the current session."

##################### SwanLab #####################

# Prompt the user for SwanLab entity
read -p "Enter your SwanLab entity (username or team name), leave empty to skip: " ENTITY

if [ -n "$ENTITY" ]; then
  export DPPO_SWANLAB_ENTITY="$ENTITY"
  echo "SwanLab entity set to: $DPPO_SWANLAB_ENTITY"
  echo "export DPPO_SWANLAB_ENTITY=\"$ENTITY\"" >> "$SHELL_CONFIG"
  echo "Environment variable DPPO_SWANLAB_ENTITY added to $SHELL_CONFIG and applied to the current session."
else
  echo "No SwanLab entity provided. Please set swanlab=null when running scripts to disable SwanLab logging and avoid errors."
fi
