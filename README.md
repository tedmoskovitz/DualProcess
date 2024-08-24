# DualProcess
# DualProcess

This repository contains the implementation of the Minimum Description Length Control (MDL-C) agent from the paper *Understanding dual process cognition via the minimum description length principle*. 

## Key Components

- `MDLCAgent`: The main agent class implementing the dual-process architecture.
- `AC_LSTM`: Actor-Critic LSTM networks used for both control and default pathways.
- Various environment implementations including FourRooms, Stroop, TwoStep, and more.

## Features

- Variational Dropout (VDO) for regularization
- Asymmetric learning rates between control and default pathways
- Configurable reaction time budgets and thresholds
- Flexible hyperparameter configuration via `configurator.py`

## Usage

To run an experiment:

1. Configure the desired environment and agent parameters in `configurator.py`
2. Run `python run.py` with the appropriate environment name

## Requirements

- PyTorch
- NumPy
- Wandb (optional, for logging)

