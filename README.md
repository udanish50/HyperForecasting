# HyperForecasting: Leveraging Hypernetworks and Learnable Kernels for Load Forecasting Across Diverse Consumer Types

## Overview
This repository contains the implementation of "HyperForecasting," a novel approach to load forecasting that utilizes hypernetworks to dynamically adapt to complex consumption patterns across different types of energy consumers. The method incorporates advanced machine learning techniques to address the challenges of traditional forecasting models, providing a robust solution for diverse scenarios including residences, detached homes, and townhouses, with and without electric vehicles.

## Publication
- **Authors**: Muhammad Umair Danish, Katarina Grolinger
- **Affiliation**: Department of Electrical and Computer Engineering, The University of Western Ontario, London, ON, Canada
- **Published in**: IEEE Transactions on Power Delivery

## Abstract
Load forecasting plays a crucial role in the management and planning of energy systems, impacting operational efficiency, cost-effectiveness, grid stability, and environmental sustainability. Current deep learning methods like LSTMs and transformers show promise but often fail in scenarios involving rapid or complex variations in energy usage, or when applied uniformly across diverse consumer types. This paper introduces "HyperForecasting," a strategy that enhances forecasting accuracy by employing hypernetworks that generate parameters for a primary LSTM network, optimized through a learnable kernel that integrates polynomial and radial basis function kernels.

## Key Contributions
1. **Innovative Approach**: Introduction of a hypernetwork-based architecture that adapts to complex energy patterns across diverse consumer profiles.
2. **Kernel Integration**: Use of a learnable adaptive kernel combining polynomial and radial basis function kernels to enhance model performance.
3. **Comprehensive Evaluation**: Demonstrates superior performance over existing models across multiple consumer types including student residences, detached homes, and townhouses with variations such as electric vehicle charging.

## Repository Structure
- `src/` - Source code including the implementation of the HyperForecasting model and utility scripts.
- `data/` - Sample datasets used for training and evaluating the model.
- `models/` - Trained models and configuration files.
- `notebooks/` - Jupyter notebooks illustrating how to train and test the models.
- `results/` - Charts and tables presenting the evaluation results.

## Usage
Details on how to setup, train, and evaluate the model are provided in the Jupyter notebooks located in the `notebooks/` directory. To get started, clone the repository and follow the instructions in `notebooks/Example_Usage.ipynb`.

## Citing This Work
If you use this method in your research, please cite the following paper:

```bibtex
@article{danish2023hyperforecasting,
  title={HyperForecasting: Leveraging Hypernetworks and Learnable Kernels for Load Forecasting Across Diverse Consumer Types},
  author={Muhammad Umair Danish and Katarina Grolinger},
  journal={IEEE Transactions on Power Delivery},
  year={2023},
  publisher={IEEE}
}

