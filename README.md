# Leveraging Mixture of Experts for Improved Speech Deepfake Detection

Welcome to the GitHub repository for the paper *"Leveraging Mixture of Experts for Improved Speech Deepfake Detection."*

## Overview

This repository contains the code and resources for implementing the Mixture of Experts (MoE) architecture to improve the performance of speech deepfake detection. The proposed approach utilizes the Mixture of Experts framework to better generalize across various unseen datasets and effectively adapt to the challenges posed by evolving deepfake techniques.

## Key Features:
- **Mixture of Experts Architecture**: A modular approach to handle input variability and improve generalization.
- **Gating Mechanism**: An efficient, lightweight dynamic expert selection for optimizing detection performance.
- **Scalable Updates**: The modular structure allows easy adaptation to new data and evolving deepfake detection methods.
- **Reproducible Pipelines**: Training and Evaluation scripts for both LCNN and MoE models are provided.


## Repository Structure

```
├── README.md               # Project documentation
├── config/                 # Configuration files for training and evaluation
├── env.yaml                # Conda environment file with dependencies
├── src/                    # Source code modules and utility scripts
├── train_lcnn_model.py     # Script for training the LCNN expert model
├── train_moe_model.py      # Script for training the MoE model
├── eval_lcnn_model.py      # Evaluation script for the LCNN model
├── eval_moe_model.py       # Evaluation script for the MoE model
├── lcnn_model_2.py         # LCNN model definition (standalone or alternative version)
```

## Setup

1. Clone the repository and create the environment:

```
git clone https://github.com/polimi-ispl/moe_speech_deepfake.git
cd moe_speech_deepfake
conda env create -f env.yaml
conda activate moe_env
```

2. Run training or evaluation scripts:

```
python train_lcnn_model.py      # Train LCNN model
python train_moe_model.py       # Train MoE model
python eval_lcnn_model.py       # Evaluate LCNN
python eval_moe_model.py        # Evaluate MoE
```

## Citation

If you use this code in your research, please cite the following paper:

**Negroni, V., Salvi, D., Mezza, A. I., Bestagini, P., & Tubaro, S. (2025).** *Leveraging Mixture of Experts for Improved Speech Deepfake Detection.* In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

```
@inproceedings{Negroni2025,
  title={Leveraging Mixture of Experts for Improved Speech Deepfake Detection},
  author={Viola Negroni, Davide Salvi, Alessandro Ilic Mezza, Paolo Bestagini, Stefano Tubaro},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025}
}
```

## Contact

For any inquiries or collaboration requests, please contact:

- **Viola Negroni**: viola.negroni@polimi.it
- **Davide Salvi**: davide.salvi@polimi.it
- **Alessandro Ilic Mezza**: alessandroilic.mezza@polimi.it
- **Paolo Bestagini**: paolo.bestagini@polimi.it
- **Stefano Tubaro**: stefano.tubaro@polimi.it
