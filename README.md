# Exploring Feature-based Data Augmentations addressing Domain Generalization

## Objective and Scope

This project is part of the course [xAI-Proj-M](https://univis.uni-bamberg.de/prg?search=lectures&department=070106&id=22594511) offered during summer semester 2025 offered by the [chair of explainable machine learning](https://www.uni-bamberg.de/en/ai/chair-of-explainable-machine-learning/) at the [university of Bamberg](https://www.uni-bamberg.de/en/). The goal of the project is to understand and tackle the problem of domain generalization with deep neural networks. For this purpose, the first step is to evaluate ResNet-18 and ResNet-50 on the PACS dataset. Building on this, feature-based data augmentations, such as [MixStyle](https://doi.org/10.1007/s11263-023-01913-8) are explored to potentially offer a solution.

## Setup

You first need to download [Anaconda or Miniconda](https://www.anaconda.com/download/success). Find the necessary dependencies in the [environment file](./environment.yml). The environment can be setup with `conda env create -f environment.yml` and be activated with `conda activate xai_proj_t2`. To be able to run the code, you also need to install [PyTorch](https://pytorch.org/) in a configuration suitable for your system into that environment.

## Contributors

-   Hannes Weber
-   Maximilian Rittler
-   [Johannes Schuster](mailto:johannes-christian.schuster@stud.uni-bamberg.de)
