# LipoNet: Graph-Based Lipophilicity Estimation for Drug Discovery

## Overview

LipoNet is a deep learning project for predicting the lipophilicity (logD) of molecules using advanced graph neural network (GNN) architectures.
Lipophilicity is a key property in drug discovery, affecting ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity). 
This project benchmarks state-of-the-art GNN models on the MoleculeNet lipo dataset, with rigorous scaffold split validation to ensure robust generalization to novel chemical scaffolds.


## Introduction

Accurate prediction of molecular properties is essential in cheminformatics and drug discovery. Lipophilicity (logD) is particularly important for understanding pharmacokinetic behavior. Traditional methods rely on expert-designed descriptors, but GNNs can learn directly from molecular graphs, capturing complex relationships. LipoNet systematically benchmarks several GNN architectures, using scaffold split validation to assess true generalization performance in real-world scenarios[1].

## Features

- **Multiple GNN Architectures:** Benchmarks models such as GIN, GraphSAGE, and hybrid approaches.
- **Realistic Validation:** Uses Bemis-Murcko scaffold split for robust, chemistry-aware model evaluation.
- **Comprehensive Analysis:** Reports RMSE, MAE, and R², with grouped boxplots and distribution plots.
- **Reproducibility:** Includes code and outputs for transparency and further research.
