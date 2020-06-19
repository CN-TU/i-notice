# iNotice
Incremental Convolutional Classifier for Network Packets

# Description

iNotice offers a new approach of detecting anomalies in network traffic and specifically in mixed input sequences without traditional flow aggregation.
The framework is based on 1D Convolutional Neural Networks and offers the possibility to learn new attacks incrementaly i.e., each time a new attack discovered, its patterns can be added to the network.

We refere the interested reader to the research paper (pending).

# Usage

The repository contains the code for reproducing the experiments in the iNotice paper.
The following componenents are included:
* Ground classifier uder `ground_model`
* CAE used to train the feature extractor under `auto_encoder`
* Incremental learning implementation uder `incremental learning`
* Data used under `data`

## Tree

├── auto_encoder
│   ├── learn.py
│   └── README.md
├── data
│   └── README.md
├── explanability
│   └── README.md
├── figs
│   └── framework.pdf
├── ground_model
│   ├── learn.py
│   └── README.md
├── incremental_learning
│   ├── learn_EWC.py
│   ├── learn_incremental_ae.py
│   ├── learn_incremental.py
│   └── README.md
├── LICENSE
└── README.md

# Framework
The framework is shown in the Figure bellow.


# Paper


# Contact
Fares Meghdouri
fares.meghdouri@tuwien.ac.at