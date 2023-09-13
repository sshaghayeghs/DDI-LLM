# `DDI-LLM:` Drug-Drug Interaction Prediction: Experimenting With Large Language-Based Drug Information Embedding For Multi-View Representation Learning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qoXDJjS__UNPf93rh6i-HtLQlo0sFXoC?usp=sharing)

`Drug-Drug Interactions (DDIs)` can arise when multiple drugs are used to treat complex or concurrent medical conditions, potentially leading to alterations in the way these drugs work. Consequently, predicting DDIs has become a crucial endeavor within the field of medical machine learning, addressing a critical aspect of healthcare.

This paper explores the application of `large language-based` embeddings, including `BERT`, `GPT`, `LLaMA`, and `LLaMA2`, within the context of `Graph Convolutional Networks (GCN)` to enhance `DDI prediction`.

We start by harnessing these advanced language models to generate embeddings for drug chemical structures and drug descriptions, providing a more comprehensive representation of drug characteristics. These embeddings are subsequently integrated into a DDI network, with GCN employed for `link prediction`. We utilize BERT, GPT, and LLaMA embeddings to improve the accuracy and effectiveness of predicting drug interactions within this network.

Our experiments reveal that the utilization of language-based drug embeddings in combination with DDI structure embeddings can yield accuracy levels `comparable` to state-of-the-art methods in DDI prediction.

<p align="center">
  <img width="500" height="800" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/DDI_LM.png">
</p>

# 1. Requirments
You need to have `Python >= 3.8` and install the following main packages:
```
!pip install torch_geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install dgl
!pip install rdkit
!pip install deepchem
!pip install gensim
!pip install git+https://github.com/samoturk/mol2vec
!pip install transformers
!pip install openai
```
