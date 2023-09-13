# `DDI-LLM:`:pill:	 Drug-Drug Interaction Prediction: Experimenting With Large Language-Based Drug Information Embedding For Multi-View Representation Learning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qoXDJjS__UNPf93rh6i-HtLQlo0sFXoC?usp=sharing)

`Drug-Drug Interactions (DDIs)` can arise when multiple drugs are used to treat complex or concurrent medical conditions, potentially leading to alterations in the way these drugs work. Consequently, predicting DDIs has become a crucial endeavor within the field of medical machine learning, addressing a critical aspect of healthcare.

This paper explores the application of `large language-based` embeddings, including `BERT`, `GPT`, `LLaMA`, and `LLaMA2`, within the context of `Graph Convolutional Networks (GCN)` to enhance `DDI prediction`.

We start by harnessing these advanced language models to generate embeddings for drug chemical structures and drug descriptions, providing a more comprehensive representation of drug characteristics. These embeddings are subsequently integrated into a DDI network, with GCN employed for `link prediction`. We utilize BERT, GPT, LLaMA and LLaMA2 embeddings to improve the accuracy and effectiveness of predicting drug interactions within this network.

Our experiments reveal that the utilization of language-based drug embeddings in combination with DDI structure embeddings can yield accuracy levels `comparable` to state-of-the-art methods in DDI prediction.



# 1. Requirements
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
# 2. Pipeline
* 2.1. [**Language-based Drug Information Embedding**](https://github.com/sshaghayeghs/DDI-LLM/tree/main/Get%20Embedding)
* 2.2. [**Multi-View Representation Fusion and Predicting DDI**](https://github.com/sshaghayeghs/DDI-LLM/blob/main/PyG__link_pred_DDI.ipynb)
  
<p align="center">
  <img width="500" height="800" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/DDI_LM.png">
</p>


  
 




# 3. Gold Standard Datasets
  |Dataset| [BioSnap](https://github.com/sshaghayeghs/DDI-LLM/blob/main/Dataset/DDI/BioSNAP_ChCh-Miner_ChCh-Miner_durgbank-chem-chem.tsv)| [DrugBank](https://github.com/sshaghayeghs/DDI-LLM/blob/main/Dataset/DDI/DrugbankDDI.csv)|
  | ------------ | ------| ------- |
  |Number of nodes| 1514|1706|
  |Number of edges|48514|191808|
  |Is undirected| True| True|
  |Average node degree|64.087|224.38|
  |Has isolated nodes|False|False|
  |Has self-loops|False| False|
# 3. Results
**`AUROC BioSnap`**
| Embedding    | 0.01  | 0.001  | 0.0001| 0.0002  | 0.0003 | 1.00E-05 |
| ------------ | ------| ------- | -------| ------- | ------ | -------|
| No Feature   | 0.8594| 0.8721| 0.8717| 0.8702 | 0.8758 | 0.7349 |
| Morgan       | 0.8628| 0.8815| 0.8669 | 0.8797 | 0.8761 | 0.8673 |
| Mol2vec      | 0.8565| 0.8735| 0.5023| 0.8837 | 0.8987 | 0.5000 |
| SPVec        | 0.8604| 0.8689| 0.7502| 0.8796 | 0.8778 | 0.5000 |
| Doc2Vec      | 0.8652| 0.8661| 0.8696| 0.8589 | 0.8664 | 0.8512 |
| BERTSMILES   | 0.9880| 0.9921| 0.8616| 0.9655 | 0.9875 | 0.4863 |
| GPTSMILES    | 0.9797| 0.9951| 0.8771| 0.9376 | 0.9732 | 0.7117 |
| LLaMASMILES  | 0.9651 | 0.9957| 0.8578| 0.9105 | 0.9521 | 0.7828 |
| LLaMA2SMILES | 0.9916| 0.9954| 0.8338| 0.9004 | 0.9480 | 0.7767  |
| BERTDesc     | 0.9881| 0.9918| 0.8937| 0.9752 | 0.9860 | 0.6264 |
| GPTDesc      | 0.9948  | 0.9946| 0.9445| 0.9816 | 0.9856 | 0.6925 |
| LLaMADesc    | 0.9959 | 0.9885| 0.9452| 0.9537 | 0.9630 | 0.8752 |
| LLaMA2Desc   | 0.994 | 0.9884| 0.9254| 0.9447| 0.9620 | 0.8284 |


**`AUPR BioSnap`**
| Embedding    | 0.01       | 0.001      | 0.0001     | 0.0002     | 0.0003     | 1.00E-05   |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| No Feature   | 0.8422 | 0.8485 | 0.84913 | 0.8442 | 0.8586 | 0.8215 |
| Morgan       | 0.8508 | 0.8614 | 0.8391 | 0.8577| 0.8560 | 0.8376 |
| Mol2vec      | 0.8499 | 0.8183 | 0.7507 | 0.8681 | 0.8646 | 0.7500 |
| SPVec        | 0.8484 | 0.8379 | 0.8262 | 0.8580 | 0.8523 | 0.7500 |
| Doc2Vec      | 0.8429 | 0.8441 | 0.8469 | 0.8397| 0.8461 | 0.8068   |
| BERTSMILES   | 0.9846 | 0.9922 | 0.8167 | 0.9642 | 0.9871 | 0.5005   |
| GPTSMILES    | 0.9753 | 0.9954 | 0.9005 | 0.9476 | 0.9758 | 0.7706 |
| LLaMASMILES  | 0.9362 | 0.9964 | 0.8731 | 0.9133| 0.9512| 0.8042 |
| LLaMA2SMILES | 0.9873 | 0.9947 | 0.8576 | 0.9129  | 0.9491| 0.8177 |
| BERTDesc     | 0.9812 | 0.9907 | 0.8617 | 0.9730 | 0.9848 | 0.6397 |
| GPTDesc      | 0.9914 | 0.9926 | 0.9473 | 0.9792| 0.9820 | 0.7488 |
| LLaMADesc    | 0.9941 | 0.9850 | 0.9447 | 0.9499| 0.95771 | 0.8930 |
| LLaMA2Desc   | 0.9924 | 0.9858 | 0.9306 | 0.9455 | 0.9595 | 0.8510 |

**`AUROC DrugBank`**
| Embedding    | 0.01       | 0.001      | 0.0001     | 0.0002     | 0.0003     | 1.00E-05   |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| No Feature   | 0.8753 | 0.8775 | 0.8773 | 0.8767 | 0.8751| 0.8743 |
| Morgan       | 0.8781 | 0.8776 | 0.8797 | 0.8773 | 0.8775 | 0.8804 |
| Mol2vec      | 0.5601 | 0.5607  | 0.5663 | 0.5644 | 0.5540 | 0.5538 |
| SPVec        | 0.5277 | 0.5906 | 0.6276 | 0.6003 | 0.6346 | 0.5396 |
| Doc2Vec      | 0.8725 | 0.8706 | 0.8694| 0.8739 | 0.8709 | 0.8707|
| BERTSMILES   | 0.8512 | 0.8939 | 0.8936| 0.8675 | 0.8369 | 0.8760 |
| GPTSMILES    | 0.8856 | 0.8739 | 0.8760| 0.8731 | 0.8760 | 0.8778 |
| LLaMASMILES  | 0.8577| 0.8616 | 0.8693 | 0.8549 | 0.8644 | 0.8767 |
| LLaMA2SMILES | 0.8438 | 0.8342 | 0.8375 | 0.8402 | 0.8313 | 0.8272 |
| BERTDesc     | 0.9321 | 0.9001 | 0.9157 | 0.9207 | 0.8926 | 0.9198|
| GPTDesc      | 0.9497 | 0.9445 | 0.9493 | 0.9465 | 0.9424 | 0.9490 |
| LLaMADesc    | 0.9334| 0.9391 | 0.9404 | 0.9374 | 0.9456  | 0.9379 |
| LLaMA2Desc   | 0.9442| 0.9325  | 0.9401 | 0.9347 | 0.9380 | 0.9368 |

**`AUPR DrugBank`**
| Embedding    | 0.01       | 0.001      | 0.0001     | 0.0002     | 0.0003     | 1.00E-05   |
| ------------ | ----- | ------ | ------ | ------ | ------ | -------|
| no_feature   | 0.8569 | 0.8651 | 0.8600| 0.8622 | 0.8569 | 0.8555 |
| Morgan       | 0.8591 | 0.8621 | 0.8594 | 0.8587  | 0.8569 | 0.8610 |
| Mol2vec      | 0.7657 | 0.7654 | 0.7662| 0.7674| 0.7641 | 0.7646 |
| SPVec        | 0.7568 | 0.7737 | 0.7854 | 0.7762 | 0.7852 | 0.7602 |
| Doc2Vec      | 0.8525 | 0.8503 | 0.8481 | 0.8530 | 0.8508 | 0.8501 |
| BERTSMILES   | 0.7972  | 0.8518 | 0.8337 | 0.7872 | 0.7455 | 0.8365 |
| GPTSMILES    | 0.9057 | 0.8947 | 0.8972 | 0.8988 | 0.8933 | 0.9067 |
| LLaMASMILES  | 0.8629 | 0.8775 | 0.8811 | 0.8719 | 0.8764 | 0.8877 |
| LLaMA2SMILES | 0.8658  | 0.8595 | 0.8633 | 0.8626 | 0.8479 | 0.8515 |
| BERTDesc     | 0.9066 | 0.8607 | 0.8889  | 0.9043  | 0.8415 | 0.8951 |
| GPTDesc      | 0.9538 | 0.9512 | 0.9558| 0.9521 | 0.9481 | 0.9501 |
| LLaMADesc    | 0.9337 | 0.9358 | 0.9361| 0.9325   | 0.9446 | 0.9347 |
| LLaMA2Desc   | 0.9452 | 0.9350 | 0.9407 | 0.9369 | 0.9403 | 0.9420 |

* Study on `Learning Rate`
  
  <img width="300" height="300" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/auc-biosnap-lr.png">

  <img width="300" height="300" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/pr-biosnap-lr.png">

  <img width="300" height="300" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/auc-drugbank-lr.png">

  <img width="300" height="300" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/pr-drugbank-lr.png">



# 4. Visualization

<p align="center">
  <img width="800" height="600" src="https://github.com/sshaghayeghs/DDI-LLM/blob/main/Image/GridVis.png">
</p>
