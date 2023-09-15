import llama_base
import bert_base
from data_bio import get_bio_dataset
'''
By running this file, a loop starts to generate embeddings based on all of the available models and datasets.
'''
'''
options for models_llama:
    "llama-7B", "llama-13B", "llama-30B", "llama-65B", "llama2-7B", "llama2-13B", "llama2-70B"
'''
models_llama = ["llama-7B", "llama-13B", "llama-30B",
                "llama-65B", "llama2-7B", "llama2-13B", "llama2-70B"]
models_bert = ["bert"]
'''
options for bio datasets:
    "hiv", "bace", "clintox"
'''

bio_datasets = ["hiv", "bace", "clintox"]
for model in models_llama:
    llama_base.Llama_Embeddings(model, bio_datasets)
for model in models_bert:
    llama_base.Bert_Embeddings(model, bio_datasets)
