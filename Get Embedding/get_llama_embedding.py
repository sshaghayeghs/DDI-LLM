import logging
import json
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import infer_auto_device_map

import tensorflow as tf
import numpy as np
import pandas as pd

from data_bio import get_bio_dataset


class Llama_Embeddings:
    def __init__(self, model_name, datasets):

        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading model and tokenizer')
        self.model_name = model_name
        ##! generate an error if the model name is not a valid key
        models_path = {
            "llama-7B": "./llama_converted/7B",
            "llama-13B": "./llama_converted/13B",
            "llama-30B": "./llama_converted/30B",
            "llama-65B": "./llama_converted/65B",
            "llama2-7B": "./llama2_converted/7B",
            "llama2-13B": "./llama2_converted/13B",
            "llama2-70B": "./llama2_converted/70B"
        }
        PATH_TO_CONVERTED_WEIGHTS = models_path[model_name]

        # Set device to auto to utilize GPU
        device = "auto"  # balanced_low_0, auto, balanced, sequential

        self.model = LlamaForCausalLM.from_pretrained(
            PATH_TO_CONVERTED_WEIGHTS,
            device_map=device,
            output_hidden_states=True
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            PATH_TO_CONVERTED_WEIGHTS
        )
        # unknow tokens. we want this to be different from the eos token
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"

        logging.info('loading datasets')
        self.datasets = datasets
        for dataset in self.datasets:
            print('>>>>>>>>', self.model_name, dataset, '<<<<<<<<')
            self.dataset_name = dataset
            dataset = get_small_dataset(dataset)
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            train_embeddings = self.get_embeddings(
                self.train_data, self.dataset_name+'_train', save_embeddings=True)
            test_embeddings = self.get_embeddings(
                self.test_data, self.dataset_name+'_test', save_embeddings=True)

    def get_embeddings(self, dataset, dataset_name, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            arguments:
                save_embeddings: T/F value to save the model in results directory
        '''

        embeddings = []
        for data_row in tqdm(self.train_data):
            tokens = self.tokenizer(data_row['text'])
            input_ids = tokens['input_ids']
            using get_input_embedding method
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()
                embedding = input_embeddings(
                    torch.LongTensor([input_ids]))
                embedding = torch.mean(
                    embedding[0], 0).cpu().detach()

                embeddings.append(embedding)

        if (save_embeddings):
            logging.info('saving train data embeddings')
            embeddings = np.array(embeddings)
            embeddings = pd.concat(
                [pd.DataFrame(self.train_data), pd.DataFrame(embeddings)], axis=1)
            embeddings.to_csv(
                f'results/embeddings/{self.model_name}_{dataset_name}_embeddings.csv', sep='\t')

        return embeddings