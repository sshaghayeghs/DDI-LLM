import logging
import json
from tqdm import tqdm

import torch
from transformers import BertModel, AutoTokenizer

import tensorflow as tf
import numpy as np
import pandas as pd

from data_bio import get_bio_dataset


class Bert_Embeddings:
    def __init__(self, model_name, datasets):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = model_name
        self.model = BertModel.from_pretrained("bert-base-uncased",
                                                # output_hidden_states = True
                                                ) # Whether the model returns all hidden-states.)
        self.model.to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        logging.info('loading datasets')
        self.datasets = datasets
        for dataset in self.datasets:
            print('>>>>>>>>', self.model_name, dataset, '<<<<<<<<')
            self.dataset_name = dataset
            dataset = get_dataset(dataset)
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            self.get_embeddings(save_embeddings=True)

    def get_embeddings(self, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            save_embeddings: T/F value to save the model in results directory
        '''
        embed_train_data = True
        embed_test_data = True

        if(embed_train_data):
            embeddings_train = []
            for data_row in tqdm(self.train_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['text'],
                    add_special_tokens=True,
                    max_length=self.tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids'].to("cuda")
                token_type_ids = tokens['token_type_ids'].to("cuda")

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze().to("cpu")
                    embeddings_train.append(np.array(embedding))
                

        
        if(embed_test_data):
            embeddings_test = []
            for data_row in tqdm(self.test_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['text'],
                    add_special_tokens=True,
                    max_length=self.tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids'].to("cuda")
                token_type_ids = tokens['token_type_ids'].to("cuda")

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze().to("cpu")
                    embeddings_test.append(np.array(embedding))
        
        if (save_embeddings and embed_train_data):
                    logging.info('saving train data embeddings')
                    embeddings_train = np.array(embeddings_train)
                    train_data = pd.concat([pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
                    train_data.to_csv(f'results/embeddings/{self.model_name}_{self.dataset_name}_embeddings_train.csv', sep='\t')
        
        if (save_embeddings and embed_test_data):
            logging.info('saving test data embeddings')
            embeddings_test = np.array(embeddings_test)
            test_data = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(embeddings_test)], axis=1)
            test_data.to_csv(f'results/embeddings/{self.model_name}_{self.dataset_name}_embeddings_test.csv', sep='\t')
