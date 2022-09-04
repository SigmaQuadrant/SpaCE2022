import torch
import jsonlines
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import DataLoader

map_6 = {
    '说话时': 'T1',
    '过去': 'T2',
    '将来': 'T3',
    '之时': 'T4',
    '之前': 'T5',
    '之后': 'T6'
}
map_17 = {
    '远': 'FAR',
    '近': 'NEAR',
    '变远': 'FARTHER',
    '变近': 'NEARER'
}

special_token_dicts = {
    'additional_special_tokens': [
        'P0', 'P1', 'P2', 'P3', 'P4',
        'P5', 'P6', 'P7', 'P8', 'P9',
        'P10', 'P11', 'P12', 'P13', 'P14',
        'P15', 'P16', 'P17'
        'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'FAR', 'FARTHER', 'NEAR', 'NEARER', 'FALSE'
    ]
}

class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode: str):
        self.mode = mode
        # in ['train'/'dev'/'test']
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in special_token_dicts['additional_special_tokens']]
        self.device = config.device
        self.dataset = self.preprocess(file_path)

    def preprocess(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for idx, item in enumerate(jsonlines.Reader(f)):

                tokens = self.tokenizer.encode(item['context'])
                corefs = [[entity['idxes'] for entity in coref] for coref in item['corefs']]
                outputs = []
                for cur_n, triple in enumerate(item['outputs']):
                    if cur_n != 0:
                        outputs.append(self.tokenizer.sep_token_id)
                    for position, element in enumerate(triple):
                        if element is None:
                            continue
                        elif type(element) == dict:
                            outputs.append(self.tokenizer.convert_tokens_to_ids('P'+str(position)))
                            outputs.extend([self.tokenizer.convert_tokens_to_ids(char) for char in element['text']])
                        elif type(element) == str:
                            outputs.append(self.tokenizer.convert_tokens_to_ids('P' + str(position)))
                            outputs.append(self.tokenizer.convert_tokens_to_ids(self.mapping(position, element)))

                if self.mode == 'test':
                    data.append([tokens, corefs])
                else:
                    data.append([tokens, outputs, corefs])
        return data

    def __getitem__(self, idx):
        tokens = self.dataset[idx][0]
        if self.mode == 'test':
            corefs = self.dataset[idx][1]
            return [tokens, corefs]
        else:
            output = self.dataset[idx][1]
            corefs = self.dataset[idx][2]
            return [tokens, output, corefs]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        sentence = [x[0] for x in batch]
        batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
        if self.mode == 'test':
            corefs = [x[1] for x in batch]
            return [batch_data, corefs]
        else:
            outputs = [x[1] for x in batch]
            corefs = [x[2] for x in batch]
            return [batch_data, outputs, corefs]


    @staticmethod
    def mapping(pos, element):
        if pos == 3 and element is not None:
            return 'FALSE'
        if pos == 6:
            return map_6[element]
        if pos == 17:
            return map_17[element]


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
    # tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)
    train_dataset = SpaceDataset(config.train_dir, config, 'train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=train_dataset.collate_fn)
    for idx, item in enumerate(train_loader):
        print(item)
        # print(tokenizer.convert_ids_to_tokens(item[1][0]))
