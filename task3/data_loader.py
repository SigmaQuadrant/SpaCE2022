import torch
import jsonlines
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import DataLoader

map_6 = {
    '说话时': 0,
    '过去': 1,
    '将来': 2,
    '之时': 3,
    '之前': 4,
    '之后': 5
}
map_17 = {
    '远': 0,
    '近': 1,
    '变远': 2,
    '变近': 3
}


class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode: str):
        self.mode = mode
        # in ['train'/'dev'/'test']
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.device = config.device
        self.dataset = self.preprocess(file_path)

    def preprocess(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for idx, item in enumerate(jsonlines.Reader(f)):

                tokens = self.tokenizer.encode(item['context'])
                corefs = [[entity['idxes'] for entity in coref] for coref in item['corefs']]
                outputs = []
                for triple in item['outputs']:
                    output = []
                    for position, element in enumerate(triple):
                        if element is None:
                            output.append(None)
                        elif type(element) == dict:
                            output.append(element['idxes'])
                        elif type(element) == str:
                            output.append(self.mapping(position, element))
                    outputs.append(output)

                if self.mode == 'test':
                    data.append([tokens])
                else:
                    data.append([tokens, outputs, corefs])
        return data

    def __getitem__(self, idx):
        tokens = self.dataset[idx][0]
        if self.mode == 'test':
            return [tokens]
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
            return [batch_data]
        else:
            outputs = [x[1] for x in batch]
            corefs = [x[2] for x in batch]
            return [batch_data, outputs, corefs]


    @staticmethod
    def mapping(pos, element):
        if pos == 3 and element is not None:
            return 0
        if pos == 6:
            return map_6[element]
        if pos == 17:
            return map_17[element]


if __name__ == '__main__':
    train_dataset = SpaceDataset(config.train_dir, config, 'dev')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=train_dataset.collate_fn)
    for idx, item in enumerate(train_loader):
        for batch_triple in item[1]:
            for triple in batch_triple:
                assert(len(triple) == 18)