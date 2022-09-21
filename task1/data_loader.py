import torch
import jsonlines
import numpy as np
from transformers import BertTokenizer, MBart50Tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import DataLoader

class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode):
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.dataset = self.preprocess(file_path)
        self.device = config.device

    def preprocess(self, file_path):
        data = []
        if self.mode == 'test':
            with open(file_path, 'r') as fr:
                for item in jsonlines.Reader(fr):
                    text = item['context']
                    tokens = self.tokenizer.encode(text)
                    data.append([tokens])
        else:
            with open(file_path, "r") as fr:
                for item in jsonlines.Reader(fr):
                    text = item['context']
                    tokens = self.tokenizer.encode(text)
                    label = 1 if item['judge'] else 0
                    data.append([tokens, label])
        return data

    def __getitem__(self, idx):
        tokens = self.dataset[idx][0]
        if self.mode == 'test':
            return [tokens]
        else:
            label = self.dataset[idx][1]
            return [tokens, label]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        sentence = [x[0] for x in batch]
        batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
        if self.mode == 'test':
            return [batch_data]
        else:
            labels = [x[1] for x in batch]
            batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
            return [batch_data, batch_label]


if __name__ == '__main__':

    file_path = '../dataset/jsonl/task1_train.jsonl'
    dataset = SpaceDataset(file_path, config, test_flag=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    for idx, (data, label) in enumerate(dataloader):
        print(data.shape, label.shape)
