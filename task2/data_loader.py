import torch
import json
import jsonlines
import numpy as np
from transformers import BertTokenizer, DebertaTokenizer
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import config
# test the model
from model import DebertaReader
from torch.utils.data import DataLoader

# subtask1 1005 train 
# subtask2
# subtask3
class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode: str):
        self.mode = mode # ['train'/'dev'/'test']
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.device = config.device
        self.subtask = config.subtask
        self.dataset = self.preprocess(file_path)

    def preprocess(self, file_path):

        if self.subtask == 'subtask1':
            data = []
            with open(file_path, "r") as fr:
                for item in jsonlines.Reader(fr):
                    text, reasons, labels = item['context'], item['reasons'], []
                    for reason in reasons:
                        fragment = reason['fragments']
                        # choose the first span
                        role1, role2 = fragment[0], fragment[1]
                        labels.extend([role1['idxes'][0], role1['idxes'][-1]])
                        labels.extend([role2['idxes'][0], role2['idxes'][-1]])
                    tokens = self.tokenizer.convert_tokens_to_ids(list(text))
                    if self.mode == 'test':
                        data.append([tokens])
                    else:
                        data.append([tokens, labels])
            return data

        elif self.subtask == 'subtask3':
            data = []
            with open(file_path, 'r') as fr:
                for item in jsonlines.Reader(fr):
                    text, reasons, labels = item['context'], item['reasons'], []
                    for reason in reasons:
                        fragment = reason['fragments']
                        answer = [-1] * 6
                        for element in fragment:
                            if element['role'] == 'S':
                                answer[0], answer[1] = element['idxes'][0], element['idxes'][-1]
                            elif element['role'] == 'P':
                                answer[2], answer[3] = element['idxes'][0], element['idxes'][-1]
                            elif element['role'] == 'E':
                                answer[4], answer[5] = element['idxes'][0], element['idxes'][-1]
                        labels.extend(answer)
                    tokens = self.tokenizer.convert_tokens_to_ids(list(text))
                    if self.mode == 'test':
                        data.append([tokens])
                    else:
                        data.append([tokens, labels])
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
        if self.subtask == 'subtask1':
            sentence = [x[0] for x in batch]
            batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
            if self.mode == 'test':
                return [batch_data]
            elif self.mode == 'train':
                # setting 1: choose the shorter span
                labels = [x[1] for x in batch]
                for idx, label in enumerate(labels):
                    if len(label) == 8:
                        if (label[1] - label[0] + label[3] - label[2]) < (label[5] - label[4] + label[7] - label[6]):
                            labels[idx] = label[0:4]
                        else:
                            labels[idx] = label[4:8]
                batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]

                # setting 2: choose all the spans
                # setting 3: separate into two parts
            else:
                 # save all the label
                 labels = [x[1] for x in batch]
                 batch_label = pad_sequence([torch.from_numpy(np.array(s)) for s in labels], batch_first=True, padding_value=-1)
                 batch_label = torch.as_tensor(batch_label, dtype=torch.long).to(self.device)
                 return [batch_data, batch_label]

        # TODO
        elif self.subtask == 'subtask3':
            sentence = [x[0] for x in batch]
            batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
            if self.mode == 'test':
                return [batch_data]
            elif self.mode == 'train':
                labels = [x[1] for x in batch]
                for idx, label in enumerate(labels):
                    if len(label) > 6:
                        labels[idx] = label[0:6]
                batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]
            else:
                labels = [x[1] for x in batch]
                batch_label = pad_sequence([torch.from_numpy(np.array(s)) for s in labels], batch_first=True, padding_value=-2)
                batch_label = torch.as_tensor(batch_label, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]


if __name__ == '__main__':
    train_dataset = SpaceDataset(config.train_dir, config, 'train')
    dev_dataset = SpaceDataset(config.dev_dir, config, 'dev')
    # qa_model = DebertaReader.from_pretrained(config.bert_model)
    # qa_model.to(torch.device('cuda'))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=dev_dataset.collate_fn)

    for idx, sample in enumerate(train_loader):

        batch_data, batch_label = sample
        batch_mask = batch_data.gt(0)
        print(idx, batch_data.shape, batch_label.shape)

    for idx, sample in enumerate(dev_loader):

        batch_data, batch_label = sample
        batch_mask = batch_data.gt(0)
        print(idx, batch_data.shape, batch_label.shape)
        if batch_label.size(1) == 8:
            print(batch_label)

    '''
    count = 0
    with open(file_path, "r") as fr:
        for item in jsonlines.Reader(fr):
            count += 1
            if count == 389:
                text = item['context']
                # print(text)
                # print(list(text))
                tokenizers = BertTokenizer.from_pretrained(config.bert_model)
                # print(text)
                result1 = tokenizers.encode(text)
                print(result1)
                print(len(result1))
                result2 = tokenizers.convert_tokens_to_ids(list(text))
                print(result2)
                print(len(result2))
    
    '''
