import torch
import config
from torch.nn import CrossEntropyLoss

from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer

if __name__ == '__main__':
    ckpt = './experiments/chinese-bart-large_lr_1e-05_bsz_16sd42'
    tokenizer = BertTokenizer.from_pretrained('../pretrained_model/chinese-bart-large')
    model = BartForConditionalGeneration.from_pretrained(ckpt)
    model.to(torch.device('cuda'))
    train_dataset = SpaceDataset(config.train_dir, config)
    dev_dataset = SpaceDataset(config.dev_dir, config)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=dev_dataset.collate_fn)

    for idx, sample in enumerate(dev_loader):
        if idx > 25:
            break
        data, answer = sample
        outputs = model(input_ids=data, labels=data)
        # print(outputs)
        # print(outputs.loss.item())
        with torch.no_grad():
            loss_fct = CrossEntropyLoss(reduction='mean')
            logit = outputs.logits.squeeze()[:-1] # [len_s - 1 , vocab_size]
            label = data.squeeze()[1:]    # [len_s - 1, 1]
            # print(logit.shape, label.shape)
            # print(loss_fct(logit, label))
            perplex = torch.exp(loss_fct(logit, label))
            print(perplex, answer)
