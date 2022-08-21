
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subtask', type=str, default='subtask1')
parser.add_argument('--model', type=str, default='chinese-deberta-large')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
arguments = parser.parse_args().__dict__

data_dir = '../dataset'
json_dir = '/json'
jsonl_dir = '/jsonl'
subtask = arguments['subtask']

original_dir = data_dir + jsonl_dir + '/task2_train.jsonl'
subtask_dir = data_dir + jsonl_dir + '/' + subtask + '/task2.jsonl'
train_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_train.jsonl'
dev_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_dev.jsonl'
test_dir = data_dir + jsonl_dir + '/' + subtask + '/task2-splited-test.jsonl'

bert_model_dir = '../pretrained_model/'
bert_model_name = arguments['model']
bert_model = bert_model_dir + bert_model_name

shuffle = True
bert_cased = True
device = torch.device('cuda')

learning_rate = arguments['lr']
weight_decay = 0.01
clip_grad = 5
seed = arguments['seed']

batch_size = arguments['batch_size']
epoch = 30
patience = 0.0002
patience_num = 5
min_epoch_num = 5

model_dir = './experiments/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + '_sd_' + str(seed)
log_dir = model_dir + '/train.log'
prediction_dir = model_dir + '/prediction.jsonl'

if __name__ == '__main__':
    print(original_dir)
    print(train_dir)
    print(subtask)
    print(type(subtask))