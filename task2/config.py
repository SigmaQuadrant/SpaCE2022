
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subtask', type=str, default='subtask2')
parser.add_argument('--model', type=str, default='chinese-deberta-large')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=0.01)
arguments = parser.parse_args().__dict__

data_dir = '../dataset'
json_dir = '/json'
jsonl_dir = '/jsonl'
subtask = arguments['subtask']

original_dir = data_dir + jsonl_dir + '/task2_train.jsonl'
subtask_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_train.jsonl'
train_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_train.jsonl'
dev_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_dev.jsonl'
test_dir = data_dir + jsonl_dir + '/' + subtask + '/task2-test.jsonl'

bert_model_dir = '../pretrained_model/'
bert_model_name = arguments['model']
bert_model = bert_model_dir + bert_model_name

shuffle = True
bert_cased = True
device = torch.device('cuda')

learning_rate = arguments['lr']
weight_decay = arguments['weight_decay']
clip_grad = 5
seed = arguments['seed']

batch_size = arguments['batch_size']
epoch = 30
patience = 0.0002
patience_num = 5
min_epoch_num = 5

tmp = arguments['tmp']
if tmp:
    model_dir = './experiments/' + subtask + '/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + '_sd_' + str(seed) + 'tmp'
else:
    model_dir = './experiments/' + subtask + '/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + '_sd_' + str(seed)
log_dir = model_dir + '/train.log'
prediction_dir = model_dir + '/prediction.jsonl'

if __name__ == '__main__':
    print(original_dir)
    print(train_dir)
    print(subtask)
    print(type(subtask))
    print(model_dir)
    print(log_dir)