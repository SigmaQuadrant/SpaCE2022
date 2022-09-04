
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subtask', type=str, default='subtask1')
parser.add_argument('--model', type=str, default='chinese-deberta-large')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--modified', action='store_true')
arguments = parser.parse_args().__dict__

data_dir = '../dataset'
json_dir = '/json'
jsonl_dir = '/jsonl'
subtask = arguments['subtask']

original_dir = data_dir + jsonl_dir + '/task2_dev.jsonl'
subtask_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_dev.jsonl'
if arguments['modified']:
    train_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_train_modified.jsonl'
else:
    train_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_train.jsonl'
dev_dir = data_dir + jsonl_dir + '/' + subtask + '/task2_dev.jsonl'
test_dir = data_dir + jsonl_dir + '/task2_dev.jsonl' # one for all

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

best_lr_1, best_lr_2, best_lr_3 = 2e-5, 1.9e-5, 1.8e-5
subtask1_model_dir = model_dir = './experiments/' + 'subtask1' + '/' + bert_model_name + '_lr_' + str(best_lr_1) + '_bsz_' + str(batch_size) + '_sd_' + str(seed) + '_modified'
subtask2_model_dir = model_dir = './experiments/' + 'subtask2' + '/' + bert_model_name + '_lr_' + str(best_lr_2) + '_bsz_' + str(batch_size) + '_sd_' + str(seed) + '_modified'
subtask3_model_dir = model_dir = './experiments/' + 'subtask3' + '/' + bert_model_name + '_lr_' + str(best_lr_3) + '_bsz_' + str(batch_size) + '_sd_' + str(seed) + '_modified'

if arguments['modified']:
    model_dir = './experiments/' + subtask + '/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size)+ '_sd_' + str(seed) + '_modified'
else:
    model_dir = './experiments/' + subtask + '/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + '_sd_' + str(seed)
log_dir = model_dir + '/train.log'
prediction_dir = './experiments/prediction_tune.jsonl'

if __name__ == '__main__':
    print(subtask1_model_dir, subtask2_model_dir, subtask3_model_dir)
    # print(log_dir, prediction_dir)
    # print(model_dir)