
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mbart-large-50')
parser.add_argument('--lr', type=float, default=1.9e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=44)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_beams', type=int, default=1)
arguments = parser.parse_args().__dict__

data_dir = '../dataset'
json_dir = '/json'
jsonl_dir = '/jsonl'

train_dir = data_dir + jsonl_dir + '/task3_train.jsonl'
dev_dir = data_dir + jsonl_dir + '/task3_dev.jsonl'
test_dir = data_dir + jsonl_dir + '/task3-test.jsonl'

bert_model_dir = '../pretrained_model/'
bert_model_name = arguments['model']
bert_model = bert_model_dir + bert_model_name

shuffle = True
bert_cased = True
num_beams = arguments['num_beams']
device = torch.device('cuda')

learning_rate = arguments['lr']
weight_decay = arguments['weight_decay']
clip_grad = 5
seed = arguments['seed']

batch_size = arguments['batch_size']
epoch = 50
patience = 0.0002
patience_num = 5
min_epoch_num = 5

model_dir = './experiments/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + '_sd_' + str(seed) + '_fix'
log_dir = model_dir + '/train.log'
prediction_dir = model_dir + '/prediction_mbart.jsonl'



