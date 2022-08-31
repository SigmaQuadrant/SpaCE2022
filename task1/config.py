# we should set the parameters in cmd
# load the current hyperparameters to config.json
# set the seed for reproduction

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='electra-large')
parser.add_argument('--lr', type=float, default=1.2e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=0.01)
arguments = parser.parse_args().__dict__

data_dir = '../dataset'
json_dir = '/json'
jsonl_dir = '/jsonl'

original_dir = data_dir + jsonl_dir + '/task1_train.jsonl'
train_dir = data_dir + jsonl_dir + '/task1_train.jsonl'
dev_dir = data_dir + jsonl_dir + '/task1_dev.jsonl'
test_dir = data_dir + jsonl_dir + '/task1_dev.jsonl'
# train/dev/test

bert_model_dir = '../pretrained_model/'
bert_model_name = arguments['model']
bert_model = bert_model_dir + bert_model_name

learning_rate = arguments['lr']
weight_decay = arguments['weight_decay']
clip_grad = 5
seed = arguments['seed']

batch_size = arguments['batch_size']
epoch = 30
patience = 0.0002
patience_num = 5
min_epoch_num = 5
# optimizer =
# scheduler =

shuffle = True
bert_cased = True
device = torch.device('cuda')

'''
We store the model/train.log/result into a file, but i hope the file_name should
contain some parameters and can be distinguished 
'''

model_dir = './experiments/' + bert_model_name + '_lr_' + str(learning_rate) + '_bsz_' + str(batch_size) + 'sd' + str(seed)
log_dir = model_dir + '/train.log'
prediction_dir = model_dir + '/prediction.jsonl'

if __name__ == '__main__':
    print(bert_model)
    print(bert_model_name)
    print(model_dir)
    print(log_dir)
    print(prediction_dir)
    # print(log_dir.split('/')[:-1])