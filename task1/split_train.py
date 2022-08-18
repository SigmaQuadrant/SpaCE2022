import json
import jsonlines
from config import original_dir, train_dir, dev_dir, test_dir

with open(original_dir, 'r') as fr:
    # length = 10993
    # 0.8/0.1/0.1 8794/9896/10993
    train, dev, test = [], [], []
    for idx, item in enumerate(jsonlines.Reader(fr)):
        context, judge = item['context'], item['judge']
        if 1 <= idx <= 8794:
            train.append({'context': context, 'judge': judge})
        elif 8794 < idx <= 9896:
            dev.append({'context': context, 'judge': judge})
        else:
            test.append({'context': context, 'judge': judge})

# split the train into train/dev/test

with jsonlines.open(train_dir, 'w') as train_f:
    train_f.write_all(train)
with jsonlines.open(dev_dir, 'w') as dev_f:
    dev_f.write_all(dev)
with jsonlines.open(test_dir, 'w') as test_f:
    test_f.write_all(test)


