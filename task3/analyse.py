import math

import jsonlines

train_dir = '../dataset/jsonl/task3_train.jsonl'

# calculate

with open(train_dir, 'r') as f:
    total, co_ref = 0, 0
    num_triples = []
    num_elements = []
    position_cnt = [0] * 18
    for idx, item in enumerate(jsonlines.Reader(f)):
        total = total + 1
        if item['corefs']:
            co_ref = co_ref + 1
        num_triples.append(len(item['outputs']))
        num_elements.extend([len([item for item in triple if item is not None]) for triple in item['outputs']])
        for triple in item['outputs']:
            for ind, value in enumerate(triple):
                if value is not None:
                    position_cnt[ind] = position_cnt[ind] + 1


def mu_and_sigma(A):
    mu = sum(A) / len(A)
    sigma = math.sqrt(sum([(v - mu) * (v  - mu) for v in A]) / len(A))
    return mu, sigma

print(co_ref, total, co_ref/total)
print(mu_and_sigma(num_triples))
print(position_cnt)




