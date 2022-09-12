import hanlp
import jsonlines

dev_dir = '../dataset/jsonl/task3_dev.jsonl'
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

with open(dev_dir, 'r') as f:
    for ind, item in enumerate(jsonlines.Reader(f)):

        srls = HanLP(item['context'], tasks='srl')['srl']
        prediction_set, golden_set = set(), set()
        for srl in srls:
            for role in srl:
                if role[1] == 'ARG0':
                    prediction_set.add(role[0])
        for triple in item['outputs']:
            golden_set.add(triple[0]['text'])

        print(prediction_set, golden_set)


