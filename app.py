import spacy
import csv
import itertools
import re
import random
import time
from spacy.util import minibatch, compounding
from spacy.training import Example

# HYPERPARAMETER
EPOCH = 100
ROW = 3091  # batas yang bisa 3091, kalo 3092 error, mari kita cari tahu
BATCH_SIZE = 10
LEARN_RATE = 0.001
DROP_RATE = 0.35


def loaddata(ROW):
    traincsv = csv.reader(open('train.csv'))
    header = next(traincsv)

    result = []
    for row in itertools.islice(traincsv, 0, ROW):
        rowtext = row[1]
        rowPOI, rowstreet = row[2].split('/')
        indexPOI = re.search(fr'\b{rowPOI}\b', rowtext)
        indexstreet = re.search(fr'\b{rowstreet}\b', rowtext)

        entities = []
        if rowPOI:
            if not indexPOI:
                continue
            else:
                entities.append((indexPOI.start(), indexPOI.end(), 'poi'))

        if rowstreet:
            if not indexstreet:
                continue
            else:
                entities.append((indexstreet.start(), indexstreet.end(), 'street'))

        result.append((
            rowtext,
            {'entities': entities}
        ))
    return result


spacy.prefer_gpu()
nlp = spacy.blank('id', config={"nlp": {"batch_size": BATCH_SIZE}})
nlp.add_pipe('ner')

ner = nlp.get_pipe('ner')
ner.add_label('street')
ner.add_label('poi')

train_data = loaddata(ROW)
# print(train_data)

optimizer = nlp.initialize()
optimizer.learn_rate = LEARN_RATE
for itn in range(EPOCH):
    start = time.time()
    random.shuffle(train_data)
    losses = {}
    examples = []

    for text, annots in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annots)
        examples.append(example)

    nlp.update(examples, sgd=optimizer, drop=DROP_RATE, losses=losses)
    elapsed = time.time() - start
    print(losses, f'{str(round(elapsed, 2))} detik')

doc2 = nlp("jln.tirta tawar, br. junjungan, ubud, barat jalan dajan rurung")
for ent in doc2.ents:
    print(ent.label_, ent.text)
