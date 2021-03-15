import spacy
import csv
import itertools
import re
import random
from spacy.util import minibatch, compounding
from spacy.training import Example

def loaddata():
  traincsv = csv.reader(open('train.csv'))
  header = next(traincsv)

  result = []
  for row in itertools.islice(traincsv, 0, 50):
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

nlp = spacy.blank('id')
nlp.add_pipe('ner')

ner = nlp.get_pipe('ner')
ner.add_label('street')
ner.add_label('poi')

train_data = loaddata()
# print(train_data)

optimizer = nlp.begin_training()
for itn in range(50):
  random.shuffle(train_data)
  losses = {}
  examples = []

  for text, annots in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annots)
    examples.append(example)

  nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
  print(losses)

doc2 = nlp("jln.tirta tawar, br. junjungan, ubud, barat jalan dajan rurung")
for ent in doc2.ents:
  print(ent.label_, ent.text)