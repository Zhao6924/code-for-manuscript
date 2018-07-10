import bilsm_crf_model
import process_data
import numpy as np
#model, (train_x, train_y), (test_x, test_y)= bilsm_crf_model.create_model(train=True)
model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下'
str, length = process_data.process_data(predict_text, vocab)
model.load_weights('model/crf.h5')
raw = model.predict(str)[0][-length:]
print(raw)
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]
print(result_tags)
print(result)
per, loc, org = '', '', ''
for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])
