#encoding=utf-8
import numpy as np
import pickle
import platform
import bilsm_crf_model
import process_data
import numpy as np
chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
def find_index(s):
    for i in range(7):
        if(s[i]==1):
            print(i)
            return i
def check(s,reavalue):
    positions=find_index(s)
    if(chunk_tags[positions]==reavalue):
        return True
    return  False
def read_data():
    with open('model/config.pkl', 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
        return vocab,chunk_tags;
def _PARSE_data(fh=None):
    fh=open('data/test_data.data', 'rb')
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data
x= _PARSE_data()
Ttrain=[]
tmp=[];
all=[];
Flag=True
s=""
model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
sum=0;
right=0;
sum=0.0;
right=0.0;
for i in range(len(x)):
    tmpx=[];tmpy=[];
    for j in ((x[i])):
        tmpx.append(j[0])
        tmpy.append(j[1])
    s=''.join(tmpx)
    predict_text=s;
    str, length = process_data.process_data(predict_text, vocab)
    model.load_weights('model/crf.h5')
    raw = model.predict(str)[0][-length:]
    for index in range(len(raw)):
        print("----------------------")
        print((raw[index]),tmpy[index])
        print("**********************")
        print(check(raw[index],tmpy[index]))
        if(check(raw[index],tmpy[index])):
            right=right+1.0
        sum=sum+1.0
        print(right,sum,right/sum)