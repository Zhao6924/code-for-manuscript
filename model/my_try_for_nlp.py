#encoding=utf-8
import platform
from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from collections import Counter
import numpy as np
def load_embedding(maxlen=300):
    data = np.random.normal(-0.5, 0.5, (4772,200));
    return data;
def find_index(vocab,x):
    index =0;
    if(x not in vocab):
        vocab.append(x)
    return vocab
def trans(vocab,word):
    index=0;
    for i in vocab:
        if(i==x):
            return index;
        index=index+1;
    return 100000;
def make_sequence(vocab):
    data=load_embedding();
    return data;

def mark(x,index):
    x[index]=1;
    return x;
def check(s):
    x=[0,0,0,0,0,0,0]
    if(len(s)==1):
        x=mark(x,0);
        return x;
    if(s[0]=='B' and s[4]=='R'):
        x = mark(x,1);
        return x;
    elif (s[0]=='I'and s[4]=='R'):
        x = mark(x,2);
        return x;
    elif (s[0]=='B' and s[4]=='C'):
        x = mark(x,3);
        return x;
    elif(s[0]=='I'and s[4]=='C'):
        x = mark(x,4);
        return x;
    elif(s[0]=='B' and s[4]=='G'):
        x = mark(x,5);
        return x;
    elif(s[0]=='I' and s[4]=='G'):
        x = mark(x,6);
        return x;
def parse_data(files):
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'
    string = files.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    files.close()
    return data

def PARSE_DATA(files):
    split_text = '\r\n'
    line=files.read().decode('utf-8')
    line=line.strip()
    character=[]
    labels_tag=[];
    ziFlag=True
    temp=[];
    for i in line:
        list1=[];tmp=[];
        for x in i:
            if( x=='\n'):
                break;
            list1.append(x)
        if(len(list1)==0):
            continue;
        if(ziFlag==True and list1[0]!='\r'):
            character.append(list1[0])
            p=list1[0];
            tmp=[]
            ziFlag=False;
        elif(list1[0]==' '):
            continue
        elif (list1[0]=='\r'):

            ziFlag=True
            continue;
        else:
            tmp.append(list1[0])
            if(len(tmp)==1 and tmp[0]=='O'):
                labels_tag.append(0);
            if(len(tmp)==5):
                labels_tag.append(check(tmp))
            #print(p,tmp)
    line = files.readline().decode('utf-8')
    for i in line:
        print(i)
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    datax=[];
    datay=[];
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            for i in sent_:
                datax.append(i)
            for i in tag_:
                datay.append(i)
            sent_, tag_ = [], []
        indedx=0;length=len(sent_);
        length1=len(tag_)
    #print(len(datax),len(datay))

    return datax,datay
x,y=read_corpus('D:/python_project/zh-NER-keras/data/train_data.data')
length=len(x)
Ytrain=[];
for i in y:
    p=check(i)
    Ytrain.append(p)
#chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
print(Ytrain)
def load_data():
    train=parse_data(open('D:/python_project/zh-NER-keras/data/train_data.data', 'rb'))
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    #print(word_counts)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    return vocab
# define documents
zidian=load_data()
vocab=zidian

pad=make_sequence(vocab=zidian)
print(zidian)
for i in x:
    if(i not in vocab):
        vocab.append(i)
        print(i)
wordmap={}
index=0;
for i in vocab:
    wordmap[i]=index
    index=index+1
#print(pad)
embedding_mat=load_embedding()
trainx=[];
for i in x:

    temp=embedding_mat[wordmap[i]]
    trainx.append(temp)
    print(temp)
print(trainx)
print(len(trainx),len(Ytrain))