#encoding=utf-8
from keras.preprocessing.sequence import pad_sequences
x=[1,1,1,1,1,1]
x=pad_sequences([x],100);
print(x);
