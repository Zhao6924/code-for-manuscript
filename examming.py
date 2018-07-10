#encoding=utf-8
from keras.layers import Dense,Embedding
import numpy as np
import tensorflow as tf
x=Embedding(100,100,mask_zero=True);
print(x)
