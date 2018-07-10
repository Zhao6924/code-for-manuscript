import numpy as np;
import bilsm_crf_model
EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model(train=True)
# train model
print(train_x.shape,train_y.shape)
model.fit(train_x, train_y,batch_size=16)
model.save('model/crf.h5')
