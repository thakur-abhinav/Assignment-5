import struct as st
import numpy as np
import tensorflow as tf
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.layers import MaxPooling2D
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.callbacks import ReduceLROnPlateau
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def conv(x,f,ksize,input_shape):
  '''
  This function returns a conv layer with same padding
  and initialiser is xavier initialisation
  x=input, f=num of filters, ksize=kernel size
  '''
  W=np.random.randn(ksize,ksize,input_shape[2],f)*np.sqrt(2/ksize**2)
  b = np.zeros((1,1,1,f))
  layer = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding="SAME")
  layer += b
  return tf.nn.relu(layer)

def pool(x,ksize):
  '''
  This function returns a maxpool layer of stride 2
  '''
  return tf.nn.max_pool(x,
                        ksize=[1, ksize, ksize, 1],
                        strides=[1, 2, 2, 1],
                        padding="SAME")

def convk(f,ksize,input_shape):
  '''
  This function returns a keras conv layer
  '''
  return tf.keras.layers.Conv2D(filters=f,kernel_size=ksize,padding='same',activation='relu',input_shape=input_shape)

def poolk(poolsize):
  '''
  This function returns a keras maxpool layer
  '''
  return tf.keras.layers.MaxPool2D(pool_size=poolsize, strides=(2,2))


def img_transform(img):
  '''
  This function takes an input image and transforms it
  spatially by rotating theta angle, shifting k1 pixels
  in horizontal direction, shifting k2 pixels in vertical
  direction.
  '''
  for i in range(10000):
    k1,k2=np.random.randint(-5,6,2)
    theta=np.random.randint(-45,46)
    cos=math.cos(math.radians(theta))
    sin=math.sin(math.radians(theta))
    transform=[cos,sin,k1,-sin,cos,k2,0,0]
    #transform=[1,0,k1,0,1,k2,0,0]
    k=tf.contrib.image.transform(img[i],transform,interpolation='NEAREST')
    img[i]=np.ndarray(k)
  return img

def keras_cnn_model(X,Y,x,y):
  '''
  This function uses tf.keras to fit training data X and Y in the model.
  The model architecture is
  conv(3x3,32)-maxpool-conv(3x3,64)-maxpool-conv(1x1,64)-dense(256)-output(10)
  X,Y training data.  x,y validation data. 
  '''
  
  model = tf.keras.Sequential()
  sgd=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9, nesterov=True)
  lrdecay=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,min_lr=0.0005)
  es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
  model.add(convk(32,3,(28,28,1)))
  model.add(poolk(2))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(convk(64,3,(14,14,32)))
  model.add(poolk(2))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(convk(64,1,(7,7,64)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy' ,metrics=['accuracy'],optimizer=sgd)
  model.fit(X,Y,batch_size=128,epochs=6,validation_data=(x,y),callbacks=[lrdecay,es])
  return

def cnn_model(features,labels,mode):
  X=tf.reshape(features["x"], [-1, 28, 28, 1])
  c1=conv(X,32,3,(28,28,1))
  p1=pool(c1,2)
  c2=conv(p1,64,3,(14,14,64))
  p2=pool(c2,2)
  flat=tf.reshape(p2,shape=[-1,7*7*64])
  dense1 = tf.layers.dense(flat,units=256,activation=tf.nn.relu)
  out=tf.layers.dense(dense1,units=10,activation=tf.nn.softmax)
  predict={
    "classes":tf.argmax(input=out, axis=1),
    "probabilities":tf.nn.softmax(out)
    }
  if mode==tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predict)
    
  loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=out, scope='loss')
    
  if mode== tf.estimator.ModeKeys.TRAIN:
    optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
    op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=op )
    
  eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predict["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)
  

filename = {'images' : '/home/abhinav/Desktop/Assignment5_AT/train-images-idx3-ubyte' ,'labels' : '/home/abhinav/Desktop/Assignment5_AT/train-labels-idx1-ubyte'}
#reading images data
train_imagesfile = open(filename['images'],'rb')
magicimages = st.unpack('>4B',train_imagesfile.read(4))
nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images = 255-np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC,1))
images=images/255
#reading label data
train_labelfile = open(filename['labels'],'rb')
magiclabel = st.unpack('>4B',train_labelfile.read(4))
nLabels = st.unpack('>I',train_labelfile.read(4))[0] #num of labels
labels = np.asarray(st.unpack('>'+'B'*nLabels,train_labelfile.read(nLabels))).reshape((nLabels))
labelstemp=np.zeros((nLabels,10))
#converting labels in a 10 class probability distribution
for i in range(nLabels):
    labelstemp[i][labels[i]]=1
labels=labelstemp

#random image transform
op=img_transform(images)
with tf.Session() as sess:
  k=sess.run(op)

#training and validation split
train_x,dev_x,train_y,dev_y=train_test_split(images,labels,test_size=0.2,random_state=0)

#keras cnn model run
with tf.Session() as sess:
  op=keras_cnn_model(train_x,train_y,dev_x,dev_y)
  sess.run(op)

#cnn estimator model run
estimator = tf.estimator.Estimator(model_fn = cnn_model)
train_input = tf.estimator.inputs.numpy_input_fn(
    x={"x": tf.convert_to_tensor(train_x,dtype=tf.float32)},
    y=tf.convert_to_tensor(train_y,dtype=tf.int32),
    batch_size=128,
    num_epochs=None,
    shuffle=False)
estimator.train(input_fn=train_input, steps=1500)
fashion_classifier.train(input_fn=train_input_fn, steps=1500)
dev_input = tf.estimator.inputs.numpy_input_fn(
    x={"x":tf.convert_to_tensor(dev_x,dtype=tf.float32)},
    y=tf.convert_to_tensor(dev_y,dtype=tf.int32)
    num_epochs=1,
    shuffle=False)

eval_results = estimator.evaluate(input_fn=dev_input)
print(eval_results)



