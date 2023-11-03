import os
import cv2
import numpy as np

import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np

import keras
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
num_classes=4
def read_dataset():
    fol=os.listdir(r'D:\Dataset\archive\Training')
    x=[]
    y=[]
    j=0
    for i in fol:
        files = os.listdir(os.path.join(r'D:\Dataset\archive\Training',i))
        for f in files:
            fn = os.path.join(r'D:\Dataset\archive\Training', i,f)
            img=cv2.imread(fn,cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, (48,48), interpolation=cv2.INTER_CUBIC)
            x.append(res)
            y.append(j)

        j=j+1

    return  (np.asarray(x, dtype=np.float32), np.asarray(y))



Xdataset,ydataet=read_dataset()


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(Xdataset, ydataet, test_size=0.2, random_state=0)

y_train1=[]
for i in y_train:
    d = keras.utils.to_categorical(i, num_classes)

    y_train1.append(d)

y_train=y_train1
x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255

print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')




model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
# batch process

print(x_train.shape)

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=45)

# ------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )

# ------------------------------

if not os.path.exists("model1.h5"):

    model.fit_generator(train_generator, steps_per_epoch=60, epochs=73)
    model.sasnive("model1.h5")  # train for randomly selected one
else:
    model = load_model("model1.h5")  # load weights
from sklearn.metrics import confusion_matrix
yp=model.predict_classes(x_test,verbose=0)
cf=confusion_matrix(y_test,yp)
print(cf)