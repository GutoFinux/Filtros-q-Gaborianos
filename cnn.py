from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,BatchNormalization
from q_gabor_calculo import *
from keras.callbacks import TensorBoard

def cnn_model(X_train,X_teste,y_train,y_teste):
   model = Sequential()
   
   model.add(Conv2D(filters = 32,kernel_size = [3,3],
                    activation='relu',padding = 'same',kernel_initializer = 'glorot_normal'))
   model.add(MaxPool2D(pool_size = (3,3),padding = 'same'))
   model.add(Conv2D(filters = 32,kernel_size = [3,3],
                    activation='relu',padding = 'same',kernel_initializer = 'glorot_normal'))
   model.add(MaxPool2D(pool_size = (2,2),padding = 'same'))
   model.add(Flatten())
   model.add(Dropout(0.15))
   model.add(Dense(units = 200,activation = q_gabor))
   model.add(Dense(units = 150,activation = q_gabor))
   model.add(Dense(units = 10,activation = 'softmax'))
   model.compile('Adadelta','categorical_crossentropy',['accuracy'])
   historico = model.fit(X_train,y_train,batch_size = 128,epochs = 5,
             validation_data=(X_teste,y_teste),verbose = 2)
   classes = model.predict_classes(X_teste)
   
   return classes,model,historico
