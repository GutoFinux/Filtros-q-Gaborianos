from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from cnn import *
import matplotlib.pyplot as plt


ohe = OneHotEncoder(10)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255

y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.fit_transform(y_test).toarray()

previsao,classificador,historico = cnn_model(x_train,x_test,y_train,y_test)

plt.plot(historico.history['acc'],label='Treino')
plt.plot(historico.history['val_acc'],label='Teste')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


plt.plot(historico.history['loss'],label='Treino')
plt.plot(historico.history['val_loss'],label='Teste')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
