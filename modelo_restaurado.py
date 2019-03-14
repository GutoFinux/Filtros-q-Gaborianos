from keras.models import model_from_json
from sklearn.metrics import classification_report
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test = y_test.reshape(-1,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255


arquivo = open('cnn_mnist_adaDelta_5ep_ReluConv_qGaborActiveFunc.json')
configuracoes = arquivo.read()

modelo = model_from_json(configuracoes)

arquivo.close()

modelo.load_weights('cnn_mnist_adaDelta_5ep_ReluConv_qGaborActiveFunc.h5')

modelo.compile('Adadelta','categorical_crossentropy',['accuracy'])

previsoes = modelo.predict_classes(x_test)

print(classification_report(y_test,previsoes))
