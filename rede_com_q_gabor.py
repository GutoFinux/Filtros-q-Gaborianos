"""
@author: duccl
@version: 0.0.1
"""

import cv2
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from random import shuffle

# ------------------------- Leitura dos dados para treino ----------------
X_treinamento = []
y_treinamento = []
index_b = [10,20,29,58,81,83,86,89,93,96]
index_m = [116,131,155,158,161,176,185,197,217,233]

def coloca_no_vetor_treino(benigno,maligno):
    imagem_benigna = cv2.imread(benigno)
    imagem_maligna = cv2.imread(maligno)
    imagem_benigna = cv2.cvtColor(imagem_benigna,cv2.COLOR_BGR2GRAY)
    imagem_maligna = cv2.cvtColor(imagem_maligna,cv2.COLOR_BGR2GRAY)
    imagem_benigna = cv2.resize(imagem_benigna,(8,10))
    imagem_maligna = cv2.resize(imagem_maligna,(8,10))
    X_treinamento.append(imagem_maligna.reshape(-1))
    X_treinamento.append(imagem_benigna.reshape(-1))
    y_treinamento.append('maligno')
    y_treinamento.append('benigno')

for i in range(1,90):
    if i <= 9:
        arquivo_benigno = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\Benigno2\\benigno00'+str(i)+'.png'
        arquivo_maligno = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\Maligno2\\maligno00'+str(i)+'.png'
        coloca_no_vetor_treino(arquivo_benigno,arquivo_maligno)
    elif (i not in index_b):
        arquivo_benigno = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\Benigno2\\benigno0'+str(i)+'.png'
        arquivo_maligno = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\Maligno2\\maligno0'+str(i)+'.png'
        coloca_no_vetor_treino(arquivo_benigno,arquivo_maligno)

tuplas_treino = list(zip(X_treinamento,y_treinamento))
shuffle(tuplas_treino) #randomizando os dados
X_treinamento,y_treinamento = zip(*tuplas_treino) #extraindo do zip

X_treinamento = np.array(X_treinamento)/255 #deixando na escala entre 0 e 1
y_treinamento = np.array(y_treinamento).reshape(-1,1)

# -----------------------------------------------------------------------


# ------------------------- Leitura dos dados para teste ----------------

index_b = [10,20,29,58,81,83,86,89,93,96]
index_m = [116,131,155,158,161,176,185,197,217,233]
X_teste = []
y_teste = []

def coloca_no_vetor_teste(arquivo,rotulo):
    imagem = cv2.imread(arquivo)
    imagem_gray = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
    imagem_gray = cv2.resize(imagem_gray,(8,10))
    X_teste.append(imagem_gray.reshape(-1))
    y_teste.append(rotulo)

for num in index_b:
    arquivo = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\Benignot\\us'+str(num)+'.png'
    coloca_no_vetor_teste(arquivo,'benigno')

for num in index_m:
    arquivo = 'C:\\Users\\ebcar\\Desktop\\backup python\\US\\malignot\\us'+str(num)+'.png'
    coloca_no_vetor_teste(arquivo,'maligno')
    

tuplas_teste = list(zip(X_teste,y_teste))
shuffle(tuplas_teste)
X_teste,y_teste = zip(*tuplas_teste)
X_teste = np.array(X_teste)/255
y_teste = np.array(y_teste).reshape(-1,1)
# -----------------------------------------------------------------------

# ----- Vectorizando y -------
from sklearn.preprocessing import OneHotEncoder
oneHot = OneHotEncoder()
y_one_hot_treino = oneHot.fit_transform(y_treinamento).toarray()
y_one_hot_teste = oneHot.fit_transform(y_teste).toarray()

#------------------------------

neuronios_entrada = X_treinamento.shape[1]
neuronios_oculta_1 = 100
neuronios_oculta_2 = 100
neuronios_oculta_3 = 40
neuronios_oculta_4 = neuronios_oculta_1
neuronios_oculta_5 = neuronios_oculta_1
neuronios_saida = y_one_hot_treino.shape[1]

xph = tf.placeholder('float',shape = [None,neuronios_entrada])
yph = tf.placeholder('float',shape = [None,neuronios_saida])

# dicionários que representam respectivamente os pesos e os pesos do bias

W_5 = {'oculta_1':tf.Variable(tf.random_normal([neuronios_entrada,neuronios_oculta_1]),name='oculta_1'),
     'oculta_2':tf.Variable(tf.random_normal([neuronios_oculta_1,neuronios_oculta_2]),name='oculta_2'),
     'oculta_3':tf.Variable(tf.random_normal([neuronios_oculta_2,neuronios_oculta_3]),name='oculta_3'),
     'oculta_4':tf.Variable(tf.random_normal([neuronios_oculta_3,neuronios_oculta_4]),name='oculta_4'),
     'oculta_5':tf.Variable(tf.random_normal([neuronios_oculta_4,neuronios_oculta_5]),name='oculta_5'),
     'saida':tf.Variable(tf.random_normal([neuronios_oculta_5,neuronios_saida]),name='saida')}

b_5 = {'oculta_1_bias':tf.Variable(tf.random_normal([neuronios_oculta_1]),name='oculta_1_bias'),
     'oculta_2_bias':tf.Variable(tf.random_normal([neuronios_oculta_2]),name='oculta_2_bias'),
     'oculta_3_bias':tf.Variable(tf.random_normal([neuronios_oculta_3]),name='oculta_3_bias'),
     'oculta_4_bias':tf.Variable(tf.random_normal([neuronios_oculta_4]),name='oculta_4_bias'),
     'oculta_5_bias':tf.Variable(tf.random_normal([neuronios_oculta_5]),name='oculta_5_bias'),
     'saida_bias':tf.Variable(tf.random_normal([neuronios_saida]),name='saida_bias')}
# ----------------------------------------------------------------------------

pi = np.pi
e = 2.7183
def sinusoidal_calculo(f,X):
    X = tf.cast(X,dtype=tf.complex128)
    potencia = 2*X*f*1j
    resposta = pow(e,potencia)
    resposta = tf.cast(resposta,dtype=tf.float32)
    return resposta
def q_exponencial_calculo(X,q):
    potencia = 1/(1-q)
    primeiro = (1-q)*X*X
    base = 1+primeiro
    potencializacao = pow(base,potencia)
    resposta = 1/potencializacao
    return resposta

def q_gabor(X,alfa,q,f,angulo,k):
    sinusoidal = sinusoidal_calculo(f,X)
    q_exponencial = q_exponencial_calculo(X,q)
    potencia = angulo*1j
    potencializacao = pow(e,potencia).real
    g = k*potencializacao*sinusoidal*q_exponencial
    return g

def calculo_camadas_q_gabor(x,W,b):
    camada_oculta_1 = q_gabor(tf.add(tf.matmul(x,W['oculta_1']),b['oculta_1_bias']),0.3,0.1,0.5,0.1,0.4)
    camada_oculta_2 = q_gabor(tf.add(tf.matmul(camada_oculta_1,W['oculta_2']),b['oculta_2_bias']),0.3,0.1,0.5,0.1,0.3)
    camada_oculta_3 = q_gabor(tf.add(tf.matmul(camada_oculta_2,W['oculta_3']),b['oculta_3_bias']),0.3,0.7,0.2,0.45,0.1)
    camada_oculta_4 = q_gabor(tf.add(tf.matmul(camada_oculta_3,W['oculta_4']),b['oculta_4_bias']),0.3,0.4,0.6,0.34,0.6)
    camada_oculta_5 = q_gabor(tf.add(tf.matmul(camada_oculta_4,W['oculta_5']),b['oculta_5_bias']),0.3,0.14,0.5,0.8,0.21)
    camada_saida = tf.add(tf.matmul(camada_oculta_5,W['saida']),b['saida_bias'])
    return camada_saida

modelo = calculo_camadas_q_gabor(xph,W_5,b_5)
erro = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels = yph,logits = modelo))
otimizador = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(erro)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(500):
        
        batch_total = int(len(X_treinamento)/5)
        
        X_batches = np.array_split(X_treinamento,batch_total)
        y_batches = np.array_split(y_one_hot_treino,batch_total)
        
        for i in range(batch_total):
            X_batch,y_batch = X_batches[i],y_batches[i]
            _,custo = sess.run([otimizador,erro],{xph:X_batch,yph:y_batch})
            
        if epoca % 100 == 0:
            print('Época :' ,epoca,' || erro: ',custo)
            
    W_final,b_final = sess.run([W_5,b_5]) #salva os valores dos pesos para futuras previsoes

# ----------------------------- previsões -------------------------
modelo_final = calculo_camadas_q_gabor(xph,W_final,b_final)
previsoes_finais = tf.nn.sigmoid(modelo_final)

with tf.Session() as sess:
    for imagem in X_teste:
        print(oneHot.inverse_transform(sess.run(previsoes_finais,{xph:imagem.reshape(1,-1)}))[0][0])
    preds = oneHot.inverse_transform(sess.run(previsoes_finais,{xph:X_teste}))

# ------------------------------------------------------------------

# --------------------------- avaliação do modelo ------------------
print(classification_report(y_teste,preds))
# -----------------------------------------
