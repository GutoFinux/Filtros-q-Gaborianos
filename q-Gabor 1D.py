# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:41:49 2017

@author: Guto
"""
#pega a gabor, substitui o envelope pela exp da q-gaussiana

q = 0.1 #abertura
f = 1 #Frequencia
k = 3 #amplitude do envelope na função gabor
O = 0 #ângulo

X = arange(-12,12,0.005)

w = 1/((1+(1-q)*((X**2)))**(1/(1-q))) # 1/Eq(X²) <- função q-exp trazida da q-Gaussiana como envelope
s = exp((2*math.pi*f*X)*1j) #onda
g = k*exp(O*1j)*w*s #q-Gabor

title("q-Gabor")
plot(X,g)
show()

title("Envelope")
plot(X,w)
show()

title("Onda")
plot(X,s)
show()














