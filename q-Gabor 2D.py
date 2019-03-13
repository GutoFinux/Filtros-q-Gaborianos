# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:03:40 2017

@author: Guto
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numpy import *

x = y = np.arange(-40,40,0.5)
X, Y = np.meshgrid(x,y)

q = 0.1 #abertura
k = 1 #amplitude
u = v = 0.08 #Frequencia do filtro
P = 0 #Fase do filtro

Xo = Yo = 0 #(Xo,Yo) = pico do envelope
a = b = 0.2 #largura do envelope

w = k*(1/((1+(1-q)*((a**2*(X-Xo)**2 + b**2*(Y-Yo)**2)))**(1/(1-q)))) #Envelope
s = exp((2*math.pi*(u*X+v*Y)+P)*1j) #onda
g = w*s #q-Gabor

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title("q-Gabor 2D")
ax.plot_wireframe(X, Y, g, rstride=10, cstride=10)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
title("Envelope")
ax2.plot_wireframe(X, Y, w, rstride=10, cstride=10)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
title("Onda")
ax3.plot_wireframe(X, Y, s, rstride=10, cstride=10)





