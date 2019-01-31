# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:03:40 2017

@author: Guto
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numpy import *
from PIL import Image

j = 3 #janelamento 3x3
D = (j-1)/2 #Fator D apenas para arrumar vetores que não possuem índices negativos

q = 2 #abertura 2
k = 1 #amplitude 1
u = v = 0.04 #Frequencia do filtro 0.04
P = 0 #Fase do filtro 0

Xo = Yo = 0 #(Xo,Yo) = pico do envelope 0
a = b = 0.075 #largura do envelope 0.075
x = y = 0 #ponto central = (0,0)

#----------Preparando o filtro q-Gabor jxj com variação de 1 unidade------------------------------------
fil = []

for y in range(0,j):
    for x in range(0,j):
        w = k*(1/((1+(1-q)*((a**2*((x-D)-Xo)**2 + b**2*((y-D)-Yo)**2)))**(1/(1-q)))) #Envelope
        s = exp((2*math.pi*(u*(x-D)+v*(y-D))+P)*1j) #Onda
        g = w*s+k #q-Gabor (somado a altura para não haver pontos abaixo de 0)
        fil.append(g.real) #adiciona a parte real no vetor do filtro

for x in range(0,j**2):
    fil[x] = round(fil[x]/(2*k),2) #Normalizando para que todos os pontos fiquem entre 0 e 1

#-----------Preparando imagem---------------------------------------------------------------------------
#img = Image.open('pluto.png')
#img = Image.open('seal.png')
#img = Image.open('textures.png')
#img = Image.open('lion.png')
img = Image.open('lenna.png')
plt.imshow(img)
print "\n\nImagem Original"
plt.show()
#-------------------------------------------------------------------------------------------------------

#apenas printando o filtro

print "\n\nFiltro"
countLine = 0
line = []
for y in range(0,j):
    for x in range(0,j):
        line.append(fil[countLine])
        countLine = countLine + 1
    print line
    line = []


#-----------Convolução----------------------------------------------------------------------------------
width, height = img.size #pegando altura e largura da imagem para andar a janela

arr = array(img) #transformando a imagem em um array

numberPixel = 0;
for y in range(D,height-D): #linhas da imagem
    for x in range(D,width-D): #colunas da imagem
        countFil = 0
        for y2 in range(0,j): #linhas da janela
            for x2 in range(0,j): #colunas da janela
                numberPixel = numberPixel + arr[x+x2-D][y+y2-D]*fil[countFil] #multiplica o pixel da imagem pelo pixel coreespondente da janela e soma na variavel
                countFil = countFil+1
        numberPixel = numberPixel/(j**2) #tira a média do pixel central
        arr[x][y] = numberPixel #armazena o resultado na posição do pixel na imagem

img2 = Image.fromarray(arr)
plt.imshow(img2, cmap='gray')
print("\n\nConvolução com q-Gabor:")
plt.show()
#-------------------------------------------------------------------------------------------------------




