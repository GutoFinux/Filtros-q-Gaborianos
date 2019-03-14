import math
import numpy as np
import tensorflow as tf

def Q_exponential(X,q):
    potencia = 1/(1-q)
    primeiro = (1-q)*X*X
    base = 1+primeiro
    potencializacao = pow(base,potencia)
    resposta = 1/potencializacao
    return resposta



def q_gabor(X,q=0.5,f=0.08,angle=0,k=1):
   """
       Non-extensive Statistics of Tsallis aplied in Gabor function.
       Combines a q-Exponential with Sinusoidal.

       F(x) = K*[e^(Angle*i)]*s(x)*q(alfa*x)

       s(x) = e^(2*x*f*i)

       q(x) = 1/((1+((1-q)*(X^2)))^(1/(1-q)))

       # Arguments
        x: Input tensor.
        alpha: float. Defaults to 0.3.
        q: float. Must not be equals 1.
        f: float. Defaults to 0.5.
	angle: float. Defaults to 0.1.
	k: float. Defaults to 0.1.

	# Returns
          A tensor.

	# Raises Error
	  If q is initialized as 1 (division by zero).

   """
   if q == 1:
      raise ValueError("The value of q must not be equals 1")
      return
   q_exponencial = Q_exponential(X,q)
   g = k*tf.sin((2*math.pi*f*X)+angle)*q_exponencial
   return g
