#Задание 8
#Молекулярная динамика

import math
import numpy as np
import matplotlib.pyplot as plt
import random as random
from numba import njit
from random import *
from numpy import *
from math import *
from tqdm import *

from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.animation import FuncAnimation


@njit(parallel=True, fastmath=True, nopython=False)
def Lennard_Jones(N, X_Y, sigma, epsilon):
    V = np.zeros((N, N))#, dtype=dtype)
    for i in range(N):
        for j in range(i):
            if i != j:
                r = ( (X_Y[i,0] - X_Y[j,0])**2 + (X_Y[i,1] - X_Y[j,1])**2 )**0.5
                if r < 2 * sigma:  
                    V[i,j] = epsilon * ( (sigma/r)**12 - (sigma/r)**6 )
                if r >= 2 * sigma:
                    V[i,j] = 0
    return V


@njit(parallel=True, fastmath=True, nopython=False)
def Buckingham_Potential(N, X_Y, A, B, L1, L2):
    V = np.zeros((N, N))#, dtype=dtype)
    for i in range(N):
        for j in range(i):
            if i != j:
                r = ( (X_Y[i,0] - X_Y[j,0])**2 + (X_Y[i,1] - X_Y[j,1])**2 )**0.5
                V[i,j] = A * math.e**( (-B)*r ) - L1/(r**6) - L2/(r**8)
                #V[j,i] = V[i,j]
                
    return V

    
@njit(parallel=True, fastmath=True, nopython=False)
def Forces_Lennard(N, X_Y, sigma, epsilon):
    F = np.zeros((2, N, N))
    for i in range(N):
        for j in range(i):
            if i != j:
                dx = X_Y[i,0] - X_Y[j,0]
                dy = X_Y[i,1] - X_Y[j,1]
                r = ( dx**2 + dy**2 )**0.5
                x = r * ( 1/( ( 1 + (dy/dx)**2 )**0.5 ) )
                y = r * ( ( dy/dx )/( ( 1 + (dy/dx)**2 )**0.5 ) )
                if r < 7 * sigma:
                    F[0, i, j] = - ( epsilon * ( (-12)*(sigma**12)/(r**13) + 6*(sigma**6)/(r**7) ) )*(x/r)
                    F[1, i, j] = - ( epsilon * ( (-12)*(sigma**12)/(r**13) + 6*(sigma**6)/(r**7) ) )*(y/r)
                if r >= 7 * sigma:
                    F[0, i, j] = 0
                    F[1, i, j] = 0
                
                #F[j,i,0], F[j,i,1] = -F[i,j,0], -F[i,j,1]
    return F


@njit(parallel=True, fastmath=True, nopython=False)
def Forces_Buckingham(N, X_Y, A, B, L1, L2):
    F = np.zeros((2, N, N))
    for i in range(N):
        for j in range(i):
            if i != j:
                dx = X_Y[i,0] - X_Y[j,0]
                dy = X_Y[i,1] - X_Y[j,1]
                r = ( dx**2 + dy**2 )**0.5
                x = r * ( 1/( ( 1 + (dy/dx)**2 )**0.5 ) )
                y = r * ( ( dy/dx )/( ( 1 + (dy/dx)**2 )**0.5 ) )
                F[0,i,j] = - ( (-B)*A*math.e**( (-B)* r ) + 6*L1/(r**7) + 8*L2/(r**9) )*(x/r)
                F[1,i,j] = - ( (-B)*A*math.e**( (-B)* r ) + 6*L1/(r**7) + 8*L2/(r**9) )*(y/r)
                #F[j,i,0], F[j,i,1] = -F[i,j,0], -F[i,j,1]
    return F


@njit(parallel=True, fastmath=True, nopython=False)
def F_x_y(N, Forces):
    F = np.zeros((N,2))
    for n in range(N):
        F[n,0] = Forces[0,n,:].sum() + Forces[0,:,n].sum()
        F[n,1] = Forces[1,n,:].sum() + Forces[1,:,n].sum()
        
    return F


@njit(parallel=True, fastmath=True, nopython=False)
def Accelerations(N, X_Y, F):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            if i!= j:
                A[i,j,0], A[i,j,1] = F[i,j,0]/m, F[i,j,1]/m
                #A[j,i,0], A[j,i,1] = -A[i,j,0], -A[i,j,1]
    return A


@njit(parallel=True, fastmath=True, nopython=False)
def Velocities(N, X_Y_new, X_Y_prev, dt):
    V = np.zeros((N,2))
    for n in range(N):
        V[n,0] = (X_Y_new[n,0] - X_Y_prev[n,0])/(2*dt)
        V[n,1] = (X_Y_new[n,1] - X_Y_prev[n,1])/(2*dt)
        
    return V


@njit(parallel=True, fastmath=True, nopython=False)        
def Period(coord):
    if (coord < 0):
        coord = coord + a
    elif (coord > a):
        coord = coord - a
    
    return coord
                   
                
#Для Леннарда-Джонса
N = 20 #количество молекул газа #
sigma = 2.8e-10 #характерный размер молекулы газа
a = 10.0e-10 #размер области
T = 40 #пусть будет комнатная температура
k = 1.38e-23 #постоянная Больцмана
epsilon = 122 * k #глубина потенциальной ямы для аргона (из учебника)
m = 2.6e-22 #масса двух молекул аргона 1.33e-22 г


#Для потенциала Букингема
#N = 12
#a = 9.0e-10
#m = 2.6e-26 #масса молекулы O2
A = 1388.77e-19 #коэффициенты потенциала Букингема для O2
B = 2.76e-10
L1 = 175e-19*((1.0e-10)**6)
L2 = 0


#Создаю и заполняю массивы координат
X_Y = np.zeros((N, 2))
X_Y[:,0] = np.random.rand(N) * a
X_Y[:,1] = np.random.rand(N) * a


#Начальные условия (для метода Верле)
X_Y_prev = np.zeros((N, 2))
for i in range(N):
    X_Y_prev[i,0] = X_Y[i,0] + 0.01 * a * random.uniform(-1,1)
    X_Y_prev[i,1] = X_Y[i,1] + 0.01 * a * random.uniform(-1,1)


#Работа с картинкой
fig = plt.figure(figsize = (5, 5))  
x1, x2 = 0, 5
y1, y2 = 0, 5
ax = fig.add_subplot() #РИСУЮ          
plt.title ("Модель")
plt.xlim (x1, x2)
plt.ylim (y1, y2) #пределы значений на осях
plt.xticks(np.linspace(x1, x2, x2 - x1 + 1)) 
plt.yticks(np.linspace(y1, y2, y2 - y1 + 1))


#Обновление координат
def animate(f):
    global X_Y
    global X_Y_prev
    global Ti
    
    for i in range(N):
        print(i)
        
        X_Y_for_prev = X_Y
        
        #Potentials = Lennard_Jones(N, X_Y, sigma, epsilon)
        Forces = Forces_Lennard(N, X_Y, sigma, epsilon)
        F = F_x_y(N, Forces)
        
        #Potentials = Buckingham_Potential(N, X_Y, A, B, L1, L2)
        #Forces = Forces_Buckingham(N, X_Y, A, B, L1, L2)
        #F = F_x_y(N, Forces)
        
        X_Y[i,0] = Period( 2 * X_Y[i,0] - X_Y_prev[i,0] + (F[i,0]/m)*(f**2) )
        X_Y[i,1] = Period( 2 * X_Y[i,1] - X_Y_prev[i,1] + (F[i,1]/m)*(f**2) )
    
        X_Y_prev = X_Y_for_prev
        
        ax.plot(5/2, 5/2, 's', color = 'white', markersize = 280)
        ax.plot(5*X_Y[:,0]/a, 5*X_Y[:,1]/a, 'o', color = "blue", markersize = 15)
        


#Анимация
time = 2.0e-15
Time = arange(time/30, 5*time + time/1000, time/30)
anim = FuncAnimation(fig, animate, frames=Time , repeat=False, interval = 1)
plt.show()
anim.save("final_MD_LD_2.gif", writer="imagemagick")








