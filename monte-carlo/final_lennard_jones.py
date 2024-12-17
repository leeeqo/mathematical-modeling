#Задание 7
#Потенциал Леннарда-Джонса

import math
import numpy as np
import matplotlib.pyplot as plt
import random as random
from numba import njit
from random import *
from numpy import *
from math import *
from tqdm import *

#dtype = float


#Реализация потенциала с ускорением вычислений
@njit(parallel=True, fastmath=True, nopython=False)
def Lennard_Jones(N, X_Y, sigma, epsilon):
    V = np.zeros((N, N))#, dtype=dtype)
    for i in range(N):
        for j in range(i):
            if i != j:
                r = ( (X_Y[i,0] - X_Y[j,0])**2 + (X_Y[i,1] - X_Y[j,1])**2 )**0.5
                if r < 2 * sigma:  
                    V[i,j] = epsilon * ( (sigma/r)**12 - (sigma/r)**6 )
                if r >=  2 * sigma:
                    V[i,j] = 0
    return V


#Разбиение области на кольца и подсчет частиц в каждом кольце
def N_in_rings(X_Y, dr):
    drobs = list(arange(0, a/2 + a/1000, dr))
    rings = np.zeros((len(drobs) - 1, 2))
    N = []
    
    for r in range(len(drobs) - 1):
        rings[r,0], rings[r,1] = drobs[r], drobs[r+1]
    #print(rings)
    #print(drobs)
    
    for index, ring in enumerate(rings):
        n_for_ring = 0
        #print('for ', index, ' ring we have: ', ring)
        for p, particle in enumerate(X_Y):
            #print('for ', p, ' particle we have: ', particle)
            x, y = particle[0], particle[1]
            if (((x - a/2)**2 + (y - a/2)**2)**0.5 >= ring[0]) and (((x - a/2)**2 + (y - a/2)**2)**0.5 <= ring[1]):
                n_for_ring = n_for_ring + 1
                #print('N for this particle grows, no N = ', N)
        N.append(n_for_ring)
    return N

#Функция радиального распределения (для заданного разбиения dr)
def radial_spread(X_Y, dr):
    N_list = N_in_rings(X_Y, dr)
    V_list = []
    
    drobs = list(arange(0, a/2 + a/10000, dr))
    for i in range(len(drobs) - 1):
        #print(i)
        V_list.append( 4 * math.pi * (drobs[i + 1]**2 - drobs[i]**2) )
        #print(V_list[i])
        
    N = sum(N_list) #по факту это list'ы, но не суть
    V = sum(V_list)
    
    ro = N/V
    
    g_r = np.zeros((1, len(drobs) - 1))
    for i in range(len(drobs) - 1):
        g_r[0,i] = (N_list[i]/V_list[i]) * (1/ro)
    
    r_average_array = np.zeros((1, len(drobs) - 1))
    for i in range(len(drobs) - 1):
        r_average_array[0,i] = (drobs[i + 1] + drobs[i])/2
    #print(r_average_array)
    
    data = np.concatenate((g_r, r_average_array))
    
    return data


#Необходимые данные
N = 100 #количество молекул газа
sigma = 2.8e-10 #характерный размер молекулы газа
a = 20.0e-10 #размер области
T = 300 #пусть будет комнатная температура
k = 1.38e-23 #постоянная Больцмана
epsilon = 122 * k #глубина потенциальной ямы для аргона (из учебника)
M = 10000 #количество итераций метода Монте-Карло
dr = a/700 #delta_r для разбиения области на кольца


#Создаю и заполняю массива координат частиц
X_Y = np.zeros((N, 2))#, dtype=dtype) 
X_Y[:,0] = np.random.rand(N) * a
X_Y[:,1] = np.random.rand(N) * a


X_Y_init = X_Y

V = Lennard_Jones(N, X_Y, sigma, epsilon)


#Метод Монте-Карло
for i in range(M):
    X_Y_new = np.copy(X_Y)
    for j in range(N):        
        while(True):
            x = X_Y[j,0] + 0.01 * a * random.uniform(-1,1) #изменение координат частицы
            y = X_Y[j,1] + 0.01 * a * random.uniform(-1,1)
            if (x > 0) and (x < a) and (y > 0) and (y < a):
                break
            
        X_Y_new[j,0] = x
        X_Y_new[j,1] = y
        
        V_new = Lennard_Jones(N, X_Y_new, sigma, epsilon)
            
        #Если новый потенциал взаим-я меньше, то переходим в новое состояние
        if V_new.sum() < V.sum():
            V = V_new
            X_Y = X_Y_new
        #в противном случае существует только вероятность перехода, зависящая от температуры
        else:                   
            if random.random() < math.e**( (-abs(V_new.sum() - V.sum())) / (k*T) ):
                V = V_new
                X_Y = X_Y_new
    if i % 500 == 0:
        print(i) #итерация


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X_Y_init[:,0], X_Y_init[:,1], 'o', color = "blue", markersize=10)
ax2.plot(X_Y[:,0], X_Y[:,1], 'o', color = "blue",  markersize=10)


#Нахожу радиальные функции распределения
data_init = radial_spread(X_Y_init, dr)
data = radial_spread(X_Y, dr)


fig, (ax3, ax4) = plt.subplots(1, 2)
ax3.plot(data_init[1,:], data_init[0,:], color = "red")
ax4.plot(data[1,:], data[0,:], color = "red")





