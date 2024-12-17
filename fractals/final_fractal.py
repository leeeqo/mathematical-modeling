#Задание 5 - Фракталы
#Вариант 10


from random import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from numpy import *
import numpy as np

from math import *
from numpy import linalg as LA
from tqdm import *


#     ДЛЯ ВИЗУАЛИЗАЦИИ ПОСТРОЕНИЯ ФРАКТАЛА:  TASK = 1
#     ДЛЯ ПОСТРОЕНИЯ ГРАФИКА ФРАКТАЛЬНОЙ РАЗМЕРНОСТИ ОТ ВЕРОЯТНОСТИ ПРИЛИПАНИЯ: TASK = 2
TASK = 1



#Создаю класс объектов - координат "областей", к которым могут прилипнуть свободные частицы
class fractal:
    def __init__(self, line):
        self.fractal = line
        
    def addition(self, particle):
        self.fractal = np.concatenate((self.fractal, particle), axis = 0)
        
    def show(self):
        print(self.fractal)
        
    def length(self):
        return np.shape(self.fractal)[0]
    
    def massiv(self):
        return self.fractal
    
    
#Для рандомного изменения одной координаты, если нужно - добавляю вес на вероятность изменения в ту или иную сторону по осям
def rand(weight_up):
    steps = [-step, step]
    if weight_up == 0.5:
        return choice(steps)
    else:
        return choices(steps, weights = [1 - weight_up, weight_up])[0]


#Обновляю координату с помощью ранее введенной rand
def new_coord(coord, weight_up = 0.5):
    return (coord + rand(weight_up)) % length


#Рассчитывает вероятность прилипания, по умолчанию p = 1
def prilip(x, y, p = 1):
    #global line
    for i in range(Fractal.length()):
        if abs(x - Fractal.massiv()[i,0]) <= step and abs(y - Fractal.massiv()[i,1]) <= step:
            return choices([True, False], weights = [p, 1 - p])[0]
        
        
#Прилипание какой-то одной частички           
def action(X_Y, Fractal, p = 1, times = 1):
    for i in range(times):
        for index, pair in enumerate(X_Y):
            x, y = pair[0], pair[1]
            new_x, new_y = new_coord(x), new_coord(y, 0.5)
            while (new_x == x1 or new_x == x2) or (new_y == y1 or new_y == y2):
                new_x, new_y = new_coord(x), new_coord(y, 0.5)
            #index = X_Y.index(j)
            pair = np.array([new_x, new_y])
            X_Y[index] = pair
            #print(pair)
            if prilip(new_x, new_y, p) == True:
                particle = np.array([[new_x, new_y]])
                Fractal.addition(particle)
                X_Y[index] = np.array([np.nan, np.nan])
                #X_Y = np.concatenate((X_Y[:index, :], X_Y[(index + 1):, :]))
                #X_Y = np.delete(j)
                #X_Y[j,0], X_Y[j,1] = 0, 0
                #num_particles = num_particles - 1
                #print(pair)
                ax.scatter(x=new_x, y=new_y, color = "darkviolet", marker = "s", s=35)


def process(dt):
    action(X_Y, Fractal, p) 
    X_Y_plot = X_Y #ОБНОВЛЕНИЕ X Y 
    mat.set_data(X_Y_plot[:,0], X_Y_plot[:,1]) #для анимации
    
    
#Параметры картинки и шаг
length = 4
x1, x2 = 0, length
y1, y2 = 0, length
step = 0.1 
    

#Количество частиц и вероятность прилипания
num_particles = 500
p = 1 


#Создаю линию, вдоль которой будет прилипание
len_line = len(arange(x1 + step, x2 - step, step))
line = np.zeros((len_line, 2))
for i in range(len_line):
    line[i,0] , line[i,1] = arange(x1 + step, x2 - step, step)[i] , length/2


#Работа с картинкой
fig = plt.figure(figsize = (length + 2, length + 2))  
ax = fig.add_subplot() #РИСУЮ          
plt.title ("Фрактал")
plt.xlim (x1, x2) #пределы значений на осях
plt.xticks(np.linspace(x1, x2, x2 - x1 + 1)) 
plt.yticks(np.linspace(y1, y2, x2 - x1 + 1))
    
#Рисую линию прилипания на графике
mat, = ax.plot(line[:,0], line[:,1], '_', color = "black", markersize = 27)

#Инициализация
Fractal = fractal(line)

#Заполняю пространство частичками, задавая рандомные координаты и помещая их в столбцы X и Y
X_Y = np.zeros((num_particles, 2))
for i in range(num_particles):
    X_Y[i,0], X_Y[i,1] = uniform(0, length), uniform(0, length)
    
#Наношу частички на график
X_Y_plot = X_Y
mat, = ax.plot(X_Y_plot[:,0], X_Y_plot[:,1], 'o', color = "lime", markersize = 5)


if TASK == 1:
    
    time = 10
    dt = 0.005   
    temps = int(time//dt)
    Time = tuple([time / temps] * temps)
    anim = FuncAnimation(fig, process, frames = Time, repeat=False, interval = 1)
    plt.show()

if TASK == 2:
    
    #Подсчет квадратиков, в которые попали частички фрактала
    #  c---d
    #  |   |
    #  a---b
    def num_of_fractaled_squares(Fractal, d):
        drobs = list(arange(0, length, d))  # 0, d, 2d, 3d,..., length - d, length
        squares = np.zeros((len(drobs) ** 2, 4))
        N = 0
        for x in range(len(drobs)):
            for y in range(len(drobs)):
                squares[x * len(drobs) + y,0], squares[x * len(drobs) + y,1] = drobs[x], drobs[x] + d
                #print(squares)
                squares[x * len(drobs) + y,2], squares[x * len(drobs) + y,3] = drobs[y], drobs[y] + d
                #print(squares)
        for index, square in enumerate(squares):
            #print(index)
            for particle_of_Fractal in Fractal.massiv():
                x, y = particle_of_Fractal[0], particle_of_Fractal[1]
                #print(x, y)
                #a, b, c, d = square[0], square[1], square[2], square[3]
                #print(x, y)
                if (x >= square[0]) and (x <= square[1]) and (y >= square[2]) and (y <= square[3]):
                    N = N + 1
                    break    
        print(N)
        return N
                    
                    
    #Вычисление фрактальной размерности для той или иной p прилипания
    def dimension(p):
        print('Вычисления для вероятности прилипания p = ', p)
        Fractal = fractal(line)
        X_Y = np.zeros((num_particles, 2))
        for i in range(num_particles):
            X_Y[i,0], X_Y[i,1] = uniform(0, length), uniform(0, length)
        while ~np.isnan(X_Y).all():
            action(X_Y, Fractal, p, 5)
        D = [length/(5*i) for i in range(1,25)]
        N = []
        for d in D:
            N.append(num_of_fractaled_squares(Fractal, d))
        D = -vstack([np.log(array(D)), np.ones(len(D))]).T
        N = np.log(array(N))
        m, c = LA.lstsq(D, N, rcond=None)[0] #МНК-ов находим m - фрактальную размерность
        return m

    
    probabilities = np.arange(0.6, 1 + 0.01, 0.1)
    dimensions = []
    for p in probabilities:
        dimensions.append( dimension(p) )
        print(dimensions)
        print(probabilities)
        
    import pylab    
        
    pylab.figure(2)
    pylab.plot (probabilities, dimensions)
    pylab.legend()


