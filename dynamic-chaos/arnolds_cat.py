#Реализация отображения "Кот Арнольда"

import numpy as np
import matplotlib.pyplot as plt

#библиотека PyTorch
import torch
from torchvision.io import read_image #для извлечения изображения


#Извлечение изображения с помощью PyTorch
path = ''
image = read_image(path)

#Операции над тензором image
indices = torch.tensor([0])
image = torch.index_select(image, 0, indices) #беру только "часть" от image
img = image.squeeze() #понижаю размерность до минимальной
plt.imshow(img, cmap="gray") 
plt.show() #вывожу изображение, которое буду отображать

#Размеры изображения в пикселях
height = img.size()[0]
width = img.size()[1]

#Создание массивов координат X и Y
X = np.full((height, 1), 0)
for i in range(width - 1):
    for_X = np.full((height, 1), i + 1) #столбец иксов
    X = np.concatenate((X, for_X), 1)
Y = np.full((1, width), height - 1)
for i in range(height - 1):
    for_Y = np.full((1, width), height - (i + 2)) #столбец иксов
    Y = np.concatenate((Y, for_Y), 0)

#Инициализация класса cat
class cat:
    brightness = 0
    x = 0
    y = 0
    
    def __init__(self, brightness, x, y):
        self.brightness = brightness
        self.x = x
        self.y = y

#Преобразование
X_ = X + Y
Y_ = X + 2 * Y

#Сдвиг координат по иксу
for i in range(height):
    for j in range(width):
        if X_[i, j] > width - 1:
            X_[i, j] = X_[i, j] - width # - (width - 1)

#Сдвиг по игреку
for i in range(height):
    for j in range(width):
        if Y_[i, j] > height - 1 and Y_[i, j] <= (height - 1) * 2 :
            Y_[i, j] = Y_[i, j] - height # - (height - 1)
        if Y_[i, j] > (height - 1) * 2 :
            Y_[i, j] = Y_[i, j] - height * 2 # - (height - 1) * 2
          
#Создание массива объектов класса cat
array_with_cats = np.full((height, width), cat(0, 0, 0))
for i in range(height):
    for j in range(width):
        array_with_cats[i, j] = cat(img[i, j].item(), X_[i, j], Y_[i, j])
        
#Отображение
bedniy_kotik = np.full((height, width), 255)
for i in range(height):
    for j in range(width):
        bedniy_kotik[X_[i, j], Y_[i, j]] = array_with_cats[i, j].brightness
        
plt.imshow(bedniy_kotik, cmap="gray")
plt.show()