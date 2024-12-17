#Спецкурс, задание 6, вариант 5
#Реализация генератора псевдослучайных чисел
#на базе последовательности Фибоначчи

x_prev = 1
x_cur = 0

m = int(input('Введите период метода m'))

num_zeros = 0
num_ones = 0
num_of_zeros_and_ones = 0


for i in range(1, m):
    x = x_cur
    x_cur = (x_cur + x_prev) % m
    x_prev = x
    
    x_dvoich = bin(x_cur)
    for j in range(len(x_dvoich) - 2):
        if x_dvoich[j + 2] == '0':
            num_zeros = num_zeros + 1
        if x_dvoich[j + 2] == '1':
            num_ones = num_ones + 1
        num_of_zeros_and_ones = num_of_zeros_and_ones + 1
            
    #print(num_ones, num_zeros)
    #print(num_of_zeros_and_ones)
    
chastota_zeros = num_zeros / num_of_zeros_and_ones
chastota_ones = num_ones / num_of_zeros_and_ones

print('Частота появления нулей:', chastota_zeros)
print('Частота появления единиц:', chastota_ones)  
