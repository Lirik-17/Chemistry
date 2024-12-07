import math
import random

random.seed(1)

# Константы
N = 6                                                               #веществ
M = 5                                                               #реакций
k0 = [100000000, 90000000, 85000000, 110000000, 95000000]
Ea= [70000, 60000, 55000, 80000, 75000]
R = 8.314
T = 250 # + random.randint(0,100)
MATR = [
    [-2, -1, 1, 1, 0, 0],
    [0, -2, 1, 0, 2, 0],
    [0, -1, -2, 1, 0, 1],
    [1, 0, 0, -2, 1, 0],
    [1, 0, 0, 0, -2, 1]
]
H = 0.01

# Вычисление вектора коэффициентов скоростей химических реакций
k = [0 for _ in range(M)]
for i, k0_val in enumerate(k0):
    k[i] = k0_val * math.exp(-Ea[i] / (R * T)) * (T**0.5)

# Переменные, будут заводиться отдельно для каждого цикла
vr_time = 9 + random.randint(0, 3)
C = [1/6 for _ in range(N)]
F = [0 for _ in range(N)]
W = [0 for _ in range(M)]

print(C)
print(k)



