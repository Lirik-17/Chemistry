'''
Код не верно работает! Графики не должны быть прямыми
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Константы
N = 6
M = 5
A = np.array([100000000, 90000000, 85000000, 110000000, 95000000])  # предэкспоненциальный множитель
E = np.array([70000, 60000, 55000, 80000, 75000])  # энергия активации (Дж/моль)
R = 8.314  # универсальная газовая постоянная
T = 300  # температура (К)

# Стехиометрическая матрица
MATR = np.array([ 
    [-2, -1,  1,  1,  0,  0],
    [ 0, -2,  1,  0,  2,  0],
    [ 0, -1, -2,  1,  0,  1],
    [ 1,  0,  0, -2,  1,  0],
    [ 1,  0,  0,  0, -2,  1]
])

# МАтрица порядков реакций
MATR_PLUS = np.where(MATR < 0, -MATR, 0)

# Начальные концентрации
C0 = np.full(N, 1 / N)

# Вычисление констант скоростей
k = A * np.exp(-E / (R * T))

# Дифференциальные уравнения
def dCdt(t, C):
    r = np.array([k[j] * np.prod(C**MATR_PLUS.T[:, j]) for j in range(len(A))])  # Скорость для каждой реакции
    dC = MATR.T @ r  # Изменение концентраций
    return dC

# Численное решение
time_span = (0, 100)  # Время реакции (с)
time_points = np.linspace(0, 100, 100000)  # Точки для вывода
solution = solve_ivp(dCdt, time_span, C0, t_eval=time_points)

# Графики концентраций
plt.figure(figsize=(10, 6))
for i, C in enumerate(solution.y):
    plt.plot(solution.t, C, label=f'Вещество {i+1}')
plt.xlabel('Время, с')
plt.ylabel('Концентрация, мол/л')
plt.title('Изменение концентраций веществ во времени')
plt.legend()
plt.grid()
plt.show()
