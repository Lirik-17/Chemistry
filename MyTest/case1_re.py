import numpy as np
from scipy.integrate import solve_ivp

# Параметры задачи
datasize = 10
n_exp_train = 6
n_exp_test = 2
n_exp = n_exp_train + n_exp_test
ns = 5
nr = 4
k = np.array([0.1, 0.2, 0.13, 0.3], dtype=np.float32)
tstep = 1.0  # временной шаг
noise = 0.01  # уровень шума
lb = 0.01  # нижняя граница для max_min

# Начальные условия
u0_list = np.random.rand(n_exp, ns).astype(np.float32)
u0_list[:, :2] += 1.0  # Увеличение первых двух компонентов
u0_list[:, 2:] = 0.0   # Остальные компоненты равны нулю

# Временные параметры
tspan = np.array([0.0, datasize * tstep], dtype=np.float32)
tsteps = np.linspace(tspan[0], tspan[1], datasize, dtype=np.float32)

# Создание массивов для результатов
ode_data_list = np.zeros((n_exp, ns, datasize), dtype=np.float32)
std_list = []

# Функция max_min
def max_min(ode_data):
    return np.max(ode_data, axis=1) - np.min(ode_data, axis=1) + lb

# Функция ОДУ
def true_ode_func(t, y, k):
    dydt = np.zeros_like(y)
    dydt[0] = -2 * k[0] * y[0]**2 - k[1] * y[0]
    dydt[1] = k[0] * y[0]**2 - k[3] * y[1] * y[3]
    dydt[2] = k[1] * y[0] - k[2] * y[2]
    dydt[3] = k[2] * y[2] - k[3] * y[1] * y[3]
    dydt[4] = k[3] * y[1] * y[3]
    return dydt

# Решение ОДУ и добавление шумов
for i in range(n_exp):
    u0 = u0_list[i, :]  # Начальные условия для i-го эксперимента

    # Решение ОДУ с помощью solve_ivp
    prob_solution = solve_ivp(
        true_ode_func, 
        tspan, 
        u0, 
        t_eval=tsteps, 
        args=(k,), 
        method='RK45'  # Аналог Tsit5
    )

    ode_data = prob_solution.y.T  # Решение на временных шагах
    ode_data += np.random.randn(*ode_data.shape) * ode_data * noise  # Добавление шума
    ode_data_list[i, :, :] = ode_data.T  # Сохранение результатов
    std_list.append(max_min(ode_data.T))  # Сохранение статистики

# Вычисление наибольшей разницы
y_std = np.max(np.hstack(std_list), axis=1)  # Максимумы по всем экспериментам

# Печать результатов
print("ODE Data Shape:", ode_data_list.shape)
print("Y_STD Shape:", y_std.shape)
