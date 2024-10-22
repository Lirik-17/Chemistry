import numpy as np
import random
import os
import pickle

import scipy.integrate as spi                       #
import matplotlib.pyplot as plt                     #
from scipy.optimize import minimize                 #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
 
# Устанавливаем начальное значение для генератора случайных чисел
random.seed(1234)
np.random.seed(1234)

###################################
# Параметры программы

is_restart = True  # Флаг для перезапуска
n_epoch = 10000  # Количество эпох для обучения
n_plot = 50  # Частота вывода графиков
datasize = 50  # Размер данных
tstep = 1  # Шаг времени

n_exp_train = 20  # Количество экспериментальных данных для обучения
n_exp_test = 10  # Количество экспериментальных данных для тестирования
n_exp = n_exp_train + n_exp_test  # Общее количество экспериментальных данных

noise = 0.05  # Уровень шума

ns = 6  # Количество состояний
nr = 3  # Количество реакций

# Пределы для параметров оптимизации
lb = 1e-6
ub = 10.0

####################################

# Определяем функцию, описывающую систему ОДУ
def trueODEfunc(t, y, k):
    r1 = k[0] * y[0] * y[1]  # реакция 1
    r2 = k[1] * y[2] * y[1]  # реакция 2
    r3 = k[2] * y[3] * y[1]  # реакция 3
    dydt = [-r1, -r1 - r2 - r3, r1 - r2, r2 - r3, r3, r1 + r2 + r3, 0]
    return dydt

# Логарифмы предэкспоненциальных факторов и энергии активации
logA = np.array([18.60, 19.13, 7.93])
Ea = np.array([14.54, 14.42, 6.47])  # в ккал/моль

# Функция для расчета констант скорости реакций по уравнению Аррениуса
def Arrhenius(logA, Ea, T):
    R = 1.98720425864083e-3  # Газовая постоянная в ккал/(моль*K)
    k = np.exp(logA) * np.exp(-Ea / (R * T))
    return k

# Генерация начальных условий для экспериментов
u0_list = np.random.rand(n_exp, ns + 1).astype(np.float32)
u0_list[:, 0:2] = u0_list[:, 0:2] * 2.0 + 0.2  # Диапазон для TG и ROH
u0_list[:, 2:ns] = 0.0  # Начальные концентрации остальных компонентов
u0_list[:, ns] = u0_list[:, ns] * 20.0 + 323.0  # Температура [K]

# Задание временного интервала для моделирования
tspan = [0.0, datasize * tstep]
tsteps = np.linspace(tspan[0], tspan[1], datasize)

# Создание массива для хранения результатов моделирования
ode_data_list = np.zeros((n_exp, ns, datasize), dtype=np.float32)
yscale_list = []

# Функция для нормализации данных
def max_min(ode_data):
    return np.max(ode_data, axis=1) - np.min(ode_data, axis=1) + lb

# Решение задачи Коши для каждого эксперимента
for i in range(n_exp):
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[-1])  # Расчет констант скорости
    ode_data = spi.solve_ivp(lambda t, y: trueODEfunc(t, y, k), tspan, u0, t_eval=tsteps).y[:-1, :]
    ode_data += np.random.randn(*ode_data.shape) * ode_data * noise  # Добавление шума к данным
    ode_data_list[i, :, :] = ode_data  # Сохранение данных
    yscale_list.append(max_min(ode_data))  # Нормализация данных

# Максимальные значения масштабов данных по всем экспериментам
yscale = np.max(np.hstack(yscale_list), axis=1)

# Инициализация параметров для оптимизации
np.random.seed(1234)
np_params = np.random.randn(nr * (ns + 2) + 1).astype(np.float32) * 0.1
np_params[0:nr] += 0.8
np_params[nr * (ns + 1):nr * (ns + 2)] += 0.8
np_params[-1] = 0.1

# Определяем функцию для преобразования параметров оптимизации в веса
def p2vec(p):
    slope = p[nr * (ns + 2)] * 100
    w_b = p[0:nr] * slope
    w_out = p[nr:nr * (ns + 1)].reshape((nr, ns))
    w_in_Ea = np.abs(p[nr * (ns + 1):nr * (ns + 2)] * slope)
    w_in = np.clip(-w_out.T, 0, 4)
    w_in = np.vstack((w_in, w_in_Ea))
    return w_in, w_b, w_out

# Отображение параметров
def display_p(p):
    w_in, w_b, w_out = p2vec(p)
    print("species (column) reaction (row)")
    print("w_in | w_b")
    print(np.round(np.hstack([w_in, w_b[:, None]]), 3))
    print("\nw_out")
    print(np.round(w_out.T, 3))
    print("\n")

display_p(np_params)

inv_R = -1 / 1.98720425864083e-3

# Определяем нейронную сеть в виде функции
class NeuralODE(nn.Module):
    def __init__(self, w_in, w_b, w_out):
        super(NeuralODE, self).__init__()
        self.w_in = w_in
        self.w_b = w_b
        self.w_out = w_out

    def forward(self, t, u):
        logX = torch.log(torch.clamp(u[:-1], min=lb, max=ub))
        w_in_x = torch.matmul(self.w_in.T, torch.cat((logX, torch.tensor([inv_R / u[-1]]))))
        du = torch.cat((torch.matmul(self.w_out, torch.exp(w_in_x + self.w_b)), torch.tensor([0.0])))
        return du

# Выбираем начальные условия для первого эксперимента
u0 = torch.tensor(u0_list[0, :])

# Функция для предсказания решения с использованием нейронной сети
def predict_neuralode(u0, p):
    w_in, w_b, w_out = p2vec(p)
    neural_ode = NeuralODE(torch.tensor(w_in, dtype=torch.float32),
                           torch.tensor(w_b, dtype=torch.float32),
                           torch.tensor(w_out, dtype=torch.float32))
    sol = spi.solve_ivp(lambda t, y: neural_ode(t, torch.tensor(y)).detach().numpy(), tspan, u0.detach().numpy(), t_eval=tsteps)
    return np.clip(sol.y, -ub, ub)

predict_neuralode(u0, np_params)

i_obs = np.arange(6)

# Функция для вычисления потерь (ошибок модели) на основе среднего абсолютного отклонения
def loss_neuralode(p, i_exp):
    ode_data = ode_data_list[i_exp, i_obs, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)[i_obs, :]
    loss = np.mean(np.abs(ode_data / yscale[i_obs, None] - pred / yscale[i_obs, None]))
    return loss

# Функция обратного вызова для отображения результатов на графике
def cbi(p, i_exp):
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    for i in range(ns):
        plt.scatter(tsteps, ode_data[i, :], label=f"data_{i}")
        plt.plot(tsteps, pred[i, :], label=f"pred_{i}")
        plt.legend()
        plt.title(f"State {i + 1}")
        plt.show()

# Листы для хранения значений потерь в процессе обучения
l_loss_train = []
l_loss_val = []

# Функция обратного вызова для обновления модели и отображения прогресса
def cb(p, loss_train, loss_val, iter):
    global l_loss_train, l_loss_val
    l_loss_train.append(loss_train)
    l_loss_val.append(loss_val)

    if iter % n_plot == 0:
        display_p(p)
        print(f"Min loss train {min(l_loss_train):.4e}, val {min(l_loss_val):.4e}")

        # Случайный эксперимент для отображения графиков
        l_exp = [random.randint(0, n_exp_train - 1)]
        print("Update plot for experiment:", l_exp)
        for i_exp in l_exp:
            cbi(p, i_exp)

        # Отображение графика потерь
        plt.figure()
        plt.plot(l_loss_train, label="Training", xscale='log', yscale='log')
        plt.plot(l_loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("figs/loss.png")
        plt.close()

        # Сохранение текущего состояния модели
        with open("./checkpoint/mymodel.pkl", "wb") as f:
            pickle.dump({"p": p, "l_loss_train": l_loss_train, "l_loss_val": l_loss_val, "iter": iter}, f)

# Если флаг is_restart установлен, загружаем сохраненное состояние модели
if is_restart and os.path.exists("./checkpoint/mymodel.pkl"):
    with open("./checkpoint/mymodel.pkl", "rb") as f:
        checkpoint = pickle.load(f)
        np_params = checkpoint["p"]
        l_loss_train = checkpoint["l_loss_train"]
        l_loss_val = checkpoint["l_loss_val"]
        iter = checkpoint["iter"] + 1
else:
    iter = 1

# Начало цикла обучения модели
epochs = range(iter, n_epoch + 1)
loss_epoch = np.zeros(n_exp, dtype=np.float32)
grad_norm = np.zeros(n_exp_train, dtype=np.float32)

# Основной цикл обучения
for epoch in epochs:
    for i_exp in random.sample(range(n_exp_train), n_exp_train):
        # Вычисляем градиент функции потерь
        grad = minimize(lambda p: loss_neuralode(p, i_exp), np_params, jac=True, method="L-BFGS-B").jac
        grad_norm[i_exp] = np.linalg.norm(grad)
        
        # Обновляем параметры модели
        np_params -= grad * 0.01  # 0.01 - коэффициент обучения (learning rate)

    # Рассчитываем потери для каждого эксперимента
    for i_exp in range(n_exp):
        loss_epoch[i_exp] = loss_neuralode(np_params, i_exp)

    # Средние потери для обучающей и тестовой выборок
    loss_train = np.mean(loss_epoch[:n_exp_train])
    loss_val = np.mean(loss_epoch[n_exp_train:])
    
    # Обновляем описание прогресса в процессе обучения
    print(f"Epoch {epoch}, Loss train {loss_train:.2e}, val {loss_val:.2e}, "
          f"gnorm {np.mean(grad_norm):.1e}, lr 1e-2")
    
    # Обратный вызов для сохранения и вывода прогресса
    cb(np_params, loss_train, loss_val, epoch)
