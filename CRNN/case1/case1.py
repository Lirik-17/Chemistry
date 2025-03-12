import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

# Установка генератора случайных чисел
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# Гиперпараметры
is_restart = False
p_cutoff = 0.0
n_epoch = 10000
n_plot = 100
lr = 0.001
datasize = 100
tstep = 1
n_exp_train = 30
n_exp_test = 10
n_exp = n_exp_train + n_exp_test
noise = 0.05
ns = 5
nr = 4
k = torch.FloatTensor([0.1, 0.2, 0.13, 0.3])
maxiters = 10000

lb = 1e-5
ub = 1.0

# Определение истинной функции ОДУ
def true_ODEfunc(t, y, k):

    dydt = torch.zeros_like(y)
    dydt[0] = -2 * k[0] * y[0]**2 - k[1] * y[0]
    dydt[1] = k[0] * y[0]**2 - k[3] * y[1] * y[4]
    dydt[2] = k[1] * y[0] - k[2] * y[2]
    dydt[3] = k[2] * y[2] - k[3] * y[1] * y[4]
    dydt[4] = k[3] * y[1] * y[4]
    return dydt

# Генерация начальных условий и данных с шумом
u0_list = torch.rand(n_exp, ns)
u0_list[:, :2] += 0.5
u0_list[:, 2:] = 0.0
tspan = [0.0, datasize * tstep]
tsteps = torch.linspace(tspan[0], tspan[1], datasize)

ode_data_list = []
std_list = []

def max_min(ode_data: any)-> any:
    """_summary_

    Args:
        ode_data (any): _description_ 

    Returns:
        any: _description_
    """    
    return ode_data.max(dim=1)[0] - ode_data.min(dim=1)[0] + lb

for i in range(n_exp):
    u0 = u0_list[i, :]
    solution = odeint(true_ODEfunc, u0, tsteps, method='dopri5', options={'rtol': 1e-5, 'atol': 1e-2}, args=(k,))
    solution += torch.randn_like(solution) * solution * noise
    ode_data_list.append(solution)
    std_list.append(max_min(solution))

y_std = torch.stack(std_list).max(dim=0)[0]

# Преобразование параметров
b0 = -10.0

def p2vec(p):
    w_b = p[:nr] + b0
    w_out = p[nr:].view(ns, nr)
    w_in = torch.clamp(-w_out, 0, 2.5)
    return w_in, w_b, w_out

# Определение функции нейросетевой модели
class CRNNODEFunc(nn.Module):
    def forward(self, t, u, p):
        w_in, w_b, w_out = p2vec(p)
        w_in_x = torch.matmul(torch.log(torch.clamp(u, lb, ub)), w_in.t())
        return torch.matmul(torch.exp(w_in_x + w_b), w_out)

# Предсказание с использованием CRNN ODE
def predict_neuralode(u0, p):
    crnn_func = CRNNODEFunc()
    pred = odeint(crnn_func, u0, tsteps, method='dopri5', options={'rtol': 1e-5, 'atol': 1e-2}, args=(p,))
    return torch.clamp(pred, -ub, ub)

# Функция потерь (MAE)
def loss_neuralode(p, i_exp):
    pred = predict_neuralode(u0_list[i_exp], p)
    loss = torch.mean(torch.abs(ode_data_list[i_exp] / y_std - pred / y_std))
    return loss

# Обратный вызов для визуализации
species = ["A", "B", "C", "D", "E"]

def cbi(p, i_exp):
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0_list[i_exp], p).detach()
    for i in range(ns):
        plt.figure()
        plt.scatter(tsteps.numpy(), ode_data[:, i].numpy(), label="Exp", alpha=0.5)
        plt.plot(tsteps.numpy(), pred[:, i].numpy(), label="CRNN-ODE")
        plt.xlabel("Time")
        plt.ylabel(f"Concentration of {species[i]}")
        plt.legend()
        plt.show()

# Оптимизация модели
p = torch.randn(nr * (ns + 1), requires_grad=True)
optimizer = optim.AdamW([p], lr=lr)

list_loss_train = []
list_loss_val = []
iter = 1

# Цикл обучения
for epoch in range(n_epoch):
    for i_exp in random.sample(range(n_exp_train), n_exp_train):
        optimizer.zero_grad()
        loss = loss_neuralode(p, i_exp)
        loss.backward()
        optimizer.step()

    # Вычисление ошибок для всех экспериментов
    loss_epoch = torch.zeros(n_exp)
    for i_exp in range(n_exp):
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)

    loss_train = loss_epoch[:n_exp_train].mean()
    loss_val = loss_epoch[n_exp_train:].mean()

    list_loss_train.append(loss_train.item())
    list_loss_val.append(loss_val.item())

    if epoch % n_plot == 0:
        print(f"Epoch {epoch}, Loss train {loss_train:.4e}, Loss val {loss_val:.4e}")
        cbi(p, random.choice(range(n_exp)))

# Сохранение графиков потерь
plt.plot(list_loss_train, label="train", color="blue")
plt.plot(list_loss_val, label="val", color="red")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
