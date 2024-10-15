import torch
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Установка сидов
torch.manual_seed(1234)
np.random.seed(1234)

###################################
# Аргументы
is_restart = False
p_cutoff = 0.0
n_epoch = 1000
n_plot = 100
opt = optim.AdamW([torch.tensor(1.0)], lr=0.001)
datasize = 100
tstep = 0.4
n_exp_train = 20
n_exp_test = 10
n_exp = n_exp_train + n_exp_test
noise = 0.05
ns = 5
nr = 4
k = torch.tensor([0.1, 0.2, 0.13, 0.3], dtype=torch.float32)
#atol = 1e-5
#rtol = 1e-2
maxiters = 10000
lb = 1e-5
ub = 1.0
####################################

# Функция ODE
def true_ode_func(t, y, k):
    dydt = torch.zeros_like(y)
    dydt[0] = -2 * k[0] * y[0] ** 2 - k[1] * y[0]
    dydt[1] = k[0] * y[0] ** 2 - k[3] * y[1] * y[4]
    dydt[2] = k[1] * y[0] - k[2] * y[2]
    dydt[3] = k[2] * y[2] - k[3] * y[1] * y[4]
    dydt[4] = k[3] * y[1] * y[4]
    return dydt

# Генерация набора данных
u0_list = np.random.rand(n_exp, ns).astype(np.float32)
u0_list[:, 0:2] += 0.5
u0_list[:, 2:] = 0.0
tspan = [0.0, datasize * tstep]
tsteps = np.linspace(tspan[0], tspan[1], datasize)
ode_data_list = np.zeros((n_exp, ns, datasize), dtype=np.float32)

def max_min(ode_data):
    return np.max(ode_data, axis=1) - np.min(ode_data, axis=1) + lb

std_list = []
for i in range(n_exp):
    u0 = u0_list[i, :]
    # ode_data = odeint(true_ode_func, torch.tensor(u0), torch.tensor(tsteps))              # Редачу эту строку
    ode_data = ode_data.numpy() + np.random.randn(*ode_data.shape) * noise
    ode_data_list[i] = ode_data
    std_list.append(max_min(ode_data))

y_std = np.max(np.hstack(std_list), axis=1)

b0 = -10.0

def p2vec(p):
    w_b = p[:nr] + b0
    w_out = p[nr:].reshape(ns, nr)
    w_in = np.clip(-w_out, 0, 2.5)
    return w_in, w_b, w_out

def crnn_func(t, u, p):
    w_in, w_b, w_out = p2vec(p)
    w_in_x = np.dot(w_in, np.log(np.clip(u, lb, ub)))
    return np.dot(w_out, np.exp(w_in_x + w_b))

u0 = u0_list[0, :]
p = np.random.randn(nr * (ns + 1)).astype(np.float32)

def predict_neural_ode(u0, p):
    pred = odeint(crnn_func, torch.tensor(u0), torch.tensor(tsteps), method='dopri5', options=dict(atol=atol, rtol=rtol))
    return torch.clamp(pred, min=-ub, max=ub)

def display_p(p):
    w_in, w_b, w_out = p2vec(p)
    print("species (column) reaction (row)")
    print("w_in")
    print(np.round(w_in.T, 3))
    print("\nw_b")
    print(np.round(np.exp(w_b), 3))
    print("\nw_out")
    print(np.round(w_out, 3))
    print("\n\n")

display_p(p)

def loss_neural_ode(p, i_exp):
    pred = predict_neural_ode(u0_list[i_exp, :], p)
    return torch.mean(torch.abs(ode_data_list[i_exp] / y_std - pred.detach().numpy() / y_std))

# Callback для отображения процесса обучения
list_loss_train = []
list_loss_val = []
iter = 1

for epoch in range(n_epoch):
    for i_exp in np.random.permutation(n_exp_train):
        grad = torch.autograd.functional.jacobian(lambda p: loss_neural_ode(p, i_exp), p)
        opt.step(lambda: grad)

    if iter % n_plot == 0:
        display_p(p)
        loss_train = np.mean([loss_neural_ode(p, i) for i in range(n_exp_train)])
        loss_val = np.mean([loss_neural_ode(p, i) for i in range(n_exp_train, n_exp)])
        list_loss_train.append(loss_train)
        list_loss_val.append(loss_val)
        print(f'Epoch {iter}: Loss train {loss_train}, val {loss_val}')
    iter += 1

# Callback функция для наблюдения за обучением и визуализации
def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("figs/loss.png")
    plt.close()

def cbi(p, i_exp):
    ode_data = ode_data_list[i_exp]
    pred = predict_neural_ode(u0_list[i_exp, :], p)
    species = ["A", "B", "C", "D", "E"]
    fig, axs = plt.subplots(ns, 1, figsize=(8, 12))
    for i in range(ns):
        axs[i].scatter(tsteps, ode_data[i, :], label="Exp", color="blue", alpha=0.5)
        axs[i].plot(tsteps, pred[:, i].detach().numpy(), label="CRNN-ODE", color="red")
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel(f"Concentration of {species[i]}")
        if i == 0:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"figs/i_exp_{i_exp}.png")
    plt.close()

# Обучение
list_loss_train = []
list_loss_val = []
iter = 1

if is_restart:
    checkpoint = torch.load("./checkpoint/mymodel.pth")
    p = checkpoint['p']
    opt = checkpoint['opt']
    list_loss_train = checkpoint['list_loss_train']
    list_loss_val = checkpoint['list_loss_val']
    iter = checkpoint['iter']

for epoch in range(iter, n_epoch + 1):
    for i_exp in np.random.permutation(n_exp_train):
        # Градиентный спуск
        opt.zero_grad()
        loss = loss_neural_ode(p, i_exp)
        loss.backward()
        opt.step()

    # Оценка качества модели на каждом шаге
    loss_train = np.mean([loss_neural_ode(p, i).item() for i in range(n_exp_train)])
    loss_val = np.mean([loss_neural_ode(p, i).item() for i in range(n_exp_train, n_exp)])
    list_loss_train.append(loss_train)
    list_loss_val.append(loss_val)

    if epoch % n_plot == 0:
        display_p(p)
        print(f'Epoch {epoch}: Loss train {loss_train:.4e}, val {loss_val:.4e}')
        plot_loss(list_loss_train, list_loss_val)

        list_exp = np.random.permutation(n_exp)[:1]
        for i_exp in list_exp:
            cbi(p, i_exp)

        torch.save({
            'p': p,
            'opt': opt,
            'list_loss_train': list_loss_train,
            'list_loss_val': list_loss_val,
            'iter': epoch
        }, "./checkpoint/mymodel.pth")
