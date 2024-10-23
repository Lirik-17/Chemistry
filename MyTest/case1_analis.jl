using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load
using DiffEqBase

Random.seed!(1234);

#
#
# Так же введу все входные параметры
#
###################################
# Argments
is_restart = false;
p_cutoff = 0.0;                                                     # Непонятно для чего
n_epoch = 1000;                                                     # Количество эпох. Максимальное
n_plot = 100;                                                       # Частота формирования графиков. Через сколько эпох
opt = ADAMW(0.001, (0.9, 0.999), 1.f-8);                            # Оптимизатор
datasize = 10;                                                     # Размер датасетов?
tstep = 1;                                                          # Шаг времени для татасетов? или типа их количество?
n_exp_train = 6;                                                   # Размер данных для обучения
n_exp_test = 2;                                                    # Размер даных для теста
n_exp = n_exp_train + n_exp_test;                                   # Общий размер данных
noise = 5.f-2;                                                      # ШУМ
ns = 5;                                                             # Количество веществ
nr = 4;                                                             # Количество хим. реакций
k = Float32[0.1, 0.2, 0.13, 0.3];                                   # константы хим. реакций
alg = Tsit5();                                                      # Алгоритм для решения ОДУ?
atol = 1e-5;                                                        # Параметр точности для ОДУ
rtol = 1e-2;                                                        # Параметр точности для ОДУ

maxiters = 10000;                                                   # Не понял для чего нужно

lb = 1.f-5;
ub = 1.f1;
####################################

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
u0_list[:, 1:2] .+= 2.f-1;
u0_list[:, 3:end] .= 0.f0;

# В результате u0_list - матрица ns столбцов и n_exp строк, в которой 3+ столбцы нули


tspan = Float32[0.0, datasize * tstep];                           # одномерный массив с двумя элементами
tsteps = range(tspan[1], tspan[2], length=datasize);              # последовательность от [1] до [2] с шагом length
ode_data_list = zeros(Float32, (n_exp, ns, datasize));            # трехмерный массив нулей. Для будущего хранения решений ODE

std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1];
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4];
    dydt[3] = k[2] * y[1] - k[3] * y[3];
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4];
    dydt[5] = k[4] * y[2] * y[4];
end

# Дале идет цикл с решением ODE для каждой строки матрицы u0_list.
# Решение записывается в std_list

for i in 1:n_exp
    u0 = u0_list[i, :];
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end

y_std = maximum(hcat(std_list...), dims=2);