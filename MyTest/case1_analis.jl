GC.gc()

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
display(u0_list)
u0_list[:, 1:2] .+= 2.f-1;
u0_list[:, 3:end] .= 0.f0;
println()
display(u0_list)

tspan = Float32[0.0, datasize * tstep];
println("tspan:")
println(tspan)

tsteps = range(tspan[1], tspan[2], length=datasize);
println("tsteps:")
println(tsteps)

ode_data_list = zeros(Float32, (n_exp, ns, datasize));
# println("ode_data_list:")
# display(ode_data_list)

std_list = [];
