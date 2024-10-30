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

###################################
# Argments
is_restart = false;
p_cutoff = 0.0;                                                     # Непонятно для чего
n_epoch = 30000;                                                     # Количество эпох. Максимальное
n_plot = 100;                                                       # Частота формирования графиков. Через сколько эпох
opt = ADAMW(0.001, (0.9, 0.999), 1.f-8);                            # Оптимизатор
datasize = 500;                                                     # Размер датасетов?
tstep = 1;                                                          # Шаг времени для датасетов? или типа их количество?
n_exp_train = 30;                                                   # Размер данных для обучения
n_exp_test = 10;                                                    # Размер даных для теста
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

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1];
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4];
    dydt[3] = k[2] * y[1] - k[3] * y[3];
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4];
    dydt[5] = k[4] * y[2] * y[4];
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
u0_list[:, 1:2] .+= 2.f-1;
u0_list[:, 3:end] .= 0.f0;
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end


for i in 1:n_exp
    u0 = u0_list[i, :];
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end
y_std = maximum(hcat(std_list...), dims=2);

b0 = -10.0

function p2vec(p)
    w_b = p[1:nr] .+ b0;
    w_out = reshape(p[nr + 1:end], ns, nr);
    # w_out = clamp.(w_out, -2.5, 2.5);
    w_in = clamp.(-w_out, 0, 2.5);
    return w_in, w_b, w_out
end

function crnn!(du, u, p, t)
    w_in_x = w_in' * @. log(clamp(u, lb, ub));
    du .= w_out * @. exp(w_in_x + w_b);
end

u0 = u0_list[1, :]
p = randn(Float32, nr * (ns + 1)) .* 1.f-1;
# p[1:nr] .+= b0;

prob = ODEProblem(crnn!, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)

function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p);
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p; 
                  maxiters=maxiters)), -ub, ub)
    return pred
end
# predict_neuralode(u0, p);

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(exp.(w_b'), digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n\n")
end
display_p(p)

function loss_neuralode(p, i_exp)
    pred = predict_neuralode(u0_list[i_exp, :], p)
    loss = mae(ode_data_list[i_exp, :, :] ./ y_std, pred ./ y_std)
    return loss
end
# loss_neuralode(p, 1)

# Callback function to observe training

species = ["A", "B", "C", "D", "E"];
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    list_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], 
                      markercolor=:transparent,
                      label="Exp",
                      framestyle=:box)
        plot!(plt, tsteps, pred[i,:], label="CRNN-ODE")
        plot!(xlabel="Time", ylabel="Concentration of " * species[i])

        if i==1
            plot!(plt, legend=true, framealpha=0)
        else
            plot!(plt, legend=false)
        end

        push!(list_plt, plt)
    end
    plt_all = plot(list_plt...)

    png(plt_all, string(joinpath(@__DIR__, "figs", ""), i_exp))                                                    # Здесь сохраняется png файл
    # joinpath самостоятельно определяет относительный путь к папке/файлу
    
    return false
end

list_loss_train = []
list_loss_val = []
iter = 1
cb = function (p, loss_train, loss_val)
    global list_loss_train, list_loss_val, iter
    push!(list_loss_train, loss_train)
    push!(list_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)

        @printf("min loss train: %.4e, and validation: %.4e\n", minimum(list_loss_train), minimum(list_loss_val))

        list_exp = randperm(n_exp)[1:1];
        println("update plot for ", list_exp)
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss_train, xscale=:log10, yscale=:log10, label="train");
        plot!(plt_loss, list_loss_val, label="val");


        png(plt_loss, joinpath(@__DIR__, "figs", "loss.png"));                                                               # тут сохраняюется loss файл

        @save joinpath(@__DIR__, "checkpoint", "mymodel.bson") p opt list_loss_train list_loss_val iter                      # тут сохраняется bson файл
    end

    iter += 1;
end

if is_restart
    #@load "./checkpoint/mymodel.bson" p opt list_loss_train list_loss_val iter;                                        Переписываю путь
    @load joinpath(@__DIR__, "checkpoint", "mymodel.bson") p opt list_loss_train list_loss_val iter;                         # тут читается bson файл
    iter += 1; 
end

# opt = ADAMW(0.001, (0.9, 0.999), 1.f-5);

i_exp = 1
epochs = ProgressBar(iter:n_epoch)
loss_epoch = zeros(Float32, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)

        grad = gradient(p) do x
            Zygote.forwarddiff(x) do x
                loss_neuralode(x, i_exp)
            end
        end
        update!(opt, p, grad[1])
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
    cb(p, loss_train, loss_val);
end

# min loss train: 2.3712e-02, and validation: 2.5013e-02     10000 тестов по 10
# min loss train: 1.0619e-02, and validation: 1.0639e-02     10000 тестов по 100
# min loss train: 2.3660e-02, and validation: 2.4966e-02     40000 тестов по 10
# min loss train: 9.1484e-03, and validation: 9.4130e-03     10000 тестов по 200
# min loss train: 9.1429e-03, and validation: 9.3993e-03     20000 тестов по 200
# min loss train: 8.7199e-03, and validation: 8.8346e-03     10000 тестов по 300
# min loss train: 9.2125e-03, and validation: 9.3619e-03     10000 тестов по 1000
# min loss train: 8.5276e-03, and validation: 8.7177e-03     20000 тестов по 1000
# min loss train: 8.4164e-03, and validation: 8.4310e-03     10000 тестов по 500