using OrdinaryDiffEq, Flux, Optim, Random, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load

# Устанавливаем начальное значение для генератора случайных чисел
Random.seed!(1234);

###################################
# Параметры программы

# Флаг для перезапуска
is_restart = true;

# Количество эпох для обучения
n_epoch = 10000;

# Частота вывода графиков
n_plot = 50;

# Размер данных и шаг времени
datasize = 50;
tstep = 1;

# Количество экспериментальных данных для обучения и тестирования
n_exp_train = 20;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;

# Уровень шума
noise = 0.05;

# Количество состояний (ns) и количество реакций (nr)
ns = 6;
nr = 3;

# Алгоритм для решения ОДУ
alg = AutoTsit5(Rosenbrock23(autodiff=false));

# Параметры точности решения ОДУ
atol = 1e-6;
rtol = 1e-3;

# Оптимизатор с экспоненциальным затуханием скорости обучения
opt = Flux.Optimiser(ExpDecay(5e-3, 0.5, 500 * n_exp_train, 1e-4),
                     ADAMW(0.005, (0.9, 0.999), 1.f-6));

# Пределы для параметров оптимизации
lb = 1.f-6;
ub = 1.f1;
####################################

# Функция для описания системы ОДУ
function trueODEfunc(dydt, y, k, t)
    # Обозначения компонентов системы
    # TG(1),ROH(2),DG(3),MG(4),GL(5),R'CO2R(6)
    r1 = k[1] * y[1] * y[2];  # реакция 1
    r2 = k[2] * y[3] * y[2];  # реакция 2
    r3 = k[3] * y[4] * y[2];  # реакция 3
    dydt[1] = - r1;  # изменение концентрации TG
    dydt[2] = - r1 - r2 - r3;  # изменение концентрации ROH
    dydt[3] = r1 - r2;  # изменение концентрации DG
    dydt[4] = r2 - r3;  # изменение концентрации MG
    dydt[5] = r3;  # изменение концентрации GL
    dydt[6] = r1 + r2 + r3;  # изменение концентрации R'CO2R
    dydt[7] = 0.f0;  # дополнительное уравнение (возможно для стабилизации)
end

# Логарифмы предэкспоненциальных факторов для уравнения Аррениуса
logA = Float32[18.60f0, 19.13f0, 7.93f0];

# Энергии активации для реакций
Ea = Float32[14.54f0, 14.42f0, 6.47f0];  # в ккал/моль

# Функция для расчета констант скорости реакций по уравнению Аррениуса
function Arrhenius(logA, Ea, T)
    R = 1.98720425864083f-3  # Газовая постоянная в ккал/(моль*K)
    k = exp.(logA) .* exp.(-Ea ./ R ./ T)  # Расчет констант скорости
    return k
end

# Генерация начальных условий для экспериментов
u0_list = rand(Float32, (n_exp, ns + 1));
u0_list[:, 1:2] = u0_list[:, 1:2] .* 2.0 .+ 0.2;  # Диапазон для TG и ROH
u0_list[:, 3:ns] .= 0.0;  # Начальные концентрации остальных компонентов
u0_list[:, ns + 1] = u0_list[:, ns + 1] .* 20.0 .+ 323.0;  # Температура [K]

# Задание временного интервала для моделирования
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);

# Создание массива для хранения результатов моделирования
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
yscale_list = [];

# Функция для нормализации данных
function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

# Решение задачи Коши для каждого эксперимента
for i in 1:n_exp
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[end])  # Расчет констант скорости
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)  # Задача ОДУ
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :]  # Решение ОДУ
    ode_data += randn(size(ode_data)) .* ode_data .* noise  # Добавление шума к данным
    ode_data_list[i, :, :] = ode_data  # Сохранение данных
    push!(yscale_list, max_min(ode_data))  # Нормализация данных
end

# Максимальные значения масштабов данных по всем экспериментам
yscale = maximum(hcat(yscale_list...), dims=2);

# Инициализация параметров для оптимизации
np = nr * (ns + 2) + 1;
p = randn(Float32, np) .* 0.1;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[end] = 0.1;

# Функция для преобразования параметров оптимизации в веса
function p2vec(p)
    slope = p[nr * (ns + 2) + 1] .* 100
    w_b = p[1:nr] .* slope
    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr)
    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope)
    w_in = clamp.(-w_out, 0, 4)
    w_in = vcat(w_in, w_in_Ea')
    return w_in, w_b, w_out
end

# Функция для отображения параметров
function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in | w_b")
    w_in_ = vcat(w_in, w_b')'
    show(stdout, "text/plain", round.(w_in_, digits=3))
    println("\\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\\n")
end

# Отображаем разделитель для форматирования
println("\\n")

# Отображаем текущие значения параметров
display_p(p)

# Обратное значение газовой постоянной для расчетов
inv_R = - 1 / 1.98720425864083f-3;

# Основная функция нейронной сети для решения ОДУ
function crnn(du, u, p, t)
    # Логарифмы концентраций компонентов (исключая последний элемент, который обычно является температурой)
    logX = @. log(clamp(u[1:end - 1], lb, ub))
    
    # Входной слой: расчет значений нейронов с учетом весов и температурного коэффициента
    w_in_x = w_in' * vcat(logX, inv_R / u[end])
    
    # Выходной слой: расчет производных
    du .= vcat(w_out * (@. exp(w_in_x + w_b)), 0.f0)
end

# Выбираем начальные условия для первой эксперимента
u0 = u0_list[1, :];

# Создаем задачу для решения ОДУ
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol)

# Используем метод обратного дифференцирования с проверкой точек
sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());

# Функция для предсказания решения с использованием нейронной сети
function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p)
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p, sensalg=sense)), -ub, ub)
    return pred
end

# Выполняем первое предсказание для проверки
predict_neuralode(u0, p)

# Индексы наблюдаемых переменных (все 6 состояний)
i_obs = [1, 2, 3, 4, 5, 6];

# Функция для вычисления потерь (ошибок модели) на основе среднего абсолютного отклонения
function loss_neuralode(p, i_exp)
    ode_data = @view ode_data_list[i_exp, i_obs, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)[i_obs, :]
    loss = mae(ode_data ./ yscale[i_obs], pred ./ yscale[i_obs])
    return loss
end

# Функция обратного вызова для отображения результатов на графике
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    l_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], markercolor=:transparent,
                      title=string(i), label=string("data_", i))
        plot!(plt, tsteps, pred[i,:], label=string("pred_", i))
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false)
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

# Листы для хранения значений потерь в процессе обучения
l_loss_train = []
l_loss_val = []
iter = 1

# Функция обратного вызова для обновления модели и отображения прогресса
cb = function (p, loss_train, loss_val)
    global l_loss_train, l_loss_val, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)
        @printf("min loss train %.4e val %.4e\\n", minimum(l_loss_train), minimum(l_loss_val))

        l_exp = randperm(n_exp)[1:1];
        println("update plot for ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, xscale=:log10, yscale=:log10, 
                        framestyle=:box, label="Training")
        plot!(plt_loss, l_loss_val, label="Validation")
        plot!(xlabel="Epoch", ylabel="Loss")
        png(plt_loss, "figs/loss")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    end
    iter += 1;
end

# Если флаг is_restart установлен, загружаем сохраненное состояние модели
if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    iter += 1;
end

# Начало цикла обучения модели
i_exp = 1
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);

# Основной цикл обучения
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p)
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.2e val %.2e gnorm %.1e lr %.1e", 
                                             loss_train, loss_val, mean(grad_norm), opt.eta)))
end
