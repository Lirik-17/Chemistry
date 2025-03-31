'''
Код для генерации данных для обучения и тестирования нейронной сети.
Изначально он содержит данные по 6 входных концентраций веществ
и 6 выходных концентраций веществ, температуру и время реакции
'''

import math
import random

random.seed(1)

def get_f(F, W, matr):
    '''

    :param F: правые части дифференциальных уравнений
    :param W: скорости химических реакций
    :param matr: матрица стехиометрических коэффициентов схемы превращения
    :return: рассчитанные правые части дифференциальных уравнений
    '''
    for i in range(len(matr[0])):       # число вещество
        F[i]=0
        for j in range(len(matr)):      # число реакций                                # pylint: disable=consider-using-enumerate
            F[i] += matr[j][i]*W[j]

    return F

def get_w(W, k, C, matr):
    '''

    :param W: скорости химических реакций
    :param k: константы скоростей химических реакций
    :param C: концентрации
    :param matr: матрица стехиометрических коэффициентов схемы превращения
    :return: рассчитанные скорости химических реакций
    '''
    for i in range(len(matr)):                                                         # pylint: disable=consider-using-enumerate
        W[i] = 1
        flag = False
        for j in range(len(matr[i])):
            if -matr[i][j] > 0:
                W[i] = W[i]*C[j]**(-matr[i][j])
                flag = True
        if not flag:
            W[i] = 0
        else:
            W[i] = W[i]*k[i]
    return W

def get_k(k, k0, Ea, T):
    '''
    Функция рассчета констант скоростей химических реакций по расширенному уравнению Аррениуса
    :param k:  константы скоростей химических реакций
    :param k0: предэкспоненты химических реакций
    :param Ea: энергии активации химических реакций
    :param T: температура
    :return: рассчитанные константы скоростей химических реакций
    '''
    for i in range(len(k)):                                                               # pylint: disable=consider-using-enumerate
        k[i] = k0[i]*math.exp(-Ea[i]/(8.314*T))*(T**0.5)
    return k

def set_default():
    '''
    Функция установления начальных параметров
    :return: кортеж (
        k0 - предэкспоненты,
        Ea - энергии активации,
        C - концентрации,
        W - скорости х.р.,
        k - константы скорости х.р.,
        F - правые части дифуров,
        T - температура,
        matr - матрица стехиометрических коэффициентов,
        vr - время контакта,
        h - шаг интегрирования по времени
    )
    '''
    n = 6     #веществ
    m = 5     #реакций
    k0 = [100000000, 90000000, 85000000, 110000000, 95000000]
    Ea = [70000, 60000, 55000, 80000, 75000]
    T = 250 # + random.randint(0,100)
    matr = [[-2, -1, 1, 1, 0, 0],
            [0, -2, 1, 0, 2, 0],
            [0, -1, -2, 1, 0, 1],
            [1, 0, 0, -2, 1, 0],
            [1, 0, 0, 0, -2, 1]
          ]
    vr = 10
    vr = 10 + random.randint(0, 3)
    h = 0.01
    # C = [0 for _ in range(n)]
    C = [1/6 for _ in range(n)]
    F = [0 for _ in range(n)]
    W = [0 for _ in range(m)]
    k = [0 for _ in range(m)]
    for i in range(len(C)):                                                      # pylint: disable=consider-using-enumerate
        C[i] = random.randint(0, 100)/100

    return k0, Ea, C, W, k, F, T, matr, vr, h

def calculation():
    '''
    Основная функция расчета
    :return:
    '''
    print('Вещество', 'Нач. конц.', 'Кон. конц.', 'Температура', 'Время')
    # задаем параметры задачи
    k0, Ea, C, W, k, F, T, matr, vr, h = set_default()
    C0str = ''
    Ckstr = ''
    # строки вывода результатов
    for i in range(len(C)):
        C0str += ' C0' + str(i) # начальная концентрация
        Ckstr += ' Ck' + str(i) # конечная концентрация

    strC = C0str + Ckstr + ' T ' + ' Время'

    print(strC)
    # n - количество кейсов, пробегаем по ним
    for _ in range(5):
        strC = ''
        # задаем параметры задачи
        k0, Ea, C, W, k, F, T, matr, vr, h = set_default()
        # запоминаем начальные концентрации
        for i in range(len(C)):                                           # pylint: disable=consider-using-enumerate
            strC += str(C[i])+ ' '
        # начинаем расчет
        for _ in range(int(vr / h)):
            # получаем константы скоростей
            k = get_k(k, k0, Ea, T)
            # получаем скорости
            W = get_w(W, k, C, matr)
            # получаем правые части диф. уравнений
            F = get_f(F, W, matr)
            # решаем методом Эйлера
            for j in range(len(C)):                                         # pylint: disable=consider-using-enumerate
                C[j] += F[j] * h
        # запоминаем конечные концентрации
        for i in range(len(C)):                                             # pylint: disable=consider-using-enumerate
            strC += str(C[i]) + ' '
        # запоминаем температуру и время
        strC += str(T) + ' ' + str(vr)
        # выводим результат в строку
        print("Начальные концентрации:")
        print(strC)
        print("Конечные концентрации:")
        print(*C)
        print()


if __name__ == "__main__":
    # расчет
    calculation()
