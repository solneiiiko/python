'''
Итак, мы знаем, как посчитать «назад» ошибку из l+1 слоя в ll-й.
Чтобы это знание не утекло куда подальше, давайте сразу его запрограммируем.
Заодно вспомним различия между .dot и *.

Напишите функцию, которая, используя набор ошибок δl+1 для n примеров, матрицу весов Wl+1 и набор значений
сумматорной функции на l-м шаге для этих примеров, возвращает значение ошибки δl на l-м слое сети.

Сигнатура: get_error(deltas, sums, weights), где
deltas — ndarray формы (n, nl+1), содержащий в i-й строке значения ошибок для i-го примера из входных данных,
sums — ndarray формы (n, nl), содержащий в i-й строке значения сумматорных функций нейронов l-го слоя для i-го
	примера из входных данных, 
weights — ndarray формы (nl+1, nl), содержащий веса для перехода между l-м и l+1-м слоем сети.
Требуется вернуть вектор δl — ndarray формы (nl, 1);
мы не проверяем размер (форму) ответа, но это может помочь вам сориентироваться.
Все нейроны в сети — сигмоидальные. Функции sigmoid и sigmoid_prime уже определены.

Обратите внимание, в предыдущей задаче мы работали только с одним примером, а сейчас вам на вход подаётся несколько.
Не забудьте учесть этот факт и просуммировать всё, что нужно. И разделить тоже.
Подсказка: J=1n∑ni=112∣∣y^(i)−y(i)∣∣2⟹∂J∂θ=1n∑ni=1∂∂θ(12∣∣y^(i)−y(i)∣∣2)J=1n∑i=1n12|y^(i)−y(i)|2⟹∂J∂θ=1n∑i=1n∂∂θ(12|y^(i)−y(i)|2) для любого параметра θθ, который не число примеров. 
'''
import numpy as np

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    n = sums.shape[0]
    res = (deltas.dot(weights)) * sigmoid_prime(sums)
    res = res.sum(axis=0)/n
    return res


def neuron(x,weights,b, func):
	return func(x.dot(weights) + b)
# n = 10
# n_l = 30
# n_l_1 = 20
# deltas = np.random.random_sample((n,n_l_1))
# sums = np.random.random_sample((n,n_l))
# weights = np.random.random_sample((n_l_1,n_l))
# print(get_error(deltas, sums, weights))
#####################
# y = 1
# weights = np.array([[[2,2], [2,2],[2,2]],[[1],[1]]])
# x = np.array([0,1,2])
# b = np.array([[0,0],[0]])
# z = list([0,0])
# z[0] = neuron(x, weights[0], b[0], sigmoid)
# z[1] = neuron(np.array(z[0]), weights[1], b[1],sigmoid)
# prime_z_1 = neuron(np.array(z[0]), weights[1], b[1],sigmoid_prime)
# delta_out = np.array([[(z[1][0]-y)*prime_z_1[0]]])
# # deltas = 
# deltas = get_error(delta_out,z[0],np.array(weights[1]).T)
# # print(x.T * deltas)
# print(deltas)
# print(np.array([2,2])*deltas)
