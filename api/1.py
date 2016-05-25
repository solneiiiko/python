'''
В этой задаче вам необходимо воспользоваться API сайта numbersapi.com

Вам дается набор чисел. Для каждого из чисел необходимо узнать, существует ли интересный математический факт об этом числе.

Для каждого числа выведите Interesting, если для числа существует интересный факт, и Boring иначе.
Выводите информацию об интересности чисел в таком же порядке, в каком следуют числа во входном файле.

Пример запроса к интересному числу:
http://numbersapi.com/31/math?json=true

Пример запроса к скучному числу:
http://numbersapi.com/999/math?json=true

Пример входного файла:
31
999
1024
502

﻿Пример выходного файла:
Interesting
Boring
Interesting
Boring
'''
import requests

url_format = 'http://numbersapi.com/{}/math?json=true'
with open('in.txt') as f:
    nums = f.read().split()
res = []
for num in nums:
    url = url_format.format(num.rstrip())
    req = requests.get(url)
    data = req.json()
    if data['found']:
        res.append('Interesting')
    else:
        res.append('Boring')
    print(res)
with open('out.txt', 'w') as f:
    f.write('\n'.join(res))
