'''
Вам дано описание наследования классов в формате JSON. 
Описание представляет из себя массив JSON-объектов, которые соответствуют классам. У каждого JSON-объекта есть поле name, которое содержит имя класса, и поле parents, которое содержит список имен прямых предков.

Пример:
[{"name": "A", "parents": []}, {"name": "B", "parents": ["A", "C"]}, {"name": "C", "parents": ["A"]}]

﻿Эквивалент на Python:

class A:
    pass

class B(A, C):
    pass

class C(A):
    pass

Гарантируется, что никакой класс не наследуется от себя явно или косвенно, и что никакой класс не наследуется явно от одного класса более одного раза.

Для каждого класса вычислите предком скольких классов он является и выведите эту информацию в следующем формате.

<имя класса> : <количество потомков>

Выводить классы следует в лексикографическом порядке.

Sample Input:
[{"name": "A", "parents": []}, {"name": "B", "parents": ["A", "C"]}, {"name": "C", "parents": ["A"]}]
Sample Output:
A : 3
B : 1
C : 2
'''
import json
cls = json.loads(input())
res = {}
for cl in cls:
    viewed_parents = set()
    chield = cl['name']
    parents = cl['parents'] + [chield]
    while len(parents)>0:
        parent_name = parents.pop(0)
        if res.get(parent_name)==None:
            res[parent_name] = set([parent_name])
        res[parent_name].add(chield)
        if (parent_name in viewed_parents)==False:
            parents += list(filter(lambda e: e['name']==parent_name,cls))[0]['parents']
            viewed_parents.add(parent_name)
for cl_name in sorted(res):
    print('{} : {}'.format(cl_name, len(res[cl_name])))