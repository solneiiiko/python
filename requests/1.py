'''
Рассмотрим два HTML-документа A и B.
Из A можно перейти в B за один переход, если в A есть ссылка на B, т. е. внутри A есть тег <a href="B">, возможно с дополнительными параметрами внутри тега.
Из A можно перейти в B за два перехода если существует такой документ C, что из A в C можно перейти за один переход и из C в B можно перейти за один переход.

Вашей программе на вход подаются две строки, содержащие url двух документов A и B.
Выведите Yes, если из A в B можно перейти за два перехода, иначе выведите No.

Обратите внимание на то, что не все ссылки внутри HTML документа могут вести на существующие HTML документы.

Sample Input 1:
https://stepic.org/media/attachments/lesson/24472/sample0.html
https://stepic.org/media/attachments/lesson/24472/sample2.html
Sample Output 1:
Yes

Sample Input 2:
https://stepic.org/media/attachments/lesson/24472/sample0.html
https://stepic.org/media/attachments/lesson/24472/sample1.html
Sample Output 2:
No

Sample Input 3:
https://stepic.org/media/attachments/lesson/24472/sample1.html
https://stepic.org/media/attachments/lesson/24472/sample2.html
Sample Output 3:
Yes
'''
import requests
import re

a = input()
b = input()
count_stop = 2
count = 0
is_ok = False
pattern_b = r'<a\s.*?href="'+b+'".*?>'
pattern_all = r'<a\s.*?href="([^"]+?)".*?>'

req = requests.get(a)
urls = re.findall(pattern_all, str(req.content))
i = 0
while is_ok==False and i<len(urls):
    try:
        req = requests.get(urls[i])
    except requests.exceptions.MissingSchema:
        pass
    else:
        if re.findall(pattern_b, str(req.content)):
            is_ok = True
    i += 1
print('Yes') if is_ok==True else print('No')
