import csv

with open('Crimes') as f:
    crimes = csv.reader(f)
    res = {}

    for crime in crimes:
        if crime[2][6:10]=='2015':
            if res.get(crime[5]):
                res[crime[5]] += 1
            else:
                res[crime[5]] = 1
    print(res)
    _res = sorted(res, key=res.get)
    print(_res[-1])
