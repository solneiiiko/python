import os

def find_py_file(lst):
	for name in lst:
		if name[-3:]=='.py':
			return True
	return False

res = []
for i in os.walk('main'):
	if find_py_file(i[2]):
		res.append(i[0])
with open('out.txt', 'w') as f:
	f.write('\n'.join(sorted(res)))
