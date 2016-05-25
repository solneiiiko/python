n, m = map(int, input().split())
matrix = [ list(map(int, input().split())) for i in range(n)]
for i in range(m):
	for j in range(n):
		print(matrix[j][i], end=' ')
	print('')