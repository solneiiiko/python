def C(n,k):
	if k==0:
		return 1
	elif k>n:
		return 0 
	else:
		return C(n-1,k)+C(n-1,k-1)
n, k = map(int, input().split())
print(C(n,k))
