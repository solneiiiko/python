# balls = input()
# n = balls.count('A')/((len(balls)+1)/2)
# print("%.2f" % n)
#######################################
import collections
from collections import Counter
cnt = Counter()
balls = input().split()
for ball in balls:
    cnt[ball]+=1
n = cnt['A']/len(balls)
print("%.2f" % n)
