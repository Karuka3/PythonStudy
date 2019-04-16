# a, b, c, d, eの平均値、中央値、分散を求めてください
# a~eの数字が変わっても答えが正確に出るよう書いてください

a = 2
b = 5
c = 10
d = 6
e = 22

### ここにコードを入れてください
data = [a,b,c,d,e]
S = sum(data)
N = len(data)
mean = S/N


data.sort()
if N % 2 == 0:
    median1 = N/2
    median2 = N/2 + 1
    
    median1 = int(median1) - 1
    median2 = int(median2) - 1

    median = (data[median1] + data[median2])/2
else :
    median = data[int(N/2)]
 
newdata = []
for n in data:
    newdata.append(n*n)

sd = sum((x - mean)**2 for x in data) / N

# 答え（下の#を消して答えを入れてください）
#mean   = 9.0
#median = 6.0 
#sd     = 
#############

print("平均値: %.3f \n中央値: %.3f \n分散  ：%.3f" %(mean, median, sd))