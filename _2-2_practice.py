f = open("titanic.txt","r")
df = []
for line in f.readlines():
    df.append(line.rstrip().split(","))

for i in df:
    print(i)