f = open("titanic.txt","r")
df = f.read()
df2 = df.split("\n")
df3 = [i.split(",") for i in df2]

del df3[-1]

for i in df3:
    print(i)

f.close()