import re
text = "apple asian africa cat ball application apply a .com"

pattarn1 = re.compile(r"a\w+")
pattarn2 = re.compile(r"a\w*")
pattarn3 = re.compile(r"app\w*")
pattarn4 = re.compile(r"\.com")

pattarn = [pattarn1,pattarn2,pattarn3,pattarn4]

for i in range(4):
    print("Pattarn %i:" % (i+1),re.findall(pattarn[i],text))
