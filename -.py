import re

def million(i):
    if i == "million":
        return 10 ** 6
    else:
        return 1
    
def dollar_calculator(text):  
    pat = "\$([0-9.]*\d)\s*(million)*"
    res = re.findall(pat,text)
    print(res)
    s = 0
    for i in res:
        s += float(i[0]) * million(i[1])
    print("total %.2f" %s)
    
dollar_calculator(text)