import re 

text = "This product is about $30.15, and the others are $0.05, $9.55 million. The last one is $5. It is 4 pm now."

def million(i):
    if i == "million":
        return 10 ** 6
    else:
        return 1

def distruct_doller(text):
    pattarn = re.compile(r"\$([0-9.]*\d)\s*(million)*")

    distruct = re.findall(pattarn,text)

    print(distruct)



    


#print("The total cost is $ %i")