import MeCab
text = "Pythonは今流行っている"

def get_words(text,conditions):
    t = MeCab.Tagger()
    prs = t.parse(text)
    #print(prs)
    prs2 = [line.split("\t") for line in prs.split("\n")[:-2]]
    #print(prs2)
    words=[]
    for _list in prs2:
        if _list[1].split(",")[0] in conditions:
            words.append(_list[0])
    return words

words = get_words(text,["名詞","動詞"])
print(words)
