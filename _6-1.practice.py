import MeCab
t = MeCab.Tagger()
node = t.parseToNode("pythonは今結構流行ってる。")
words = []
while node:
    #単語
    word = node.surface
    #名詞、形容詞など
    pos = node.feature.split(",")[0]
    if pos == "名詞":
        words.append(word)

        node = node.nextprint(words)
print(words)