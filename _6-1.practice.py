import MeCab
t = MeCab.Tagger("-Ochasen")
print(t.parse("こいつは強いよ"))
