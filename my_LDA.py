import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle as pkl
import MeCab
import re

def get_word(text, conditions):
    t = MeCab.Tagger()
    text = re.subn(r"\W", "", text)[0]  # 正規表現を用いて()や絵文字を抜く
    tmp = t.parse(text)
    tmp2 = [line.split("\t") for line in tmp.split("\n")][:-2]
    words = []
    for i in tmp2:
        if i[1].split(",")[0] in conditions:
            words.append(i[0])
    return words

def my_LDA(docs, K=5, Iter=1000, alpha=0.1, beta=0.1,trace=False):
    np.random.seed(1000)

    def sampling(ndk, nkv, nd, nk, i, v):
        probs = np.zeros(K)
        for k in range(K):
            prob = (ndk[i][k] + alpha)*(nkv[k][v] + beta) / ((nd[i] + K * alpha)*(nk[k] + M * beta))
            probs[k] = prob
        probs /= probs.sum()
        return np.argmax(np.random.multinomial(1,probs))

    ## 1. すべての単語にランダムに初期値を与える

    topics = [[np.random.randint(K) for w in d] for d in docs]

    wordmove = []
    V = len(word2num)
    M = len(docs)

    ndk = np.zeros((M, K))  # <-　文章のトピック分布
    nkv = np.zeros((K, V))  # <-　トピックの単語分布

    for i,d in enumerate(topics):
        for j,z in enumerate(d):
            ndk[i,z] += 1
            nkv[z,ndocs[i][j]] += 1

    nd = ndk.sum(axis=1)
    nk = nkv.sum(axis=1)

    ##
    for ite in tqdm(range(Iter)):
        count = 0
        for i, d in enumerate(topics):  # Every Documents
            for j, k in enumerate(d):  # Every word and topics
                ##　事前にサンプリング中の単語を集計から抜く
                v = docs[i][j]
                ndk[i, k] -= 1
                nkv[k, v] -= 1
                nk[k] -= 1
                ## サンプリング
                new_z = sampling(ndk, nkv, nd, nk, i, v)

                ##　新たな結果の元に、再集計
                topics[i][j] = new_z
                ndk[i, new_z] += 1
                nkv[new_z, v] += 1
                nk[new_z] += 1

    save = {"topics": topics, "nkv": nkv, "ndk": ndk, "nk": nk, "nd": nd}

    if trace:
        # print("word move",wordmove)
        plt.xlabel("Iteration")
        plt.ylabel("number of renewal topics")
        plt.plot(wordmove)
        plt.show()

    return save


comments = pd.read_csv("iphone7.csv", encoding="cp932")
print(comments.head())
docs = [get_word(comment, ["名詞", "動詞", "形容詞"]) for comment in comments["comment"]]

word2num = dict()
num2word = dict()
count = 0
for d in docs:
    for w in d:
        if w not in word2num.keys():
            word2num[w] = count
            num2word[count] = w
            count += 1
ndocs= [[word2num[w] for w in d ] for d in docs]

result = my_LDA(ndocs, K=5,Iter = 100,trace=True)

print("After sampling\n",result["topics"])