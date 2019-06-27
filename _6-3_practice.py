from wordcloud import WordCloud
import MeCab
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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

img = Image.open('mask3.jpg')
plt.imshow(img)
plt.show()

with open("test.txt") as f:
    text = f.read()
mask = np.array(img)
text = get_words(text,["名詞"])
text = " ".join(text) ## <-WordCloudは単語は全部スペースで区切られることが必要
wordcloud = WordCloud(background_color="white",font_path="C:/WINDOWS/Fonts/UDDigiKyokashoN-B.ttc",
width=800,height=600,mask = mask,contour_width=2, contour_color='steelblue')
wordcloud.generate(text)

plt.imshow(wordcloud)
plt.show()

wordcloud.to_file('wordcloud.png')


