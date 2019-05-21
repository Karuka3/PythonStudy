#まずtxtファイルをダウンロードします。
#txtファイルを読み込み、[]にある数字の数をカウントしてください。
#例: input -> "[] [123] [abc] [1ab3] [5] 123 [345 56] 1]"
#    output-> 2

# 次の問題に備え、最後にこのコードを"Problem3_1.py"として保存する。
import re

def my_readfile(path):
	with open(path,"r",encoding="utf-8") as f:
		text = f.read()
	return preprocessing(text) #<-前処理
	

def preprocessing(text):
	### Code 文章の\nを全部取る
	cleaned_text = re.subn(r'\n','',text)[0]
	#print(cleaned_text)
	return cleaned_text

def counter(cleaned_text):
	pattarn = re.compile(r"(?<=\[)\d*(?=\])")
	distruct = re.findall(pattarn,cleaned_text)
	#print(distruct)
	count = len(distruct)
	return count

def main():
	cleaned_text = my_readfile("wikipedia_ubc.txt")  #<- txtファイルのパス
	count = counter(cleaned_text)
	print("合計で{}個の数字が[]の中にある。".format(count))

if __name__ == "__main__":
	main()
	