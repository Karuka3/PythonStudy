# 先程のファイルを読み込み、$がついている数字の合計を計算してください。
# 1) ただし、前や後ろに CADがついているのは除外します。
# 2) 20-M 20 million 10 billion	など百万や10億の単位もちゃんと数字に直してください
# 例 "$700,000.01 ... CAD $20.05 ... 50 ... $1-M, ... $50.05 CAD"
# Output -> $1700000.01


#　先程保存したProblem3_1.pyの同じフォルダに, Problem3_2.pyを作ります。(空っぽでもいい）
# ３－１でtxtファイルを読み込み、前処理のコードを作りましたが、今回もその関数を使いたいのですが、
# もう一度書くのは効率的ではありません。
# そこで、前に自分が書いた関数を呼び出して使います。

# 以下の通り自分が作った関数が入ったpythonファイルの名前(.pyは不要)から直前の読み込みが可能です。
from Problem3_1 import my_readfile, preprocessing
import re

def unit_convert_num(i):
	if i == "million" or "M" or "-million":
		return 10 ** 6
	elif i == "billion":
		return 10 ** 9
	else:
		return 1

def dollar_calculator(text):
	pattarn = re.compile(r"(?<!CAD)\$([0-9.]*\d)\s*(million|billion|M|-million)*")
	_eliminated = re.subn(',','',text)
	distruct = re.findall(pattarn, _eliminated[0])
	
	print(distruct)
	s = 0
	for i in distruct:
		s += float(i[0]) * unit_convert_num(i[1])
	return s
	
def main():
	cleand_text = my_readfile("wikipedia_ubc.txt") #<-　このまま自分が前に作った関数を引用
	s = dollar_calculator(cleand_text)
	print("Total %.2f" %s)
	
if __name__ == "__main__":
	main()