#pandasのread_csvと同じ基本機能を持つ関数を作成してください。
#open() , read(), readlines(), pd.Dataframe()などを使い,
#pd.read_csv("titanic.csv")と同じ結果が得られることが望ましい。


def my_read_csv(x):
	import pandas as pd

	dat = []
	with open(x,"r") as f:
		for line in f.readlines():
			dat.append(line.rstrip().split(","))
	
	_columns = dat[0]
	_index = []
	for i in range(len(dat)-1):
		_index.append(i+1)

	del dat[0]

	mat = pd.DataFrame(dat,columns=_columns,index=_index).iloc[0:,1:]

	return mat

def main():
	path = "titanic.txt"
	df = my_read_csv(path)
	print(df.head())

if __name__ == "__main__":
	main()