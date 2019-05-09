def my_read_csv(x):
	import pandas as pd
	dat = []
	with open(x,"r") as f:
		_columns = (f.readline()).rstrip().split(",")
		for line in f.readlines():
			dat.append(line.rstrip().split(","))
	mat = pd.DataFrame(dat,columns=_columns)
	mat = mat.set_index("")
	return mat

def main():
	path = "titanic.txt"
	df = my_read_csv(path)
	print(df.head())

if __name__ == "__main__":
	main()