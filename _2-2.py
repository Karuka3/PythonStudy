##　上の問題を解いてください。


def my_calculation(x):
	import numpy as np

	A = np.array(x)
	lamda, P = np.linalg.eig(A)
	_lamda = np.sqrt(lamda)
	_P = np.linalg.inv(P)

	mat = np.dot((P*_lamda),_P)
	return mat

def main():
	a = [[3,4,1,4],[1,2,1,1],[1,1,2,1],[1,1,1,2]]
	print(my_calculation(a))

if __name__ == "__main__":
	main()