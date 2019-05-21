import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


dat = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Icecream.csv",index_col = 0)
print(dat.head())

X = dat.iloc[:,1:]
Y = dat.iloc[:,0]
print("\nXの次元：", X.shape,"\nYの次元：",Y.shape)

my_linear = LinearRegression()
result = my_linear.fit(X,Y)
pred = result.predict(X)

print(pred)

x = range(len(Y))

plt.plot(x,Y,color= "red")
plt.plot(x,pred,"--", color = "blue")
plt.legend(["Data", "Estimate"])
plt.title("Result")
plt.show()

