##目的：
#１．予測したいのがmdu(医者に訪問する外来患者数),他のが説明変数
#２．複数のモデルから一番良いモデルを選択
#３．モデル選択の指標はテストデータに対するRMSE
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

##
#必要なパッケージは全部上で読み込む
##

#データの説明：https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DoctorContacts.html

def my_model(Xtr,Ytr,Xt,Yt):   #<-訓練ｘ，ｙ、テストｘ，ｙ
    #いくつかの提案モデルを作る
    model1 = LinearRegression()
    model2 = Lasso()
    model3 = MLPRegressor(hidden_layer_sizes=(10,5,1),max_iter=1000,random_state=1)
    model4 = MLPRegressor(hidden_layer_sizes=(100,10,5),max_iter=1000,random_state=1)
    model5 = RandomForestRegressor(n_estimators=10)
    model6 = RandomForestRegressor(n_estimators=20)
    model7 = RandomForestRegressor(n_estimators=30)
    model8 = RandomForestRegressor(n_estimators=50)

    #...hyperparameterの違いによって10個くらい違うパターンの提案モデル
    models = [model1,model2,model3,model4,model5,model6,model7,model8]
    rmses = []  #<-ここにすべてのRMSEを保管

    for model in models:
        result = model.fit(Xtr,Ytr)

        y_pred = result.predict(Xt)  #<- テストデータのXでYを予測
        rmses.append(rmse(Yt, y_pred))    #<- 平均二乗誤差の平方根

    return rmses

def rmse(y,y_pred):
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return rmse

def convert_to_dummy(label):
    return [1 if x == True or x == "male" else 0 for x in label]


def main():
    dat = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/DoctorContacts.csv",index_col=0)
    print("次元:",dat.shape)
    print("変換前\n")
    print(dat.head())

    ## データを分割&変換
    change_labels = ["idp","physlim","sex","child","black"]

    for label in change_labels:
        dat[label] = convert_to_dummy(dat[label])

    print(dat)
    dat = pd.get_dummies(dat,'health',drop_first=True)


    print("\n変換後\n")
    print(dat.head())

    ##　前の15,000個のデータを訓練に、残りをテスト

    #下のコードを基に修正、補完
    Xtr = dat.iloc[:15000,1:]
    Ytr = dat.iloc[:15000,0]
    Xt  = dat.iloc[15000:,1:]
    Yt  = dat.iloc[15000:,0]

    print(my_model(Xtr,Ytr,Xt,Yt))


if __name__ == "__main__":
    main()