import numpy as np
import pandas as pd
from seaborn import heatmap
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score



def simple_bar(dat,col):  #<-簡単なbarをプロット
    pass

def balance_data(x,y,ratio=1,random_seed=1): #<-データのバランスを解決
    """
    x, yを受け取って、ratioに従いバランスした、 x,yを return
    """
    tmp1 = y.loc[y == 1].index
    tmp2 = y.loc[y == 0].sample(tmp1.shape[0]*ratio).index
    ind = tmp1.union(tmp2)   #<- indexのappend
    return x.loc[ind,:], y.loc[ind]

def preprocessing(dat,convert_cols,drop_variables,dependent_variable,convert_to_num,normalization=False,train=0.9,balance=False, ratio= 1, random_seed=1, debug = False):
    """
    :param dat                  : 元の行列
    :param convert_cols         : ダミー変数に置き換えたい列名のリスト
    :param drop_variables       : 不要な列名のリスト
    :param dependent_variable   : 目的変数列名
    :param convert_to_num       : ダミー変数に変換
    :param normalization        : 標準化するかどうか
    :param train        : トレーニングセットの比率
    :param balance      : バランスデータにするかどうか
    :param ratio        : バランスの比率(>1だと 1が0より多い）
    :param random_seed  : seed
    :param debug        : debugのために一部を出力
    :return             : 整備された [Trx, Try, Tx, Ty]
    """
    np.random.seed(random_seed)
    #[Trx,Tx,Try,Ty] = [[],[],[],[]]

    #欠損値の削除
    naomit_data = convert_to_dummy(processing_NA(dat),convert_cols)
    #データの目的変数と説明変数に分割
    [X,Y] = drop_data(naomit_data,drop_variables,dependent_variable,convert_to_num)

    if normalization:
        X.iloc[:,:] = StandardScaler().fit_transform(X)
        if debug:
            print("After normalization\n",X)
    
    if train == 1:
        [Trx,Tx,Try,Ty] = [X,[],Y,[]]
    elif train == 0:
        [Trx,Tx,Try,Ty] = [[],X,[],Y]
    else:
        [Trx, Tx, Try, Ty] = train_test_split(X,Y,test_size=1-train,random_state=random_seed)
    
    #[Trx, Tx, Try, Ty] = train_test_split(X,Y,test_size=1-train,random_state=random_seed)

    if balance:
        Trx, Try = balance_data(X,Y,ratio,random_seed)
        if debug:
            print("After balancing")
            print(Try.head())

    return [Trx, Tx, Try, Ty]

def simulation(coln):
    #　講義のような確率ヒートマップを書くためのシミュレーションデータ
    simdata = []
    return simdata

def verify_NA(data):
    column_na = data.isnull().sum(axis = 0)
    row_na = data.isnull().sum(axis = 1)
    print("列毎の欠損値\n",column_na)
    print("行毎の欠損値\n",row_na.head(10))
    heatmap(data.isnull(),cmap="Greys")

def processing_NA(data):
    ind_na = data.isnull().any(axis=1)
    data = data.loc[ind_na==False,:]
    return data

def convert_to_dummy(dat,cols):
    ndat = pd.get_dummies(dat, columns=cols)
    return ndat

def drop_data(dat,drop_variables,dependent_variable,change_word=None):
    X = dat.drop(drop_variables, axis=1)
    Y = dat[dependent_variable]
    if change_word != None:
        Y.iloc[:] = np.array([1 if x in change_word else 0 for x in Y])
    return [X,Y]

def main():
    #　１．データを見る
    train = pd.read_csv("adult.csv")
    test = pd.read_csv("adult_test.csv")
    
    #  2.欠損値を抜く
    #verify_NA(train)
    
    #  3.カテゴリデータの処理 && 訓練、テストデータの準備
    convert_cols = ["職種","学位","結婚状態","関係","職業","人種","性別","国籍"]
    drop_variables = ["fnlwgt", "income"]
    dependent_variable = ["income"]
    change_word = [" >50K."]

    
    [Trx,a,Try,b] = preprocessing(train,convert_cols,drop_variables,dependent_variable,change_word,True,1,True,1,1,False)
    #[c,Tx,d,Ty] = preprocessing(test,convert_cols,drop_variables,dependent_variable,change_word,True,0,True,1,1,False)
    
    #  4.提案モデル  #<- logistic? neural network?
    logistic = LogisticRegression()
    result = logistic.fit(Trx, Try)

    pred = result.predict(Tx)
    cmat = confusion_matrix(Ty, pred)
    print(cmat)
    #   5.モデル評価 #<- accuracy, precision, recall, f1で評価
    evaluation = [accuracy_score,precision_score,recall_score,f1_score]
    evaluation_tag = ["accuracy","precision","recall","f1"]
    scores = []
    for i in range(4):
        scores.append(evaluation[i](Ty,pred))

    scores = pd.DataFrame(scores,index=evaluation_tag)
    print("Scores:\n",scores)

    ## 6.シミュレーションデータでヒートマップ

    #
 


if __name__ == "__main__":
    main()