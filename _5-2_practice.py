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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier


def simple_bar(dat,col):  #<-簡単なbarをプロット
    pass

def balance_data(x,y,ratio=1,random_seed=1): #<-データのバランスを解決
    """
    x, yを受け取って、ratioに従いバランスした、 x,yを return
    """
    np.random.seed(random_seed)
    tmp1 = y.loc[y == 1].index
    tmp2 = y.loc[y == 0].sample(tmp1.shape[0]*ratio).index
    ind = tmp1.union(tmp2)   #<- indexのappend
    return x.loc[ind,:], y.loc[ind]

def preprocessing(data,drop_variables,dependent_variable,change_word=None,normalization=False,train=0.9,balance=False, ratio=1, random_seed=1, debug = False):
    """
    :param dat          : 元の行列
    :param normalization: 標準化するかどうか
    :param train        : トレーニングセットの比率
    :param balance      : バランスデータにするかどうか
    :param ratio        : バランスの比率(>1だと 1が0より多い）
    :param random_seed  : seed
    :param debug        : debugのために一部うを出力
    :return             : 整備された [Trx, Try, Tx, Ty]
    """
    np.random.seed(random_seed)

    [X,Y] = drop_data(data,drop_variables,dependent_variable,change_word)

    if normalization:
        normalized(X,debug)

    [Trx, Tx, Try, Ty] = train_test_split(X,Y,test_size=1-train,random_state=random_seed,shuffle=False)

    if balance:
        Trx, Try = balance_data(Trx,Try,ratio,random_seed)
        if debug:
            print("After balancing")
            print(Try.head())

    return [Trx, Tx, Try, Ty]


def normalized(X,debug):
    X.iloc[:,:] = StandardScaler().fit_transform(X)
    if debug:
        print("After normalization\n",X)
    return X

def omitNA(data):
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

def simulation(coln):
    # 講義のような確率ヒートマップを書くためのシミュレーションデータ
    # 教育年数 0-20
	# 毎週仕事時間 0 - 40
	# 教育年数 -> 1 , 毎週仕事時間 -> 4 (列番号)
	simdata = np.zeros((600,coln))
	count = 0
	for edu in range(20):
		for work in range(30):
			simdata[count, 1] = edu
			simdata[count, 4] = work
			count += 1
	return simdata

def transition(data, column_name):
    """
    For exaple: column_name に"結婚状態"といれると"結婚状態"の列番と列名のリストを返す関数
    """
    col_num = []
    for i in range(len(data.columns)):
        if column_name in data.columns[i]:
            col_num.append(i)

    labels = data.iloc[:, col_num].columns
    return col_num, labels

def simulation_dummy(data_num,dummy_1,col_order_num=None,_range=None,dummy_2=None):
    """
    もし、ダミー変数と数値データでsimulationデータを生成したい場合、'col_order_num'と'_range'は必須
    Parameter
    :data_num: データの総数
    :dummy_1: カテゴリ変数1
    :col_order_num : データの列番号
    :_range: column's range
    :dummy_2: カテゴリ変数2
    """
    count = 0
    length_dummy1 = len(dummy_1)
    #dummy変数と数値でのsimdataの生成
    if dummy_2 == None:
        print("dummy変数と数値データでのsimulation dataを生成します")
        if col_order_num == None or _range == None:
            print("col_order_numもしくは_rangeがNoneです")
            print("col_order_numもしくは_rangeを設定してください")
        else:
            simdata = np.zeros((_range * length_dummy1, data_num))
            for i in range(_range):
                for j in range(length_dummy1):
                    simdata[count, col_order_num] = i
                    simdata[count, dummy_1[j]] = 1
                    count += 1
            return simdata
    #dummy変数とdummy変数でsimdataの生成
    else:
        print("dummy変数とdummy変数でsimulation dataを生成します")
        simdata = np.zeros((length_dummy1 * len(dummy_2), data_num))
        for i in range(length_dummy1):
            for j in range(len(dummy_2)):
                simdata[count, dummy_1[i]] = 1
                simdata[count, dummy_2[j]] = 1
                count += 1
        return simdata

def main():
    #　１．データを見る
    train = pd.read_csv("adult.csv")
    test = pd.read_csv("adult_test.csv")
    data = pd.concat([train,test],ignore_index=True)
    #  2.欠損値を抜く
    convert_cols = ["職種","学位","結婚状態","関係","職業","人種","性別","国籍"]
    drop_variables = ["fnlwgt", "income"]
    dependent_variable = "income"
    change_word = " >50K."

    data = omitNA(data)
    data = convert_to_dummy(data,convert_cols)

    #  3.カテゴリデータの処理 && 訓練、テストデータの準備

    Trx,Tx,Try,Ty = preprocessing(data,drop_variables,dependent_variable,change_word,True,0.95,True,3,5,False)

    #  4.提案モデル  #<- logistic? neural network?
    logistic = LogisticRegression()
    result = logistic.fit(Trx, Try)
    #NNT = MLPClassifier(hidden_layer_sizes=(400,200,100,50,1))
    #result = NNT.fit(Trx,Try)

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
    col_num_list,labels = transition(Trx,"職種")
    simdata = simulation_dummy(Trx.shape[1],col_num_list,1,20)
    heatmap(simdata)
    plt.show()
    #print("\n\nSimulation data\n",simdata[:,[1,4]])

    sim_pred =result.predict_proba(simdata)
    #print(sim_pred[:,1])
    sim_pred = np.array(sim_pred[:,1])
    ax =heatmap(sim_pred)
    ax.invert_yaxis()
    ax.set_title("Probability of income >=50K")
    ax.set_ylabel("Education year")
    ax.set_xlabel("Working Hour per week")
    plt.show()

if __name__ == "__main__":
    main()