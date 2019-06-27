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
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV


def simple_bar(dat,col):  #<-簡単なbarをプロット
    balance = dat[col].value_counts()
    plt.clf()
    plt.bar(balance.index, balance)
    plt.title(col)
    plt.show()

def balance_data(x,y,ratio=1,random_seed=1): #<-データのバランスを解決
    """
    x, yを受け取って、ratioに従いバランスした、 x,yを return
    """

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

def simulate_dummy(coln, col_number, col_val, dummy):  # edu(0~20)とmarriageの関係、 workとmarriageの関係
    """
    :coln:
    :col:
    :col_val:
    :dummy: ダミー変数の列のリスト
    """
    simdata = np.zeros((col_val * len(dummy), coln))  # col には edu, work の列番号、col_valには範囲(ex,edu=0~20)
    count = 0
    for i in range(col_val):
        for j in range(len(dummy)):
            simdata[count, col_number] = i
            simdata[count, dummy[j]] = 1
            count += 1
    return simdata

def simulate_dummy_and_dummy(coln, dummy_1, dummy_2):  # marriageとdegreeの関係
    """
    :coln :　
    :dummy_1 :1つ目のダミー変数の列番号のリスト
    :dummy_2 :2つ目のダミー変数の列番号のリスト
    """
    simdata = np.zeros((len(dummy_1) * len(dummy_2), coln))
    count = 0
    for i in range(len(dummy_1)):
        for j in range(len(dummy_2)):
            simdata[count, dummy_1[i]] = 1
            simdata[count, dummy_2[j]] = 1
            count += 1
    return simdata

def make_heatmap(data,result_model,column,simdata,title,xlabel,ylabel,number_of_simdata):
    column_num_list,labels = transition(data,column)
    heatmap(simdata)
    plt.show()

    sim_pred =result_model.predict_proba(simdata)
    sim_pred = np.array(sim_pred[:,1].reshape(number_of_simdata,len(column_num_list)))

    ax =heatmap(sim_pred,xticklabels=labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.show()

def main():
    #　１．データを見る
    train = pd.read_csv("adult.csv")
    test = pd.read_csv("adult_test.csv")
    data = pd.concat([train,test],ignore_index=True)
    #simple_bar(data,"income")
    #simple_bar(data,"職種")
    #  2.欠損値を抜く
    convert_cols = ["職種","学位","結婚状態","関係","職業","人種","性別","国籍"]
    drop_variables = ["fnlwgt", "income"]
    dependent_variable = "income"
    change_word = " >50K."

    data = omitNA(data)

    data = convert_to_dummy(data,convert_cols)

    #  3.カテゴリデータの処理 && 訓練、テストデータの準備

    Trx,Tx,Try,Ty = preprocessing(data,drop_variables,dependent_variable,change_word,True,0.86,True,2,1,False)

    #  4.提案モデル  #<- logistic? neural network?

    logistic = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.8)
    result = logistic.fit(Trx, Try)
    #SVM = SVC(C=100,probability=True,random_state=1)
    #result = SVM.fit(Trx,Try)

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
    profession,profession_labels = transition(Trx,"関係")

    simdata_1 = simulate_dummy(Trx.shape[1],4,30,profession)
    make_heatmap(Trx,result,"関係",simdata_1,title="Probability of income >=50K",xlabel="Profession",ylabel="Working per week",number_of_simdata=30)
    simdata_2 = simulate_dummy(Trx.shape[1],1,20,profession)
    make_heatmap(Trx,result,"関係",simdata_2,title="Probability of income >=50K",xlabel="Profession",ylabel="Education year",number_of_simdata=20)

    #print("\n\nSimulation data\n",simdata[:,[1,profession[0]]])
    #print(sim_pred[:,1])

if __name__ == "__main__":
    main()