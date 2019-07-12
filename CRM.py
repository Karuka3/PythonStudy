import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def rank_rfm(series, n):
    edges = pd.Series([float(i)/n for i in range(n + 1)])
    f = lambda x: (edges >= x).values.argmax()
    return series.rank(pct=1).apply(f)

def tukey_hsd(index, *args ):
    '''
    多重比較
    第1引数:名称のリスト（index）
    第2引数以降: データ (*args: 複数の引数をタプルとして受け取る)
    '''

    data_array = np.hstack( args )
    index_array = np.array([])
    for x in range(len(args)):
        index_array = np.append(index_array, np.repeat(index[x], len(args[x]))) # np.repeat(A,N)は配列A内の各要素をN回繰り返す
    print(pairwise_tukeyhsd(data_array,index_array))

def main():
    df = pd.read_csv('C:/Users/Kazuki/data/OnlineRetail.csv')
    # RFM分析できるようにデータを前処理
    now = datetime.date(2011, 12, 31)
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y/%m/%d')
    df['InvoiceDate'] = df['InvoiceDate'].dt.date

    rfm_table = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: now - x.max(),  # Recency
                                              'CustomerID': lambda x: len(x),             # Frequency
                                              'Sales': lambda x: x.sum()})                # Monetary Value

    rfm_table['CustomerID'] = rfm_table['CustomerID'].astype(int)
    rfm_table.rename(columns={'InvoiceDate': 'Recency','CustomerID': 'Frequency','Sales': 'Monetary'}, inplace=True)


    # 記述統計
    print(rfm_table.describe())


    # Rank付け（顧客の分類）
    rfm_table['R_Rank'] = 6 - rank_rfm(rfm_table['Recency'], 5)
    rfm_table['F_Rank'] = rank_rfm(rfm_table['Frequency'], 5)
    rfm_table['M_Rank'] = rank_rfm(rfm_table['Monetary'], 5)
    rfm_table['RFMClass'] = rfm_table.R_Rank.map(str) + rfm_table.F_Rank.map(str) + rfm_table.M_Rank.map(str)

    # Rのランク毎に累積購買金額での平均
    rec_rank_m = rfm_table.groupby(['R_Rank'])['Monetary'].mean()
    rec_rank_m.plot(kind='bar')
    plt.title("average")
    plt.xlabel("R_Rank")
    plt.ylabel("Monetary")
    plt.show()

    # Rのランク毎に累積購買金額の平均
    rec_rank_f = rfm_table.groupby(['R_Rank'])['Frequency'].mean()
    rec_rank_f.plot(kind='bar')
    plt.title("average")
    plt.xlabel("R_Rank")
    plt.ylabel("Frequency")
    plt.show()

    # 各項目毎にRankによってグループ分け
    R_group = rfm_table.groupby('R_Rank')
    print(R_group.describe()['Frequency'])
    print(R_group.describe()['Monetary'])

    F_group = rfm_table.groupby('F_Rank')
    print(F_group.describe()['Monetary'])

    RM_rank1 = R_group.get_group(1)['Monetary']
    RM_rank2 = R_group.get_group(2)['Monetary']
    RM_rank3 = R_group.get_group(3)['Monetary']
    RM_rank4 = R_group.get_group(4)['Monetary']
    RM_rank5 = R_group.get_group(5)['Monetary']

    RF_rank1 = R_group.get_group(1)['Frequency']
    RF_rank2 = R_group.get_group(2)['Frequency']
    RF_rank3 = R_group.get_group(3)['Frequency']
    RF_rank4 = R_group.get_group(4)['Frequency']
    RF_rank5 = R_group.get_group(5)['Frequency']

    # 分散分析
    #print(st.shapiro(R_group)) # Shapiro-Wilk検定(統計量，p値)

    f, p = st.f_oneway(RM_rank1,RM_rank2,RM_rank3,RM_rank4,RM_rank5)
    print("RecencyとMonetaryでの" + "F=%f, p-value = %f"%(f,p))
    tukey_hsd(list('123456789'), RM_rank1,RM_rank2,RM_rank3,RM_rank4,RM_rank5)

    f, p = st.f_oneway(RF_rank1,RF_rank2,RF_rank3,RF_rank4,RF_rank5)
    print("RecencyとFrequencyでの" + "F=%f, p-value = %f"%(f,p))
    tukey_hsd(list('123456789'), RF_rank1,RF_rank2,RF_rank3,RF_rank4,RF_rank5)

    # R_Rank毎の累積購買金額の分布
    sns.boxplot(x="R_Rank", y="Monetary", data=rfm_table)
    plt.grid()
    plt.ylim([0,10000])
    plt.show()

    # R_Rank毎の累積購買回数
    sns.boxplot(x="R_Rank", y="Frequency", data=rfm_table)
    plt.grid()
    plt.ylim([0,500])
    plt.show()
    # F_Rank事の累積購買金額
    sns.boxplot(x="F_Rank", y="Monetary", data=rfm_table)
    plt.grid()
    plt.ylim([0,10000])
    plt.show()

if __name__ == '__main__':
    main()