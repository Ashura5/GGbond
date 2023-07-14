import os
import json
import pandas as pd
from sklearn.impute import SimpleImputer

def process_1(directory,project,file):
    if not os.path.exists(os.path.join(directory+project, file)):
        return pd.DataFrame()
    with open(os.path.join(directory+project, file), 'r') as f:
        data = json.load(f)
        if len(data)==0:
             return pd.DataFrame()
        temp_df = pd.DataFrame.from_dict(data, orient='index')
        temp_df.columns = [file[:-5]]
    return temp_df
def process_2(directory,project,file):
    df=pd.DataFrame()
    if not os.path.exists(os.path.join(directory+project, file)):
        return pd.DataFrame()
    with open(os.path.join(directory+project, file), 'r') as f:
        data = json.load(f)
        for key in data:
            if len(data[key])==0:
                continue
            temp_df = pd.DataFrame.from_dict(data[key], orient='index')
            if key=="levels":
                temp_df.columns = [str(file[:-5])+"_"+key+"_0",str(file[:-5])+"_"+key+"_1",str(file[:-5])+"_"+key+"_2",str(file[:-5])+"_"+key+"_3"]
            else:
                temp_df.columns = [str(file[:-5])+"_"+key]
            if df.empty:
                df = temp_df
            else:
                df = df.join(temp_df, how='outer')
    return df

def get_data(directory,projects,class_1,class_2):
    df_list=[]
    for project in projects:
        df = pd.DataFrame()
        for file in class_1:
            temp_df=process_1(directory,project,file)
            if df.empty:
                df = temp_df
            else:
                df = df.join(temp_df, how='outer')
        for file in class_2:
            temp_df=process_2(directory,project,file)
            if df.empty:
                df = temp_df
            else:
                df = df.join(temp_df, how='outer')
        if df.shape[0]>12:
            df_list.append(df.fillna(0))
    return df_list

def data_process(df_list):
    # 将所有的 DataFrame 合并为一个
    df_all = pd.concat(df_list)

    # 删除具有特定索引的行
    df_all = df_all.drop('2021-10-raw', errors='ignore')

    # 将日期转换为特征
    df_all.index = pd.to_datetime(df_all.index, errors='coerce')
    df_all['year'] = df_all.index.year
    df_all['month'] = df_all.index.month

    # 将数据分为特征和目标变量
    X_all = df_all.drop('openrank', axis=1)
    y_all = df_all['openrank']
    feature_names = X_all.columns
    # 将数据分为训练集和测试集
    # 在这个例子中，我们使用2015年到2022年的数据作为训练集，2023年的数据作为测试集
    X_train_all = X_all[X_all['year'] < 2023]
    y_train_all = y_all[X_all['year'] < 2023]

    imputer = SimpleImputer(strategy='mean')  
    X_train_all = imputer.fit_transform(X_train_all)

    all_features = df_all.columns
    X_test_all = []
    y_test_all = []

    for df in df_list:
        # 删除具有特定索引的行
        df = df.drop('2021-10-raw', errors='ignore')

        # 将日期转换为特征
        df.index = pd.to_datetime(df.index, errors='coerce')
        df['year'] = df.index.year
        df['month'] = df.index.month

        # 确保 DataFrame 包含所有可能的特征
        for feature in all_features:
            if feature not in df.columns:
                df[feature] = 0

        # 确保特征的顺序与训练数据一致
        df = df[all_features]

        # 将数据分为特征和目标变量
        X = df.drop('openrank', axis=1)
        y = df['openrank']

        # 选择2023年的数据作为测试集
        X_test = X[X['year'] == 2023]
        y_test = y[X['year'] == 2023]
        X_test = imputer.transform(X_test)
        # 在测试集上进行预测
        X_test_all.append(X_test)
        y_test_all.append(y_test)
    return X_train_all,y_train_all,X_test_all,y_test_all,feature_names,all_features
