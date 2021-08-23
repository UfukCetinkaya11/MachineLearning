import pandas as pd
import numpy as np

df = pd.read_csv('icml_face_data.csv')
#print(df.dtypes)
#print(df.head(20))
#print(df.emotion)
#print(df[' Usage'])
#print(df.isnull().any())
#print(type(df.iloc[1][2]))
#print(df.count()[1])
#print((df.loc[:, df.columns != 'emotion']).iloc[0][1])

def convert(data):
    if(data[1] == True):
        for z in range(len(data[0])):
            data[0][z] = np.asarray([int(s) for s in data[0][z][0].split(' ')])
        data = np.asarray(data[0])
        return data
    elif(data[1] == False):
        for z in range(len(data[0])):
            data[0][z] = np.asarray(data[0][z])
        data = np.asarray(data[0])
        return data



def Split_in_Train_and_Test(data):
    train_input = []
    train_target = []
    test_input = []
    test_target = []
    x = data.loc[:, data.columns != 'emotion']
    y = data.emotion
    for z in range(data.count()[1]):
        if(data.iloc[z][1] == "Training"):
            train_input.append([x.iloc[z][1]])
            train_target.append(y[z])
        elif(data.iloc[z][1] == "PublicTest"):
            test_input.append([x.iloc[z][1]])
            test_target.append(y[z])



    return train_input,train_target,test_input,test_target

