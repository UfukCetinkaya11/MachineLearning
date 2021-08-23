import pandas as pd
from Preprocess import Split_in_Train_and_Test
from Preprocess import convert
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.models import load_model
import seaborn as sns



df = pd.read_csv('icml_face_data.csv')



train_input,train_target,test_input,test_target = Split_in_Train_and_Test(df)
liste = [(train_input,True),(train_target,False),(test_input,True),(test_target,False)]
for z in range(len(liste)):
    liste[z] = convert(liste[z])
train_input,train_target,test_input,test_target= liste[0],liste[1],liste[2],liste[3]




temp = train_target

train_input = train_input.reshape(train_input.shape[0],48,48,1)
train_input = train_input/255
train_target = np_utils.to_categorical(train_target,num_classes=7)

test_input = test_input.reshape(test_input.shape[0],48,48,1)
test_input = test_input/255
test_target = np_utils.to_categorical(test_target,num_classes=7)

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

model = load_model('Saved_modelV3Augmented.h5')

#------------------------------------------- 3D Hyperparameters


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

x = [1,1,1,2,2,2,3,3,3]
y = [1,2,3,1,2,3,1,2,3]
z = [29.11,33.32,22.06,49.58,56.72,55.21,57.83,61.97,60.95]

ax.scatter(x,y,z, c='b', marker='o')
ax.set_xlabel('Filter Size')
ax.set_ylabel('Max pooling Size')
ax.set_zlabel('Accuracy in %')

plt.show()



#------------------------------------------- # Zeigt Bilder an


records = np.array([321,321,487,1,43,432,541,345,543,676,436])
plt.figure(figsize=[14,14])
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow((train_input[records, :, :, :])[i,:,:,0], cmap='binary_r')
    plt.title(emotions[(temp[records])[i]])
    plt.axis('off')
plt.show()
#-------------------------------------------------- Confusion Matrix

test_pred = model.predict(test_input)

conf_mat = confusion_matrix(np.argmax(test_target, axis=1), np.argmax(test_pred, axis=1))
arrayConf = []
sum_per_row = []
for x in range(len(conf_mat)):
    sum = 0
    for y in range(len(conf_mat[0])):
        sum += conf_mat[x][y]
    sum_per_row.append(sum)

for x in range(len(conf_mat)):
    arrayConf.append([])
    for y in range(len(conf_mat[0])):
        arrayConf[x].append(round(conf_mat[x,y]/sum_per_row[x],2))
conf_mat = np.array(arrayConf)


ax = sns.heatmap(conf_mat, annot=True,
                 xticklabels=["angr","disgus","fear","happy","sad","surpr","neutral"],
                 yticklabels=["angr","disgus","fear","happy","sad","surpr","neutral"],
                 cbar=False, cmap='Blues')
ax.set_xlabel("Predictions")
ax.set_ylabel("Actual")
plt.show()
plt.clf()




#------------------------------------------------------ table of amount of data per emotion
df_train = df[df[' Usage'] == 'Training']
df_test = df[df[' Usage'] == 'PrivateTest']
anger = df_train['emotion'].value_counts()[0]
happy = df_train['emotion'].value_counts()[3]
sad = df_train['emotion'].value_counts()[4]
fear = df_train['emotion'].value_counts()[2]
surprise = df_train['emotion'].value_counts()[5]
neutral = df_train['emotion'].value_counts()[6]
disgust = df_train['emotion'].value_counts()[1]

anger_test = df_test['emotion'].value_counts()[0]
happy_test = df_test['emotion'].value_counts()[3]
sad_test = df_test['emotion'].value_counts()[4]
fear_test = df_test['emotion'].value_counts()[2]
surprise_test = df_test['emotion'].value_counts()[5]
neutral_test = df_test['emotion'].value_counts()[6]
disgust_test = df_test['emotion'].value_counts()[1]

x_values = ["angr","surpri","neutr","disgu","happy","fear","sad"]
y_values = [anger,surprise,neutral,disgust,happy,fear,sad]
y_values_test = [anger_test,surprise_test,neutral_test,disgust_test,happy_test,fear_test,sad_test]

plt.subplot(1,2,1)
plt.xlabel("Emotions")
plt.ylabel("Quantity of images")
c = ['red', 'orange', 'grey', 'green','yellow','black','blue']
plt.bar(x_values,y_values,color = c)

plt.subplot(1,2,2)
plt.xlabel("Emotions")
plt.ylabel("Quantity of images")
c = ['red', 'orange', 'grey', 'green','yellow','black','blue']
plt.bar(x_values,y_values_test,color = c)

plt.show()
#----------------------------------




