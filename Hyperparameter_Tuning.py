import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Preprocess import Split_in_Train_and_Test
from Preprocess import convert
from sklearn.model_selection import StratifiedKFold



df = pd.read_csv('icml_face_data.csv')
disgust_data_training = df[df[' Usage']=='Training']
disgust_data_training = disgust_data_training[disgust_data_training['emotion']== 1]
disgust_data_training = disgust_data_training.loc[disgust_data_training.index.repeat(10)]
frame = [df,disgust_data_training]
df = pd.concat(frame)
df = df.reset_index(drop=True)


train_input,train_target,test_input,test_target= Split_in_Train_and_Test(df)
liste = [(train_input,True),(train_target,False),(test_input,True),(test_target,False)]
for z in range(len(liste)):
    liste[z] = convert(liste[z])
train_input,train_target,test_input,test_target= liste[0],liste[1],liste[2],liste[3]


train_input = train_input.reshape(train_input.shape[0],48,48,1)
train_input = train_input/255
target = np_utils.to_categorical(train_target,num_classes=7)

test_input = test_input.reshape(test_input.shape[0],48,48,1)
test_input = test_input/255
test_target = np_utils.to_categorical(test_target,num_classes=7)

aug_train = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
generator_val = ImageDataGenerator()

aug_train.fit(train_input)




kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
cvscores = []

hyperpara_conv_filter_size = [4,5]
hyperpara_maxPooling_size = [4,5]


for hyp1 in range(len(hyperpara_conv_filter_size)):
    for hyp2 in range(len(hyperpara_maxPooling_size)):
        for train, test in kfold.split(train_input, train_target):
            model = Sequential()

            Filter_Size = hyperpara_conv_filter_size[hyp1]
            Num_Filters = 64
            Input_Size = 48
            MaxPool_Size = hyperpara_maxPooling_size[hyp2]

            model.add(Conv2D(64, (Filter_Size, Filter_Size), padding='same', input_shape=(48, 48, 1), activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (Filter_Size, Filter_Size), padding='same', activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(MaxPool_Size, MaxPool_Size),padding='same'))
            model.add(Dropout(0.2))

            model.add(Conv2D(Num_Filters, (Filter_Size, Filter_Size), padding='same', activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(Num_Filters, (Filter_Size, Filter_Size), padding='same', activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(MaxPool_Size, MaxPool_Size),padding='same'))
            model.add(Dropout(0.2))

            model.add(Conv2D(Num_Filters, (Filter_Size, Filter_Size), padding='same', activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(Num_Filters, (Filter_Size, Filter_Size), padding='same', activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(MaxPool_Size, MaxPool_Size),padding='same'))
            model.add(Dropout(0.2))

            model.add(Flatten())

            model.add(Dense(units=Num_Filters, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=Num_Filters, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=7, activation='softmax'))

            #model.summary()

            model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(aug_train.flow(train_input[train], target[train]),
                      batch_size=16, epochs=25, verbose=0, shuffle=True)

            # Results - Accuracy
            scores = model.evaluate(train_input[test], target[test], verbose=False)
            print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
            cvscores.append(scores[1] * 100)
        print(str(hyp1+4)+str(hyp2+4)+":","%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        cvscores = []
        print("-------------------------------")




