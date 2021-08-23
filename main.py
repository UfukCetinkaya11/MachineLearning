from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
from Preprocess import Split_in_Train_and_Test
from Preprocess import convert
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt





df = pd.read_csv('icml_face_data.csv')

disgust_data_training = df[df[' Usage']=='Training']
disgust_data_training = disgust_data_training[disgust_data_training['emotion']== 1]
disgust_data_training = disgust_data_training.loc[disgust_data_training.index.repeat(10)]
frame = [df,disgust_data_training]
df = pd.concat(frame)
df = df.reset_index(drop=True)


train_input,train_target,test_input,test_target = Split_in_Train_and_Test(df)
liste = [(train_input,True),(train_target,False),(test_input,True),(test_target,False)]
for z in range(len(liste)):
    liste[z] = convert(liste[z])
train_input,train_target,test_input,test_target= liste[0],liste[1],liste[2],liste[3]


train_input = train_input.reshape(train_input.shape[0],48,48,1)
train_input = train_input/255
train_target = np_utils.to_categorical(train_target,num_classes=7)



test_input = test_input.reshape(test_input.shape[0],48,48,1)
test_input = test_input/255
test_target = np_utils.to_categorical(test_target,num_classes=7)



# data augementation

aug_train = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest")
generator_val = ImageDataGenerator()

aug_train.fit(train_input)
model = Sequential()

Filter_Size = 3
Num_Filters = 64
Input_Size = 48
MaxPool_Size = 2


model.add(Conv2D(64,(Filter_Size, Filter_Size),padding='same' ,input_shape=(48,48,1),activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(Filter_Size, Filter_Size),padding='same' ,activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MaxPool_Size,MaxPool_Size)))
model.add(Dropout(0.2))

model.add(Conv2D(Num_Filters,(Filter_Size, Filter_Size),padding='same',activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(Num_Filters,(Filter_Size, Filter_Size),padding='same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MaxPool_Size,MaxPool_Size)))
model.add(Dropout(0.2))

model.add(Conv2D(Num_Filters,(Filter_Size, Filter_Size),padding='same',activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(Num_Filters,(Filter_Size, Filter_Size),padding='same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MaxPool_Size,MaxPool_Size)))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(units = Num_Filters, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units = Num_Filters, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units = 7, activation='softmax'))

#model.summary()


model.compile(optimizer = Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#model.fit(train_input,train_target,batch_size=32,epochs=25,verbose=1,shuffle=True,validation_data=(test_input,test_target))
history = model.fit(aug_train.flow(train_input, train_target),
                   validation_data=generator_val.flow(test_input, test_target),
                    batch_size=16,epochs=30,verbose=1,shuffle=True)
model.save('Saved_modelV3Augmented.h5')


# Result - Accuracy
scores = model.evaluate(train_input,train_target, verbose=True)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(test_input,test_target, verbose=True)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))


# History of accuracy and loss

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs,accuracy, 'ro', label='training accuracy')
plt.plot(epochs,val_accuracy, 'r', label='testing accuracy')
plt.title('Training and testing accuracy')

plt.figure()

plt.plot(epochs,loss,'ro', label='training loss')
plt.plot(epochs,val_loss,'r', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()




