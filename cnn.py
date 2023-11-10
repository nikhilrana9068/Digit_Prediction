import tensorflow as tf
import matplotlib.pyplot as plt


(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

# print(X_train.shape)
# print(X_test.shape)

# print(y_train[1:6])

# for x in range(1,6):
#     plt.subplot(1,5,x)
#     plt.imshow(X_train[x,:,:])
    
# plt.show()
import numpy as np

X_train= np.asarray(X_train,'float32')/255
X_test= np.asarray(X_test,'float32')/255

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
in_shape=X_train.shape[1:]

model=tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))

model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation='relu',input_shape=(in_shape,)))
model.add(tf.keras.layers.Dense(30,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,batch_size=32,verbose=2) 

_,acc=model.evaluate(X_test,y_test)

print(acc)

image = X_test[0]
y_pred = model.predict(np.asarray([image]))


cls=np.argmax(y_pred)

print("Digit Prediced:",cls)