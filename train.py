from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D ,MaxPooling2D,Dense,Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
train_path="dataset/train"
test_path="dataset/test"
train_data=ImageDataGenerator(rescale=1./255,rotation_range=30,zoom_range=0.2)
test_data=ImageDataGenerator(rescale=1./255)
train_generator=train_data.flow_from_directory(train_path,target_size=(244,244),batch_size=20,class_mode="categorical")
test_generator=test_data.flow_from_directory(test_path,target_size=(244,244),batch_size=20,class_mode="categorical")
print(train_generator.class_indices)
print(train_generator.image_shape)
model=Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(244,244,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation="relu",input_shape=(244,244,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation="relu",input_shape=(244,244,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256,(3,3),activation="relu",input_shape=(244,244,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense((128),activation="relu"))
model.add(Dense(9,activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
model.summary()
early_stop=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
history=model.fit(train_generator,validation_data=test_generator,epochs=20,callbacks=early_stop)
model.save("fruits.keras")
print("model saved")