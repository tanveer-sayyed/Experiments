import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

classif = Sequential()
classif.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classif.add(MaxPooling2D(pool_size = (2, 2)))
classif.add(Conv2D(32, (3, 3), activation = 'relu'))
classif.add(MaxPooling2D(pool_size = (2, 2)))
classif.add(Flatten())
classif.add(Dense(units = 128,
                     activation = 'relu'))
classif.add(Dense(units = 1, activation = 'sigmoid'))
classif.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
train = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip=False)
test = ImageDataGenerator(rescale = 1./255)
train_ = train.flow_from_directory('dataset/NavClothes/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_ = test.flow_from_directory('dataset/Navclothes/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classif.fit_generator(train_,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_,
                         validation_steps = 2000)

#predict
test_image = image.load_img('dataset/NavClothes/Test/test_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
classifier.predict(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
print('Cloth is of colour: {}'.format('Green' if result[0][0]==1 else 'Red'))
