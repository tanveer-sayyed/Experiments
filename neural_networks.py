import keras
"""
    CNN
"""
input_shape = (batch_size, height, width, depth)
model = keras.Sequential()
for i in range(3): # architecture with 3 convolution layers
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu",input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
# Flatten the output and slowly move towards output layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=64, activation="relu"))
model.add(keras.layers.Dropout(rate=0.3))
# output layer
model.add(keras.layers.Dense(units=3, activation="softmax")); model.summary()
# add optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
# compile
model.compile(optimizer=optimizer, loss="categorical_crossentropy",metrics=["accuracy"])
# fit
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)
# output will be of shape = input_shape
test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

"""
    LSTM
"""
