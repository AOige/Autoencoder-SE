
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

train_df = pd.read_csv('/content/gdrive/MyDrive/UCI_HAR/train.csv')
test_df = pd.read_csv('/content/gdrive/MyDrive/UCI_HAR/test.csv')

train_df.head()

train_df = train_df.drop(['angle(tBodyAccMean,gravity)', 'angle(tBodyAccJerkMean),gravityMean)', 'angle(tBodyAccJerkMean),gravityMean)'], axis=1)

test_df = test_df.drop(['angle(tBodyAccMean,gravity)', 'angle(tBodyAccJerkMean),gravityMean)', 'angle(tBodyAccJerkMean),gravityMean)'], axis=1)

print('train set missing data:', train_df.isna().sum().sum())

train_df['Activity'].value_counts()

train_y = train_df['Activity']
train_X = train_df.drop('Activity', axis=1)

test_y = test_df['Activity']
test_X = test_df.drop('Activity', axis=1)

##Encoding Labels

label_encoder = LabelEncoder()
label_encoder.fit(train_y)

train_y = label_encoder.transform(train_y)
test_y = label_encoder.transform(test_y)

scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_X.shape, test_X.shape

from tensorflow.keras.utils import to_categorical
train_y = to_categorical(train_y, 6)
test_y = to_categorical(test_y, 6)
train_y.shape, test_y.shape

train_X = np.expand_dims(train_X, axis=-1)
test_X = np.expand_dims(test_X, axis=-1)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, MaxPool1D, Flatten, Dropout, UpSampling1D, Reshape, GlobalAveragePooling2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from keras.layers import GlobalAveragePooling1D, AvgPool1D

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def SqueezeAndExcitation(inputs, ratio=8):
    b, c = (560, 1)
    x = GlobalAveragePooling1D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(inputs)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = Multiply()([inputs, x])
    return x

def build_autoencoder(shape, dim=128):
    inputs = Input(shape)

    x = inputs
    num_filters = [256, 128, 64, 32]
    kernel_size = [11, 9, 7, 5]
    for i in range(len(num_filters)):
        nf = num_filters[i]
        ks = kernel_size[i]

        x = Conv1D(nf, ks, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool1D((2))(x)
    
    b, f, n = x.shape
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Reshape((512,1), input_shape=(512,))(x)
    x = SqueezeAndExcitation(x)
    x = Flatten()(x)
    latent = Dense(128, activation="linear", name="LATENT")(x)
    x = Dense(512, activation="relu")(latent)
    x = Dense(f*n, activation="relu")(x)
    x = Reshape((f, n))(x)
    
    
    num_filters = [32, 64, 128, 256]
    kernel_size = [5, 7, 9, 11]
    
    for i in range(len(num_filters)):
        nf = num_filters[i]
        ks = kernel_size[i]
        
        x = UpSampling1D((2))(x)
        x = Conv1D(nf, ks, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
    x = Conv1D(shape[1], 1, padding="same")(x)
    
    model = Model(inputs, x)
    return model

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Create a folder to save files """
create_dir("files")

""" Hyperparameters """
batch_size = 32
num_epochs = 20
input_shape = (560, 1)
num_classes = 6
latent_dim = 128
lr = 1e-4

model_path = "files/model_autoencoder.h5"
csv_path = "files/log_autoencoder.csv"

""" Dataset """
print(f"Train: {train_X.shape}/{train_y.shape} - Test: {test_X.shape}/{test_y.shape}")

""" Adding noise to training dataset """
mu, sigma = 0, 0.1  
noise = np.random.normal(mu, sigma, train_X.shape)

train_X1 = train_X + noise

""" Model & Training """
autoencoder = build_autoencoder(input_shape, dim=latent_dim)
autoencoder.summary()
adam = tf.keras.optimizers.Adam(lr)
autoencoder.compile(loss='mse', metrics=['accuracy'], optimizer=adam)
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]
autoencoder.fit(train_X1, train_X,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks = callbacks
    )

plt.('latent')
plt.show()

def build_classifier(autoencoder, num_classes=12):
    inputs = autoencoder.input
    outputs = autoencoder.get_layer("LATENT").output
    x = Dense(num_classes, activation="softmax")(outputs)
    
    model = Model(inputs, x)
    return model

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Create a folder to save files """
create_dir("files")

""" Hyperparameters """
batch_size = 32
num_epochs = 100
input_shape = (560, 1)
num_classes = 6
latent_dim = 128
lr = 1e-4

model_path = "files/model_classifier.h5"
csv_path = "files/log_classifier.csv"

""" Model & Training """
classifier = build_classifier(autoencoder, num_classes=num_classes)
classifier.summary()
adam = tf.keras.optimizers.Adam(lr)
classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]
histo = classifier.fit(train_X, train_y,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_X, test_y),
    callbacks = callbacks
    )

def plot_learningCurve(history, epochs):
  #accuracy
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()
#validaion loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()

plot_learningCurve(histo, 17)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Model: Weight file """
model_path = "files/model_classifier.h5"
model = load_model(model_path)

""" Evalution """
y_pred = np.argmax(model.predict(test_X), axis=-1)
y_true = np.argmax(test_y, axis=1)

acc = accuracy_score(y_true, y_pred, normalize=True)
print(f"Accuracy: {acc}")

mat = confusion_matrix(y_true, y_pred)
print(mat)

from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_true,y_pred))

!pip install mlxtend

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score

mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf_mat = mat,hide_spines= False, cmap = 'Greens', hide_ticks= False, figsize=(10, 10) ,show_normed=True)

from sklearn.metrics import f1_score

f1score = f1_score(y_true, y_pred, average= 'weighted')
print(f"Accuracy: {f1score}")

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer("LATENT").output)
activations = activation_model(train_X[10])

 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

display_activation(activations, 8, 8, 1)

