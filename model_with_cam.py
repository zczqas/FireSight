#%% md
# <h1>Table of Contents</h1>
# 
# <div>
#   <ul>
#       <li><a href="https://#unzip_data"> Unzip data</a></li>
#       <li><a href="https://#auxiliary"> Imports and Auxiliary Function </a></li>
#       <li><a href="https://#examine_files">Examine Files</a></li>
#       <li><a href="https://#Display">Display and Analyze Image With No Trees</a></li>
#   </ul>
# </div>
#%% md
# <h2 align=center id="unzip_data">Upload Data</h2>
#%%
# !tar -xf archive.zip
!unzip archive.zip
#%% md
# <h2 align=center id="auxiliary">Imports and Auxiliary Function</h2>
# 
# <p>The following are the libraries we are going to use for this lab:</p>
#%%
import os
import tensorflow as tf
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
#%%
for device in tf.config.list_physical_devices():
    print(f"{device.name}")
#%%
try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization,  GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard

import warnings
warnings.filterwarnings("ignore")
#%%
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#%%
def plot_loss(history):
  plt.figure(figsize=(20, 10))
  sns.set_style('whitegrid')
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Wildfire Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['loss', 'val_loss'], loc='upper left')
  plt.show()
#%%
def plot_acc(history):
  plt.figure(figsize=(20, 10))
  sns.set_style('whitegrid')
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Wildfire Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
  plt.show()
#%% md
# <h2 align=center id="examine_files">Dataset Preparation</h2>
#%% md
# <p><i><b>Please run following block if you are running for the first time.</b></i></p>
#%%
!mkdir -p saved_model
!mkdir -p predictions
!mkdir -p model_plots
#%% md
# <p><i><b>Make sure three folders are created<b></i></p>
#%%
%ls
#%%
train_path = "train"
valid_path = "valid"
test_path = "test"
#%%
im_size = 224 #@param {type:"slider", min:64, max:350, step:1}
image_resize = (im_size, im_size, 3) 
batch_size_training = 100 #@param {type:"number"}
batch_size_validation = 100 #@param {type:"number"}
batch_size_test = 100 #@param {type:"number"}
num_classes = 2 #@param {type:"number"}
#%%
data_generator = ImageDataGenerator(dtype='float32', rescale= 1./255.)
#%%
train_generator = data_generator.flow_from_directory(train_path,
                                                   batch_size = batch_size_training,
                                                   target_size = (im_size, im_size),
                                                   class_mode = 'categorical')

valid_generator = data_generator.flow_from_directory(valid_path,
                                                   batch_size = batch_size_validation,
                                                   target_size = (im_size, im_size),
                                                   class_mode = 'categorical')
#%%
class_mapping = train_generator.class_indices
class_mapping
#%%
first_batch_train = next(train_generator)
first_batch_train
#%%
first_batch_valid = next(valid_generator)
first_batch_valid
#%%
# labels = np.array(['nowildfire', 'wildfire'])
class_names = list(train_generator.class_indices.keys())
print("Class names :", class_names)
#%%
custom_palette = {'nowildfire': 'skyblue', 'wildfire': 'orange'}

data = pd.DataFrame({'Class': class_names, 'Count': [sum(train_generator.labels == c) for c in range(num_classes)]})

plt.figure(figsize=(20, 7))
sns.barplot(x='Class', y='Count', data=data, palette=custom_palette)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
#%%
labels_train = train_generator.classes
unique_labels_train, label_counts_train = np.unique(labels_train, return_counts=True)

print("Number of unique labels in train data:", len(unique_labels_train))
for label, count in zip(unique_labels_train, label_counts_train):
    print("Label:", class_names[label], "- Count:", count)
#%%
custom_palette = {'nowildfire': 'skyblue', 'wildfire': 'orange'}

data = pd.DataFrame({'Class': class_names, 'Count': [sum(valid_generator.labels == c) for c in range(num_classes)]})

plt.figure(figsize=(20, 7))
sns.barplot(x='Class', y='Count', data=data, palette=custom_palette)
plt.title('Class Distribution in Validation Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
#%%
labels_valid = valid_generator.classes
unique_labels_valid, label_counts_valid = np.unique(labels_valid, return_counts=True)

print("Number of unique labels in valid data:", len(unique_labels_valid))
for label, count in zip(unique_labels_valid, label_counts_valid):
    print("Label:", class_names[label], "- Count:", count)
#%% md
# <br>
# 
# <h2 align=center id="fit_custom_model">Compile and Fit Custom Model</h2>
#%% md
# <p>If you can use my model, please uncomment the following code block and go to <code>model.summary()</code> block.</p>
#%%
model = load_model('saved_model/custom_best_model.h5')
#%%
def base_model(input_shape, repetitions): 
  
  input_ = tf.keras.layers.Input(shape=input_shape, name='input')
  x = input_
  
  for i in range(repetitions):
    n_filters = 2**(4 + i)
    x = Conv2D(n_filters, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)

  return x, input_
#%%
def final_model(input_shape, repetitions):
    
    x, input_ = base_model(input_shape, repetitions)

    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    class_out = Dense(num_classes, activation='softmax', name='class_out')(x)

    model = Model(inputs=input_, outputs=class_out)

    print(model.summary())
    return model
#%%
model = final_model(image_resize, 4)
#%%
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_plots/custom_model.png')
#%%
get_ipython().system('rm -rf logs')
#%%
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
checkpoint = tf.keras.callbacks.ModelCheckpoint('saved_model/custom_best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#%%
callbacks_list = [checkpoint, tensorboard_callback]
#%%
num_epochs = 2 #@param {type:"number"}
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(valid_generator)
#%%
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[callbacks_list],
)
#%% md
# <br>
# 
# <h2 align=center id="analyze_model">Analyze the Model</h2>
#%%
# %load_ext tensorboard
%reload_ext tensorboard
#%% md
# <p>Please modify the path of <code>logs</code>.</p>
#%%
%tensorboard --logdir logs
#%%
plot_acc(history)
#%%
plot_loss(history)
#%%
model.save('saved_model/custom_model.keras')
print("Model saved!")
#%%
# model.save('saved_model/custom_model.h5')
# print("Model saved!")
#%% md
# <h2 align=center id="make_dataframe">Make DataFrame for the Predictions</h2>
#%%
test_generator = data_generator.flow_from_directory(test_path,
                                                   batch_size = batch_size_test,
                                                   target_size = (im_size, im_size),
                                                   class_mode = 'categorical')
#%%
filenames = test_generator.filenames
#%%
pred = model.predict(test_generator, steps=len(test_generator), verbose=1).round(3)
#%%
filenames_df = pd.DataFrame(filenames, columns=['File Path'])
pred_df = pd.DataFrame(pred, columns=['No Wildfire Probability', 'Wildfire Probability'])
model_predictions = pd.concat([filenames_df, pred_df], axis=1)
model_predictions
#%%
file_name='predictions/custom_model_predictions.csv'
model_predictions.to_csv(file_name, sep=',', encoding='utf-8')
#%% md
# <br>
# <h2 align=center id="build_cam">Building Class Activation Maps</h2>
#%%
outputs = [layer.output for layer in model.layers[1:9]]
#%%
vis_model = Model(model.input, outputs)
#%%
layer_names = []
for layer in outputs:
    layer_names.append(layer.name.split("/")[0])
#%%
print("Layers that will be used for visualization: ")
print(layer_names)
#%%
gap_weights = model.layers[-1].get_weights()[0]
gap_weights.shape
#%%
cam_model  = Model(inputs=model.input, outputs=(model.layers[-3].output,model.layers[-1].output))
cam_model.summary()
#%%
plot_model(cam_model, show_shapes=True, show_layer_names=True, to_file='model_plots/cam_model.png')
#%%
cam_model.save('saved_model/cam_model.keras')
print("Model saved!")
#%%
# cam_model.save('saved_model/cam_model.h5')
# print("Model saved!")
#%%
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_cam(image_value, features, results, gui=False):
  features_for_img = features[0]
  prediction = results[0] # noqa

  class_activation_weights = gap_weights[:,0]
  class_activation_features = sp.ndimage.zoom(features_for_img, (im_size/10, im_size/10, 1), order=2)  
  cam_output  = np.dot(class_activation_features,class_activation_weights)
  
  # Visualize the results
  plt.figure(figsize=(6, 6))
  plt.imshow(cam_output, cmap='jet', alpha=0.5)
  plt.imshow(tf.squeeze(image_value), alpha=0.5)
  plt.title('Class Activation Map')
  plt.figtext(.5, .05, f"No Wildfire Probability: {results[0][0] * 100}%\nWildfire Probability: {results[0][1] * 100}%", ha="center", fontsize=12, bbox={"facecolor":"green", "alpha":0.5, "pad":3})
  plt.colorbar()
  
  fig = plt.gcf()
  plt.show()
  
  if gui is True:
    new_window = tk.Toplevel(root) # type: ignore
    new_window.title("Class Activation Map")
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#%%
def convert_and_classify(image, gui=False):

  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img = cv2.resize(img, (im_size, im_size)) / 255.0
  tensor_image = np.expand_dims(img, axis=0)
  features, results = cam_model.predict(tensor_image)
  
  # generate the CAM
  if gui is True:
    show_cam(tensor_image, features, results, gui=True)
  else:
    show_cam(tensor_image, features, results)
#%% md
# <p><i>Simple GUI windows to add image and check model</i></p>
#%%
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        convert_and_classify(file_path, gui=True)

root = tk.Tk()
root.geometry("400x200")
root.title("Image Classification with CAM")

label = tk.Label(root, text="Click the button to select an image")
label.pack(pady=20)

btn = tk.Button(root, text="Select Image", command=open_file)
btn.pack(pady=20)

root.mainloop()
#%%
convert_and_classify('test/nowildfire/-113.91777,50.901087.jpg')
#%%
convert_and_classify('test/wildfire/-59.03238,51.85132.jpg')