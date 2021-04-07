import re
import tensorflow as tf
import config
import matplotlib.pyplot as plt

def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= config.BEST_MODEL_PATH,
                                                save_weights_only=True, 
                                                monitor='accuracy',
                                                verbose=1)

def plotModelHistory(h):
    fig, ax = plt.subplots(1, 1, figsize=(15,4))
    ax.plot(h.history['loss'])   
    ax.plot(h.history['acc'])
    ax.legend(['loss','acc'])
    ax.title.set_text("Train loss vs Train accuracy")
    print("Max. Training Accuracy", max(h.history['acc']))
    
# defining function to plot images pair
def plot_images(color,grayscale,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('Color Image', color = 'green', fontsize = 20)
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.title('Grayscale Image ', color = 'black', fontsize = 20)
    plt.imshow(grayscale)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)
    plt.show()

