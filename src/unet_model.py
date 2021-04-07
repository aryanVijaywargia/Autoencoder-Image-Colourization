import tensorflow as tf
from tensorflow.keras import layers
from utils import cp_callback
from config import SIZE, EPOCHS, BATCH_SIZE

class UNet():
    def __init__(self, input_shape=SIZE):
        self.input_shape = input_shape
        
    def __downsample_block(self, filters , kernel_size, apply_batch_normalization = True):
        downsample = tf.keras.models.Sequential()
        downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
        if apply_batch_normalization:
            downsample.add(layers.BatchNormalization())
        downsample.add(layers.LeakyReLU())
        return downsample
    
    def __upsample_block(self, filters, kernel_size, dropout = False):
        upsample = tf.keras.models.Sequential()
        upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
        if dropout:
            upsample.dropout(0.2)
        upsample.add(layers.LeakyReLU())
        return upsample

    def __model(self):
        inputs = layers.Input(shape= [self.input_shape,self.input_shape,3])
        d1 = self.__downsample_block(128,(3,3),False)(inputs)
        d2 = self.__downsample_block(128,(3,3),False)(d1)
        d3 = self.__downsample_block(256,(3,3),True)(d2)
        d4 = self.__downsample_block(512,(3,3),True)(d3)
        d5 = self.__downsample_block(512,(3,3),True)(d4)
        #upsampling
        u1 = self.__upsample_block(512,(3,3),False)(d5)
        u1 = layers.concatenate([u1,d4])
        u2 = self.__upsample_block(256,(3,3),False)(u1)
        u2 = layers.concatenate([u2,d3])
        u3 = self.__upsample_block(128,(3,3),False)(u2)
        u3 = layers.concatenate([u3,d2])
        u4 = self.__upsample_block(128,(3,3),False)(u3)
        u4 = layers.concatenate([u4,d1])
        u5 = self.__upsample_block(3,(3,3),False)(u4)
        u5 = layers.concatenate([u5,inputs])
        output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
        return tf.keras.Model(inputs=inputs, outputs=output)
    
    def __compile(self):
        model = self.__model()
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), 
                    loss = 'mean_absolute_error', 
                    metrics = ['acc'])
        return model
    
    def train(self, train_gray_image, train_color_image):
        model = self.__compile()
        model.summary()
        history = model.fit(train_gray_image, train_color_image, epochs = EPOCHS,batch_size = BATCH_SIZE,verbose = 1, callbacks = [cp_callback])
        return model, history
    