from src.data_loader import DataGenerator2D
from src.unet_model import UNet
from src.utils import *
import numpy as np
from src.config import *

def main():
    
    data = DataGenerator2D(base_path_color=BASE_PATH_COLOR, 
                        base_path_bw=BASE_PATH_BW, 
                        img_size=SIZE,shuffle=True)
    
    test_gray_image,test_color_image, train_gray_image, train_color_image = data.getitem(train_files=NUM_TEST_FILES)
    
    print('Train gray images shape:',train_gray_image.shape)
    print('Train color images shape:',train_color_image.shape)
    print('Test gray images shape:',test_gray_image.shape)
    print('Test color images shape:',test_color_image.shape)

    model = UNet()
    history,model = model.train(train_gray_image, train_color_image)
    model.evaluate(test_gray_image,test_color_image)
    
    plotModelHistory(history) 
    
    for i in range(50,58):
        predicted = np.clip(model.predict(test_gray_image[i].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
        plot_images(test_color_image[i],test_gray_image[i],predicted)

if __name__ == "__main__":
    main()
