from data_loader import StanfordLoader, Hmdb51Loader
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    print('Start loading data')

    data_params = {
        'image_size': (224, 224),
        'batch_size': 32,
        'print_info': True,
        'test_size': 0.1,
        'val_size': 0.1
    }
    SL = StanfordLoader(data_params)
    HL = Hmdb51Loader(data_params)

    S_train_gen, S_val_gen, S_test_gen = SL.get_image_generators()
    H_train_im_gen, H_val_im_gen, H_test_im_gen = HL.get_image_generators()
    H_train_flow_gen, H_val_flow_gen, H_test_flow_gen = HL.get_optical_flow_generators()
    print("Loading data finished")

    cv.imshow("stanford train image", np.float32(S_train_gen.next()[0][0]))
    cv.imshow("HMDB train image", np.float32(H_train_im_gen.__next__()[0][0]))
    flowstack = np.float32(H_train_flow_gen.__next__()[0][0])
    #SOME WAY TO VISUALIZE OPTICAL FLOW
    cv.waitKey(0)
    