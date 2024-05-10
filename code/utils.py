import os, sys
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from glob import glob
import random
from sklearn.mixture import GaussianMixture
from joblib import load

import tensorflow as tf
import keras.backend as K

from tensorflow.keras.layers import *     # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import * # type: ignore

def write_dict_to_csv(file, dict):
    with open(file, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])

def export_predicted_validation_images(X_in, model, storepath, epoch = "", X_in_paths=[], cmap=None, Interpolation=True):
    Y = []
    for x_idx in range(len(X_in)):
        Y.append(np.concatenate(model.predict([X_in[x_idx:x_idx+1][..., :1], X_in[x_idx:x_idx+1][..., 1:]], verbose=0), axis=-1))
    validationPath = os.path.join(storepath, "epoch_" + epoch if epoch != "" else "final_results")

    if not os.path.exists(validationPath):
        os.makedirs(validationPath) 

    for image, path in zip(Y, X_in_paths):
        
        if Interpolation:
            image = np.concatenate([image[..., 0:1] * 255, probabilities_2_images(cmap,  image[..., 1:])[..., 1:]], axis=-1)
        else:
            image = np.concatenate([image[..., 0:1] * 255, classes_2_images(cmap,  np.argmax(image[...,1:], -1))[..., 1:]], axis=-1)
        image = (image + 0.5).astype(np.uint8)[0]

        # Convert to BGR
        image = cv.cvtColor(image, cv.COLOR_Lab2BGR)

        path = path[0]
        # Get the base name (file name)
        file_name = os.path.basename(path)

        # Get the full path of the directory containing the file
        directory_path = os.path.dirname(path)

        # Get the name of the upper directory
        upper_directory_name = os.path.basename(directory_path)

        train_test =  os.path.basename(os.path.dirname(directory_path))

        if not os.path.exists(os.path.join(validationPath, train_test,  upper_directory_name)):
            os.makedirs(os.path.join(validationPath, train_test, upper_directory_name)) 
        
        cv.imwrite(os.path.join(validationPath, train_test, upper_directory_name, file_name[:-6] + ("Interpolate_" if Interpolation  else "") + "Predict.bmp"), image)

def initialize_dataset(datapath):
    InputTrainDataFiles = []
    TargetTrainDataFiles = []
    InputTestDataFiles = []
    TargetTestDataFiles = [] 
    InputValidationDataFiles = []
    TargetValidationDataFiles = [] 
    

    for directory in  [os.path.join(datapath, "Train", dirs) for dirs in os.listdir(os.path.join(datapath, "Train")) if os.path.isdir(os.path.join(datapath, "Train", dirs))]:
            TargetTrainDataFiles.append(glob(os.path.join(directory, "*REF.*"))[0])
            InputTrainDataFiles.append([glob(os.path.join(directory, "*IR.*"))[0],
                                        glob(os.path.join(directory, "*II.*"))[0],
                                        glob(os.path.join(directory, "*Vis.*"))[0]]
                                    )
    for directory in  [os.path.join(datapath, "Validation", dirs) for dirs in os.listdir(os.path.join(datapath, "Validation")) if os.path.isdir(os.path.join(datapath, "Validation", dirs))]:
            TargetValidationDataFiles.append(glob(os.path.join(directory, "*REF.*"))[0])
            InputValidationDataFiles.append([glob(os.path.join(directory, "*IR.*"))[0],
                                        glob(os.path.join(directory, "*II.*"))[0],
                                        glob(os.path.join(directory, "*Vis.*"))[0]]
                                    )
    for directory in  [os.path.join(datapath, "Test", dirs) for dirs in os.listdir(os.path.join(datapath, "Test")) if os.path.isdir(os.path.join(datapath, "Test", dirs))]:
        TargetTestDataFiles.append(glob(os.path.join(directory, "*Vis.*"))[0])
        InputTestDataFiles.append([glob(os.path.join(directory, "*IR.*"))[0],
                                    glob(os.path.join(directory, "*II.*"))[0],
                                    glob(os.path.join(directory, "*Vis.*"))[0]]
                                )      

    train_zip = list(zip(InputTrainDataFiles, TargetTrainDataFiles))
    random.shuffle(train_zip)
    InputTrainDataFiles, TargetTrainDataFiles = [list(a_list) for a_list in zip(*train_zip)]


    test_zip = list(zip(InputTestDataFiles, TargetTestDataFiles))
    random.shuffle(test_zip)
    InputTestDataFiles, TargetTestDataFiles = [list(a_list) for a_list in zip(*test_zip)]

    dataset = {"InputTrainDataFiles": InputTrainDataFiles, "TargetTrainDataFiles": TargetTrainDataFiles,
               "InputValidationDataFiles": InputValidationDataFiles, "TargetValidationDataFiles": TargetValidationDataFiles,
                "InputTestDataFiles": InputTestDataFiles, "TargetTestDataFiles": TargetTestDataFiles}    

    return dataset

def prepare_dataset(X, Y, imWidth, imHeight, scaleFactor, gmm):
    
    X_train_batch = np.array([[cv.resize(cv.imread(image_path,  cv.IMREAD_GRAYSCALE), (int(imWidth * scaleFactor), int(imHeight * scaleFactor)), interpolation = cv.INTER_LINEAR) for image_path in image_paths] for image_paths in X]).transpose(0, 2, 3, 1)

    Y_train_batch = [cv.resize(cv.imread(image_path), (int(imWidth * scaleFactor), int(imHeight * scaleFactor)), interpolation = cv.INTER_LINEAR)  for image_path in Y]

    Y_train_batch = np.array([cv.cvtColor(img, cv.COLOR_BGR2Lab) for img in Y_train_batch])  

    Y_train_batch = np.concatenate([Y_train_batch[..., 0:1] / 255, image_2_probabilities(gmm, Y_train_batch)], axis=-1)

    return X_train_batch.astype(np.float32), Y_train_batch.astype(np.float32)

def initialize_get_batch():
    
    rotation = tf.keras.layers.RandomRotation(factor=1/24) # 2*pi*x=pi/12 => x = 1/24
    flip = tf.keras.layers.RandomFlip(mode="horizontal")
    crop = tf.keras.layers.RandomCrop(256, 320)
    resize = tf.keras.layers.Resizing(256, 320)

    def get_batch(X, Y):
     
        concanated = tf.concat([X, Y], axis=-1)
        concanated = rotation(concanated)
        concanated = flip(concanated)
        do_crop = tf.random.uniform([], 0, 1) > 0.25 # 256 * 320 / (640 * 480) => 0.266 => 0.25
        if do_crop:
            concanated = crop(concanated)
        else:
            concanated = resize(concanated)
        
        return  concanated[...,:3], concanated[...,3:]
    return get_batch

def calculate_ssim_psnr(model, X, Y, cmap):
    prediction = []
    for idx in range(X.shape[0]):
        prediction.append(np.concatenate(model.predict([X[idx:idx+1, ...,  :1], X[idx:idx+1, ..., 1:3]], verbose=0), axis=-1))
    Y = np.array(Y)
    prediction = np.array(prediction).squeeze()

    prediction = np.concatenate([prediction[..., 0:1] * 255, probabilities_2_images(cmap, prediction[...,1:])[..., 1:]], axis=-1)
    
    prediction = (prediction + 0.5).astype(np.uint8)
    
    prediction = np.array([cv.cvtColor(prediction[idx,...,:], cv.COLOR_Lab2BGR) for idx in range(prediction.shape[0])])
    
    return tf.image.ssim(prediction, Y,255).numpy(), np.array([cv.PSNR(im1, im2) for im1, im2 in zip(prediction, Y)])

def store_results(folder_name, loss_and_acc, params, models, total_execution_time, execution_times_per_epoch, intermediate_results=False):
    
    #Generate folder name from parameters
    # dtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = folder_name + os.sep if folder_name[-1] != os.sep else folder_name
    for param in params.keys():
        folder_name = folder_name + param + "_" + str(params[param]) + "_"
    
    if not os.path.exists(os.path.dirname(folder_name)):
        os.makedirs(os.path.dirname(folder_name))

    store_test_dir = folder_name
    
    #Store model
    for key in models:
        model_json = models[key].to_json() 
        with open(store_test_dir+key+".yaml", "w") as json_file:
            json_file.write(model_json)
        models[key].save_weights(store_test_dir+key+'.h5')

    if not intermediate_results:

        if not os.path.exists(store_test_dir):
            os.makedirs(store_test_dir)
        
        #Store losses and accuracies in file and plot them 
        write_dict_to_csv(store_test_dir + "losses_and_acc.csv", loss_and_acc)
        
        for key in loss_and_acc:
            for metric in loss_and_acc[key]:
                if isinstance(loss_and_acc[key][metric], float) or loss_and_acc[key][metric].size > 0:
                    plt.clf()
                    plt.plot(loss_and_acc[key][metric])
                    plt.title(key + "_" + metric)
                    plt.ylabel(metric)
                    plt.xlabel("epoch")
                    plt.savefig(store_test_dir + key + "_" + metric + "_plot.png")

                    # with open(store_test_dir + key + "_" + metric + ".txt", 'w') as file:
                    #     file.write(f"{loss_and_acc[key][metric]}\n")

        #Store parameters in file
        write_dict_to_csv(store_test_dir + "parameters.csv", params)

        #Write model summary to file
        for key in models:
            orig_stdout = sys.stdout
            f = open(store_test_dir + key + "_summary.txt", "w")
            sys.stdout = f
            print(models[key].summary())
            sys.stdout = orig_stdout
            f.close()
        
        #Write execution times
        f = open(store_test_dir + "execution_times.txt", "w")
        f.write("Total execution time: " + total_execution_time + "\n\n")
        f.write("Execution time per epoch\n")
        f.write("------------------------------------------\n")
        for idx, time in enumerate(execution_times_per_epoch): f.write(str(idx) + ": " + time + "\n")
        f.close()
    
    return store_test_dir

def update_dictionary(orig, new, dictInDict = False):
    for key in new.keys():
        if key in orig.keys():
            if dictInDict:
                for key2 in new[key].keys():
                    if key2 in orig[key].keys():
                        orig[key][key2] = np.append(orig[key][key2], new[key][key2])
                    else:
                        orig[key][key2] = new[key][key2]
            else:
                orig[key] = np.append(orig[key], new[key])
        else:
            orig[key] = new[key]
    return orig

def generate_test_data(file_list, scaleFactor, imWidth, imHeight):
    
    images = []
    
    for files in file_list:
        # Read image using OpenCV
        image = [cv.imread(file, cv.IMREAD_COLOR) for file in files] # OpenCV loads images as BGR
        
        # Resize image
        image = [cv.resize(img, (int(imWidth * scaleFactor), int(imHeight * scaleFactor))) for img in image]

        image = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in image]

        images.append(np.array(image))
    
    images = np.array(images, dtype='float32').transpose(0, 2, 3, 1)
    
    return images

def cluster(ImagePaths, number_of_classes, gmm_path=None):

    if gmm_path is None:
        rgb = cv.resize(cv.imread(ImagePaths[0]), (640, 512))
        for path in ImagePaths[1:]:
            rgb = np.concatenate((rgb,  cv.resize(cv.imread(path), (640, 512))), axis=1)

        ab = cv.cvtColor(rgb, cv.COLOR_BGR2Lab)
        # Reshape the image to be a list of RGB values
        flat_image = ab.reshape(-1, 3)

    if gmm_path is None:
        print("----Clustering Starts----")
        model = GaussianMixture(n_components=number_of_classes).fit(flat_image)
        print("----Clustering Finished----")
    else:
        model = load(gmm_path)
    cmap = model.means_
    return model, (cmap + 0.5).astype(np.uint8)

def image_2_classes(kmeans, img):
    return kmeans.predict(img.reshape(-1, 3)).reshape(img.shape[:-1] + (1, ))

def image_2_probabilities(gmm, img):
    return gmm.predict_proba(img.reshape(-1, 3)).reshape(img.shape[:-1] + (gmm.n_components, ))

def classes_2_images(cmap, img):
    return cmap[img]

def probabilities_2_images(cmap, img):
    return (np.dot(img, cmap) + 0.5).astype(np.uint8)

def output_color_palette(cmap, path):
    def bgr_to_hsv(bgr_color):
        """
        Convert a BGR color to HSV using OpenCV
        """
        bgr_color = np.uint8([[bgr_color]])
        hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)
        return hsv_color[0][0]

    # Convert the BGR colors to HSV
    hsv_colors = [bgr_to_hsv(color) for color in cmap]

    # Sort the colors by their Hue value
    hsv_colors_sorted = sorted(hsv_colors, key=lambda x: x[0])

    # Create a new image
    height, width = len(hsv_colors_sorted) * 20, len(hsv_colors_sorted) * 20
    sorted_image = np.ones((height, width, 3), np.uint8) * 250

    # Calculate the width of each color segment
    segment_width = width // len(hsv_colors_sorted)

    # Paint each segment with the sorted colors
    for i, hsv_color in enumerate(hsv_colors_sorted):
        start_x = i * segment_width
        end_x = start_x + segment_width
        # Convert HSV back to BGR for display
        bgr_color = cv.cvtColor(np.uint8([[hsv_color]]), cv.COLOR_HSV2BGR)[0][0]
        sorted_image[:, start_x:end_x] = bgr_color

    # Display the sorted image
    cv.imwrite(os.path.join(path, "color_palette.bmp"), sorted_image)

# Loss functions ---------------------------------------------
def extract_image_patches(X, ksizes, strides, rates, padding='valid', data_format='channels_first'):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    strides : 2-d tuple with the strides size
    padding : 'same' or 'valid'
    data_format : 'channels_last' or 'channels_first'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    https://github.com/farizrahman4u/keras-contrib/blob/master/keras_contrib/backend/theano_backend.py
    '''
    return tf.image.extract_patches(X, ksizes, strides, rates, padding.upper())

class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=[4,16], max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
            https://github.com/farizrahman4u/keras-contrib/blob/master/keras_contrib/losses/dssim.py
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size[0], self.kernel_size[1], 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = extract_image_patches(y_pred, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        #bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        #patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        #patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)

class SSIM():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=[4,16], max_value=1.0):
        self.__name__ = 'SSIM'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size[0], self.kernel_size[1], 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = extract_image_patches(y_pred, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean(ssim)

class L():
    def __init__(self):
        self.__dssim = DSSIMObjective()

    def __call__(self, y_true, y_pred):
        return [0.85 * self.__dssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1)) + 0.15 *  tf.keras.metrics.mean_absolute_error(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])] 

# Architectures ---------------------------------------------
def C_block(H, nch, bn, downsample, first=False):
    if first:
        H = BatchNormalization(axis = -1)(H)
    else:
        H = LeakyReLU(0.2)(H)
    if downsample:
        H = Conv2D(nch, (3,3), strides = (2,2), padding = 'same', kernel_initializer='glorot_normal')(H)
    else:
        H = Conv2D(nch, (3,3), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = -1)(H)

def CD_block(H, nch, bn, upsample):
    H = LeakyReLU(0.0)(H)
    if upsample:
        H = UpSampling2D(size = (2,2), interpolation = "bilinear")(H)
    H = Conv2D(nch, (3,3), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = -1)(H)
    H = Dropout(0.5)(H)
    return H    

def CD_skip_block_dual(H, Hskip1, Hskip2, nch, bn, upsample):
    H = LeakyReLU(0.0)(H)
    if upsample:
        H = UpSampling2D(size = (2,2), interpolation = "bilinear")(H)
    H = Conv2D(nch, (3,3), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = -1)(H)
    H = Dropout(0.5)(H)
    H = Concatenate(axis = -1)([H, Hskip1, Hskip2])    
    return H

def create_dual_encoder_unet(input_shape, number_of_classes):
    nch_max=256

    nch = [int(nch_max/8), int(nch_max/4), int(nch_max/2), int(nch_max/2)]
    depth = len(nch)

    ir_input  = Input(shape=input_shape + [1], name='ir_input')
    H_ir = ir_input

    encoder_layers_ir  = []

    for d in range(depth):
        if d == 0:
            H_ir = C_block(H_ir, nch[d], False, True, True)
            encoder_layers_ir .append(H_ir)
        else:
            if d < 2:
                H_ir = C_block(H_ir, nch[d], True, True)
            else:
                H_ir = C_block(H_ir, nch[d], True, False)
            encoder_layers_ir.append(H_ir)


    ii_vis_input  = Input(shape=input_shape + [2], name='ii_vis_input')
    H_ii_vis  = ii_vis_input

    encoder_layers_ii_vis = []

    for d in range(depth):
        if d == 0:
            H_ii_vis = C_block(H_ii_vis, nch[d], False, True, True)
            encoder_layers_ii_vis.append(H_ii_vis)
        else:
            if d < 2:
                H_ii_vis = C_block(H_ii_vis, nch[d], True, True)
            else:
                H_ii_vis = C_block(H_ii_vis, nch[d], True, False)
            encoder_layers_ii_vis.append(H_ii_vis)

    # Concatenate the outputs of the two encoders
    H_dec = Concatenate(axis=-1)([encoder_layers_ir[3], encoder_layers_ii_vis[3]])

    #Decoder
    for d in range(depth):
        if d < depth-1:
            if depth-d-1 < 2:
                H_dec = CD_skip_block_dual(H_dec, encoder_layers_ir[depth-d-2], encoder_layers_ii_vis[depth-d-2], nch[depth-d-1], True, True)
            else:
                H_dec = CD_skip_block_dual(H_dec, encoder_layers_ir[depth-d-2], encoder_layers_ii_vis[depth-d-2], nch[depth-d-1], True, False)
        else:
            H_dec = CD_block(H_dec, nch[depth-d-1], True, True)

    H_dec = LeakyReLU(0.0)(H_dec)

    # Final output layer
    output_L = Conv2D(1, (3, 3), activation="sigmoid", padding='same', kernel_initializer='glorot_normal', name='out_l')(H_dec)
    output_ab = Conv2D(number_of_classes, (3, 3), activation="softmax", padding='same', kernel_initializer='glorot_normal', name='out_ab')(H_dec)

    # Create Model
    model = Model(inputs=[ir_input, ii_vis_input], outputs=[output_L, output_ab])
    return model
