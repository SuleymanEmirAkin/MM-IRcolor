import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import timedelta
import numpy as np
from keras.activations import *
from keras.optimizers import *
from tqdm import tqdm
from collections import OrderedDict
from utils import *
import tensorflow as tf
from tensorflow.keras.utils import plot_model # type: ignore
import joblib

# Output Path
output_folder = "Output" 
folder_name = "YOUR_FOLDER_NAME" 

# Dataset Path
datapath = "YOUR_FOLDER_NAME"

complete_output_path = os.path.join(output_folder , folder_name)
os.makedirs(complete_output_path, exist_ok=True)

#Parameters
typeOfTest = ""

# Training Procedure
epochs = 300
batch_size = 8
batches_per_epoch = 100

# Classification
number_of_classes = 80
gmm_path = None  # Instead of finding classes again and again you can reuse your old GMM
                 # If you do not have any please write None

# Image Size
imWidth = 640
imHeight = 512
scaleFactor = 1  #Image scale factor

# Optimizer
lr = 1e-3
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
opt = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

# Build model ---------------------------------------------------------------
model = create_dual_encoder_unet([None, None], number_of_classes)

model.compile(loss={"out_l": L() , "out_ab": "categorical_crossentropy"}, optimizer=opt, metrics=["accuracy"])
model.summary()
plot_model(model, to_file=os.path.join(complete_output_path, 'our_model.png'), show_shapes=True, show_layer_names=True)

# Initialize dataset ---------------------------------------------------------------
dataset = initialize_dataset(datapath)
predict_file_list = dataset['InputTrainDataFiles'] + dataset['InputValidationDataFiles'] 

gmm, cmap = cluster(dataset['TargetTrainDataFiles'], number_of_classes, gmm_path)
joblib.dump(gmm, os.path.join(complete_output_path, 'gmm_model.pkl'))
output_color_palette(cv.cvtColor(cmap[np.newaxis, :], cv.COLOR_Lab2BGR)[0], complete_output_path)

X_input = generate_test_data(predict_file_list, scaleFactor, imWidth, imHeight)

X_val = generate_test_data(dataset['InputValidationDataFiles'] , scaleFactor, imWidth, imHeight)
Y_val = [cv.resize(cv.imread(image_path), (int(imWidth * scaleFactor), int(imHeight * scaleFactor)), interpolation = cv.INTER_LINEAR)  for image_path in dataset['TargetValidationDataFiles']]

X_train = generate_test_data(dataset['InputTrainDataFiles'] , scaleFactor, imWidth, imHeight)
Y_train = [cv.resize(cv.imread(image_path), (int(imWidth * scaleFactor), int(imHeight * scaleFactor)), interpolation = cv.INTER_LINEAR)  for image_path in dataset['TargetTrainDataFiles']]

loss_and_accuracy = {}
params = {}

# Set params for result storage ----------------------------------------------------
params = OrderedDict([(typeOfTest, ''),('ep', epochs),('bs', batch_size),('#samp', int(len(dataset['InputTrainDataFiles']))),('lr',lr)])

# Define training procedure --------------------------------------------------------
def train_for_n(nb_epoch, time_per_epoch):
    print("----Training Starts----\n")
    preprocessed_images, preprocessed_targets = prepare_dataset(X=dataset['InputTrainDataFiles'], Y=dataset['TargetTrainDataFiles'], imWidth=imWidth, imHeight=imHeight, scaleFactor=scaleFactor, gmm=gmm)

    dataset_preprocessed  = tf.data.Dataset.from_tensor_slices((preprocessed_images, preprocessed_targets))

    dataset_processed = dataset_preprocessed.map(initialize_get_batch(), num_parallel_calls=tf.data.AUTOTUNE).repeat(int(batches_per_epoch * batch_size / len(dataset_preprocessed) + 0.5))

    dataset_processed = dataset_processed.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset_processed  = dataset_processed.batch(batch_size)

    loss_and_accuracy_e = {"model": {}} 
    epoch_losses = {}

    for epoch in tqdm(range(nb_epoch)):
        epoch_time_start = time.perf_counter()
        for inputs, targets in dataset_processed.take(batches_per_epoch):
            update_dictionary(epoch_losses, dict(zip(model.metrics_names, model.train_on_batch([inputs[...,:1], inputs[...,1:3]], [targets[...,0:1], targets[...,1:]]))))

        epoch_losses["val_ssim"], epoch_losses["val_psnr"] = calculate_ssim_psnr(model, X_val, Y_val, cmap)
        epoch_losses["val_ssim_max "] = np.array(max(epoch_losses["val_ssim"]))
        epoch_losses["val_psnr_max"] = np.array(max(epoch_losses["val_psnr"]))

        epoch_losses["train_ssim"] , epoch_losses["train_psnr"] = calculate_ssim_psnr(model, X_train, Y_train, cmap)
        epoch_losses["train_ssim_max"] = np.array(max(epoch_losses["train_ssim"]))
        epoch_losses["train_psnr_max"] = np.array(max(epoch_losses["train_psnr"]))

        for key in epoch_losses.keys():
            epoch_losses[key] = epoch_losses[key].mean()

        update_dictionary(loss_and_accuracy_e["model"], epoch_losses)
        epoch_losses = {}

        total_epoch_time_elapsed = str(timedelta(seconds=(time.perf_counter() - epoch_time_start)))
        time_per_epoch.append(total_epoch_time_elapsed)

        #For each epoch, store intermediate results
        store_path = store_results(complete_output_path, loss_and_accuracy_e, params, {"model": model}, str(timedelta(seconds=(time.perf_counter() - time_start))), time_per_epoch)
        if epoch % 5 == 4:
            store_results(os.path.join(complete_output_path, "checkpoint", str(epoch)), loss_and_accuracy, params, {"model": model}, -1, -1, intermediate_results=True)
            export_predicted_validation_images(X_input, model, store_path, epoch = str(epoch), X_in_paths = predict_file_list, cmap=cmap, Interpolation=True)
            export_predicted_validation_images(X_input, model, store_path, epoch = str(epoch), X_in_paths = predict_file_list, cmap=cmap, Interpolation=False)

    return loss_and_accuracy_e
# End train_for_n --------------------------------------------------------

# Train network!  --------------------------------------------------------
time_start = time.perf_counter()
time_per_epoch = []

loss_and_accuracy_lr = train_for_n(epochs, time_per_epoch)   
update_dictionary(loss_and_accuracy, loss_and_accuracy_lr, dictInDict = True)

total_time_elapsed = str(timedelta(seconds=(time.perf_counter() - time_start)))
store_path = store_results(complete_output_path, loss_and_accuracy, params, {"model": model}, total_time_elapsed, time_per_epoch)

# Validate on some example images --------------------------------------------------------
X_validation_in = generate_test_data(dataset['InputTestDataFiles'] + predict_file_list, scaleFactor, imWidth, imHeight)
export_predicted_validation_images(X_validation_in, model, store_path, X_in_paths=predict_file_list, cmap=cmap, Interpolation=True)
export_predicted_validation_images(X_validation_in, model, store_path, X_in_paths=predict_file_list, cmap=cmap, Interpolation=False)