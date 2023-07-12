import numpy as np
import pandas as pd
import imageio
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_path = ["/home/violayu/input-image/"+"data"+i+"/"+"data"+i+"/CameraRGB/" for i in ['A', 'B', 'C', 'D', 'E']]
mask_path = ["/home/violayu/input-image/"+"data"+i+"/"+"data"+i+"/CameraSeg/" for i in ['A', 'B', 'C', 'D', 'E']]

def list_image_paths(directory_paths):
        image_paths = []
        for directory in range(len(directory_paths)):
            image_filenames = os.listdir(directory_paths[directory])
            for image_filename in image_filenames:
                image_paths.append(directory_paths[directory] + image_filename)
        return image_paths

image_paths = list_image_paths(image_path)
mask_paths = list_image_paths(mask_path)
number_of_images, number_of_masks = len(image_paths), len(mask_paths)
print(f"1. There are {number_of_images} images and {number_of_masks} masks in our dataset")
print(f"2. An example of an image path is: \n {image_paths[0]}")
print(f"3. An example of a mask path is: \n {mask_paths[0]}")


import random
number_of_samples = len(image_paths)

for i in range(3):
    N = random.randint(0, number_of_samples - 1)

    img = imageio.imread(image_paths[N])
    mask = imageio.imread(mask_paths[N])
    mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

    fig, arr = plt.subplots(1, 3, figsize=(20, 8))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[0].axis("off")
    arr[1].imshow(mask)
    arr[1].set_title('Segmentation')
    arr[1].axis("off")
    arr[2].imshow(mask, cmap='Paired')
    arr[2].set_title('Image Overlay')
    arr[2].axis("off")


# First split the image paths into training and validation sets
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, train_size=0.8, random_state=0)

# Keep part of the validation set as test set
validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = train_test_split(val_image_paths, val_mask_paths, train_size = 0.80, random_state=0)

print(f'There are {len(train_image_paths)} images in the Training Set')
print(f'There are {len(validation_image_paths)} images in the Validation Set')
print(f'There are {len(test_image_paths)} images in the Test Set')


def read_image(image_path, mask_path):

        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (256, 256), method='nearest')

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        mask = tf.image.resize(mask, (256, 256), method='nearest')

        return image, mask

def data_generator(image_paths, mask_paths, buffer_size, batch_size):

        image_list = tf.constant(image_paths)
        mask_list = tf.constant(mask_paths)
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)

        return dataset


batch_size = 8
buffer_size = 100

train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, batch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, batch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, batch_size)


def encoding_block(inputs, filters, dropout_probability=0, max_pooling=True):

        """
        Convolutional encoding/downsampling block

        Arguments:
            inputs -- Input tensor
            filters -- Number of filters for the convolutional layers
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns:
            next_layer, skip_connection --  Next layer and skip connection inputs
        """

        C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
        C = BatchNormalization()(C)
        C = Activation("relu")(C)

        C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
        C = BatchNormalization()(C)
        C = Activation("relu")(C)

        skip_connection = C  # Set aside residual

        # if max_pooling is True, add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2, 2))(C)
        else:
            next_layer = C

        return next_layer, skip_connection


def decoding_block(inputs, skip_connection_input, filters):
    """
    Convolutional decoding/upsampling block

    Arguments:
        inputs -- Input tensor from previous layer
        skip_connection_input -- Input tensor from previous skip layer
        filters -- Number of filters for the convolutional layers
    Returns:
        C -- Tensor output
    """

    CT = Conv2DTranspose(filters, 3, strides=(2,2), padding="same", kernel_initializer="he_normal")(inputs)

    residual_connection = concatenate([CT, skip_connection_input], axis=3)

    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(residual_connection)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)

    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)


    return C


def unet_model(input_size, filters, n_classes):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)

    # Contracting Path (encoding)
    C1, S1 = encoding_block(inputs, filters, max_pooling=True)
    C2, S2 = encoding_block(C1, filters * 2, max_pooling=True)
    C3, S3 = encoding_block(C2, filters * 4, max_pooling=True)
    C4, S4 = encoding_block(C3, filters * 8, max_pooling=True)

    C5, S5 = encoding_block(C4, filters * 16, max_pooling=False)

    # Expanding Path (decoding)
    U6 = decoding_block(C5, S4, filters * 8)
    U7 = decoding_block(U6, S3,  filters * 4)
    U8 = decoding_block(U7, S2,  filters = filters * 2)
    U9 = decoding_block(U8, S1,  filters = filters)

    C10 = Conv2D(filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(U9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    C11 = Conv2D(filters = n_classes, kernel_size = (1,1), activation='sigmoid', padding='same')(C10)

    model = Model(inputs=inputs, outputs=C11)

    return model

img_height = 256
img_width = 256
num_channels = 3
filters = 32
n_classes = 13

model = unet_model((img_height, img_width, num_channels), filters=32, n_classes=23)
model.summary()


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
batch_size = 8
epochs = 30


history = model.fit(train_dataset,
                    validation_data = validation_dataset,
                    epochs = epochs,
                    verbose=1,
                    callbacks = [callback, reduce_lr],
                    batch_size = batch_size,
                    shuffle = True)



acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


model.save('carla-image-segmentation-model.h5')

train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 8)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 8)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 8)


print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


def create_mask(dataset, model):
        true_masks, predicted_masks = [], []
        for images, masks in dataset:
            pred_masks = model.predict(images)
            pred_masks = tf.expand_dims(tf.argmax(pred_masks, axis=-1), axis=-1)
            true_masks.extend(masks)
            predicted_masks.extend(pred_masks)

        true_masks = np.array(true_masks)
        predicted_masks = np.array(predicted_masks)

        return true_masks, predicted_masks

true_train_masks, predicted_train_masks = create_mask(train_dataset, model)
true_validation_masks, predicted_validation_masks = create_mask(validation_dataset, model)
true_test_masks, predicted_test_masks = create_mask(test_dataset, model)


def evaluate_model(true_masks, predicted_masks, n_classes, smooth = 1e-6):

        """
        Evaluates semantic segmentation model

        Argument:
            true_masks: ground truth segmentations
            predicted_masks: predicted segmentations
            n_classes: number of segmentation classes
            smooth: a minute float digit added to denominators to avoid error from a zero division

        Returns:
            class_wise_evaluations: a dictionary containing evaluation metric
                                    outputs the for each segmentation class
            overall_evaluations: a dictionary containing evaluation metric
                                 outputs the for all segmentation classes
            """
        # Create empty lists to store evaluation metric outputs
        class_wise_true_positives, class_wise_true_negatives = [],[]
        class_wise_false_positives, class_wise_false_negatives = [],[]
        class_wise_precisions, class_wise_recalls = [],[]
        class_wise_specificities, class_wise_ious = [],[]
        class_wise_tdrs, class_wise_f1_scores = [],[]
        classes = []

        for clas in range(n_classes):
            true_positives, true_negatives, false_positives, false_negatives = 0,0,0,0
            precisions, recalls, specificities, ious, f1_scores, tdrs = 0,0,0,0,0,0

            number_of_masks = true_masks.shape[0]

            for mask_id in range(number_of_masks):
                true_positive = np.sum(np.logical_and(true_masks[mask_id]==clas, predicted_masks[mask_id]==clas))
                true_negative = np.sum(np.logical_and(true_masks[mask_id]!=clas, predicted_masks[mask_id]!=clas))
                false_positive = np.sum(np.logical_and(true_masks[mask_id]!=clas, predicted_masks[mask_id]==clas))
                false_negative = np.sum(np.logical_and(true_masks[mask_id]==clas, predicted_masks[mask_id]!=clas))

                true_positives += true_positive
                true_negatives += true_negative
                false_positives += false_positive
                false_negatives += false_negative

            recall = round(true_positives/(true_positives + false_negatives + smooth), 2)
            precision = round(true_positives/(true_positives + false_positives + smooth), 2)
            specificity = round(true_negatives/(true_negatives + false_positives + smooth), 2)
            tdr = round((1 - (false_negatives/(true_positives + false_negatives + smooth))), 2)
            iou = round(true_positives/(true_positives + false_negatives + false_positives + smooth), 2)
            f1_score = round((2 * precision * recall)/(precision + recall + smooth), 2)

            class_wise_true_positives.append(true_positives)
            class_wise_true_negatives.append(true_negatives)
            class_wise_false_positives.append(false_positives)
            class_wise_false_negatives.append(false_negatives)
            class_wise_recalls.append(recall)
            class_wise_precisions.append(precision)
            class_wise_specificities.append(specificity)
            class_wise_ious.append(iou)
            class_wise_tdrs.append(tdr)
            class_wise_f1_scores.append(f1_score)
            classes.append("Class " + str(clas+1))
            # class_wise_pixel_accuracies.append(pixel_accuracy)

        total_true_positives = np.sum(class_wise_true_positives)
        total_true_negatives = np.sum(class_wise_true_negatives)
        total_false_positives = np.sum(class_wise_false_positives)
        total_false_negatives = np.sum(class_wise_false_negatives)
        mean_recall = round(np.average(np.array(class_wise_recalls)), 2)
        mean_precision = round(np.average(np.array(class_wise_precisions)), 2)
        mean_specificity = round(np.average(np.array(class_wise_specificities)), 2)
        mean_iou = round(np.average(np.array(class_wise_ious)), 2)
        mean_tdr = round(np.average(np.array(class_wise_tdrs)), 2)
        mean_f1_score = round(np.average(np.array(class_wise_f1_scores)), 2)

        class_wise_evaluations = {"Class": classes,
                                  "True Positive Pixels": class_wise_true_positives,
                                  "True Negative Pixels": class_wise_true_negatives,
                                  "False Positive Pixels": class_wise_false_positives,
                                  "False Negative Pixels": class_wise_false_negatives,
                                  "Recall": class_wise_recalls,
                                  "Precision": class_wise_precisions,
                                  "Specificity": class_wise_specificities,
                                  "IoU": class_wise_ious,
                                  "TDR": class_wise_tdrs,
                                  "F1-Score": class_wise_f1_scores}

        overall_evaluations = {"Class": "All Classes",
                            "True Positive Pixels": total_true_positives,
                            "True Negative Pixels": total_true_negatives,
                            "False Positive Pixels": total_false_positives,
                            "False Negative Pixels": total_false_negatives,
                            "Recall": mean_recall,
                            "Precision": mean_precision,
                            "Specificity": mean_specificity,
                            "IoU": mean_iou,
                            "TDR": mean_tdr,
                            "F1-Score": mean_f1_score}

        evaluations = {"Overall Evaluations": overall_evaluations,
                       "Class-wise Evaluations": class_wise_evaluations}

        return evaluations

def show_evaluations(evaluations,
                         metrics=["Recall", "Precision", "Specificity", "IoU", "TDR", "F1 Score"],
                         class_list=None,
                         display_evaluations="All"):
        """
        Returns a pandas dataframe containing specified metrics

            Arguments:
                evaluations: evaluation output from the evaluate_model function
                metrics: a list containing one or more of the following metrics:
                         'True Positive', 'True Negative', 'False Positive', 'False Negative',
                         'Recall', 'Precision', 'Specificity', 'F1 Score', 'IoU', 'TDR'
                display_evaluations: one of 'All' to display both overall and class-wise evaluations,
                                     'Overall' to display only the overall evaluations,
                                     'Class-wise' to display only the classwise evaluations.
                class_list: list or tuple containing names of segmentation class.
        """

        # Split evaluations into overall and class-wise evaluations
        overall_evaluations = evaluations["Overall Evaluations"]
        class_wise_evaluations = evaluations["Class-wise Evaluations"]

        # Validate list of metrics
        for metric_id in range(len(metrics)):
            metric = metrics[metric_id]
            if metric not in overall_evaluations:
                raise ValueError("'metrics argument' not properly defined. "
                                "Kindly create a list containing one or more of the following metrics: "
                                 "'True Positive', 'True Negative', 'False Positive', 'False Negative', "
                                 "'Recall', 'Precision', 'Specificity', 'F1 Score', 'IoU', 'TDR'")

        # Check if class_list is none
        if class_list != None and all(isinstance(class_, str) for class_ in class_list):
            if len(class_list) == len(class_wise_evaluations["Class"]):
                class_list = [class_list]
            else:
                raise ValueError("class_list argument' not properly defined. "
                                 "List is either shorter or longer than segmentation classes")
        else:
            class_list = [class_wise_evaluations["Class"]]

        # Extract data from the evaluations
        overall_data = [overall_evaluations["Class"]] + [overall_evaluations[metrics[metric_id]] for metric_id in range(len(metrics))]
        classwise_data = class_list + [class_wise_evaluations[metrics[metric_id]] for metric_id in range(len(metrics))]
        overall_data = np.array(overall_data).reshape(1,-1)
        classwise_data = np.array(classwise_data).transpose()

        # Determine the type of evaluation report to display
        if display_evaluations.lower() == "all":
            data = np.concatenate((overall_data, classwise_data), axis=0)
        elif display_evaluations.lower() == "overall":
            data = overall_data
        elif display_evaluations.lower() == "class-wise" or "classwise":
            data = classwise_data
        else:
            raise ValueError("Display argument are not properly defined."
                            "Kindly use 'All' to display both overall and class-wise evaluations."
                            "Use 'Overall' to display only the overall evaluations."
                            "Or use 'Class-wise' to display only the class-wise evaluations")


        # Create evaluation report as a pandas dataframe
        dataframe = pd.DataFrame(data)
        dataframe_titles = ["Class"] + metrics
        dataframe.columns = dataframe_titles
        # dataframe = dataframe.set_index(dataframe_titles[0], col_level=1)

        return dataframe

model_evaluation_on_train_dataset = evaluate_model(true_train_masks, predicted_train_masks, n_classes=13)

show_evaluations(model_evaluation_on_train_dataset,
                 metrics=["Recall", "Precision", "Specificity", "IoU", "TDR", "F1-Score"],
                 class_list=None,
                 display_evaluations="All")


model_evaluation_on_validation_dataset = evaluate_model(true_validation_masks, predicted_validation_masks, n_classes=13)

show_evaluations(model_evaluation_on_validation_dataset,
                 metrics=["Recall", "Precision", "Specificity", "IoU", "TDR", "F1-Score"],
                 class_list=None,
                 display_evaluations="All")



model_evaluation_on_test_dataset = evaluate_model(true_test_masks, predicted_test_masks, n_classes=13)

show_evaluations(model_evaluation_on_test_dataset,
                 metrics=["Recall", "Precision", "Specificity", "IoU", "TDR", "F1-Score"],
                 class_list=None,
                 display_evaluations="All")


model = load_model('carla-image-segmentation-model.h5')

def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, axis=-1)

        return pred_mask[0]

def display(display_list):
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()



def show_predictions(dataset, num):
        """
        Displays the first image of each of the num batches
        """
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions(train_dataset, 6)
show_predictions(validation_dataset, 6)
show_predictions(test_dataset, 6)
