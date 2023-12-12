import torch
import matplotlib.pyplot as plt
from Network.train_and_evaluate import Evaluator
from Data_Preprocess.Transform_MNIST import MNIST_Transformer
from Data_Preprocess.Download_MNIST import download_MNIST_data

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Specify the folder where you want to save the dataset
data_location = "../../data/"
experiment_name = "MNIST_scaled_and_translated"
test_location = data_location + experiment_name + "/"

##################### Constants ######################
data_creation = True
training = True
plotting = training and True

##################### Data creation ######################

# Specify parameters
image_width = 50
image_height = 50

if data_creation:
    train_images, train_labels, test_images, test_labels = download_MNIST_data(data_location)

    # Create a transformer object
    transformer = MNIST_Transformer(train_images, train_labels, test_images, test_labels, image_size=(image_width, image_height))

    # Create the dataset
    transformer.create_rgbd_MNIST_with_transforms(
        test_location, 
        train_transforms=[
            lambda img, depth: transformer.scale_and_place(img, depth, scale_range=(0.2, 1.0), placement_range=(0, 0.8)),
            # lambda img, depth: transformer.add_background(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_gradient(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_noise(img, depth, img_noise_range=(0, 255)),
            # lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 10), occlusion_color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 15), occlusion_color_range=(255, 255, 255)),
        ],
        test_transforms=[
            lambda img, depth: transformer.scale_and_place(img, depth, scale_range=(0.2, 1.0), placement_range=(0, 0.8)),
            # lambda img, depth: transformer.add_background(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_gradient(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_noise(img, depth, img_noise_range=(150, 255)),
            # lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 10), occlusion_color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 15), occlusion_color_range=(255, 255, 255)),
        ]
    )

########################## RGBD example ##########################

if training:
    max_data_size = 100

    # Create the evaluator object
    evaluator = Evaluator(device, test_location, image_width=image_width, image_height=image_height)

    # Load the dataset
    evaluator.load_dataset()

    # Evaluate the model
    rgbd_accuracy_list, smart_rgbd_accuracy_list, rgb_accuracy_list, predictions = evaluator.evaluate(max_data_size=max_data_size, verbose=True)


# Plot the accuracy over epochs
if plotting:
    x_values = [i*(100 // max_data_size) for i in range(len(rgbd_accuracy_list))]
    plt.plot(x_values, rgbd_accuracy_list, label="RGBD")
    # plt.plot(x_values, smart_rgbd_accuracy_list, label="Smart RGBD")
    plt.plot(x_values, rgb_accuracy_list, label="RGB")
    plt.ylabel("Accuracy")
    plt.xlabel("Data size (percentage)")
    plt.title("Accuracy of a RGBD and RGB classifier on different data sizes")
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(test_location + "results.png")
    plt.show()
    

# Plot the predictions
if plotting:
    prediction_index = 0
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle('Predictions', fontsize=20)
    for i in range(5):
        for j in range(5):
            image = predictions[prediction_index]["image"][i+5*j][:, :, :3]
            image = (image - image.min()) / (image.max() - image.min())  # Normalize image
            ax[i, j].imshow(image)

            label = predictions[prediction_index]['label'][i+5*j]
            rgbd_prediction = predictions[prediction_index]['rgbd_prediction'][i+5*j]
            smart_rgbd_prediction = -1 # predictions[prediction_index]['smart_rgbd_prediction'][i+5*j]
            rgb_prediction = predictions[prediction_index]['rgb_prediction'][i+5*j]
            ax[i, j].set_title(f"Label: {label}\nRGBD Prediction: {rgbd_prediction}\nRGB Prediction: {rgb_prediction}\n Smart RGBD Prediction: {smart_rgbd_prediction}")
            ax[i, j].axis('off')

    plt.show()