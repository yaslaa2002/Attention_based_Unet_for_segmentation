# -*- coding: utf-8 -*-
# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import itertools
from sklearn.metrics import confusion_matrix
from skimage.segmentation import mark_boundaries
from data import dataLoaderMaking
from model import UNet2d
from model import UNetAug2D


def evaluation_pixel(model, val_loader, criterion, device, num_classes=5):
    """
    Evaluates the model's performance on a validation set using pixel-wise accuracy and loss.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function used for evaluation (e.g., CrossEntropyLoss).
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        num_classes: The number of classes in the classification task, default is 5.

    Returns:
        val_loss: The average validation loss over the entire dataset.
        overall_accuracy: The overall pixel-wise accuracy across all classes.
        class_correct: List of correct pixel counts for each class.
        class_total: List of total pixel counts for each class.
    """
    val_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()  # Set the model to evaluation mode
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Remove channel dimension if necessary

        with torch.no_grad():
            output = model(data)

        # Calculate the loss
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Pixel-wise predictions
        _, preds = torch.max(output, dim=1)

        # Calculate correct pixels for each class
        for i in range(num_classes):
            class_mask = (label == i)
            class_correct[i] += torch.sum((preds == i) & class_mask).item()
            class_total[i] += torch.sum(class_mask).item()

    # Calculate the average loss
    val_loss /= len(val_loader.dataset)

    # Display class-wise accuracies
    print('Validation loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy_class = 100.0 * class_correct[i] / class_total[i]
            print('Accuracy for class ' + str(i) + ': ' + str(round(accuracy_class, 2)) + '% (' + str(class_correct[i]) + '/' + str(class_total[i]) + ')')
        else:
            print('Accuracy for class ' + str(i) + ': N/A (no pixels)')

    # Calculate overall accuracy
    total_correct_pixels = sum(class_correct)
    total_pixels = sum(class_total)
    overall_accuracy = 100.0 * total_correct_pixels / total_pixels

    print('Overall pixel-wise accuracy: {:.2f}% ({}/{})'.format(overall_accuracy, total_correct_pixels, total_pixels))

    return val_loss, overall_accuracy, class_correct, class_total

def evaluation_with_dice(model, val_loader, criterion, device, num_classes=5):
    """
    Evaluates the model's performance on a validation set using pixel-wise loss and Dice Score.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function used for evaluation (e.g., CrossEntropyLoss).
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        num_classes: The number of classes in the classification task, default is 5.

    Returns:
        val_loss: The average validation loss over the entire dataset.
        global_dice: The overall average Dice score across all classes.
        dice_scores: List of Dice scores for each class.
    """
    val_loss = 0.0
    dice_scores = [0.0] * num_classes  # Dice scores per class
    class_total = [0] * num_classes

    model.eval()  # Set the model to evaluation mode
    for data, label in val_loader:
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long).squeeze(1)  # Remove channel dimension if necessary

        with torch.no_grad():
            output = model(data)

        # Calculate the loss
        loss = criterion(output, label)
        val_loss += loss.item() * data.size(0)

        # Pixel-wise predictions
        _, preds = torch.max(output, dim=1)

        # Calculate Dice Score for each class
        for i in range(num_classes):
            class_mask = (label == i)
            pred_mask = (preds == i)

            # Intersection and union for Dice Score
            intersection = torch.sum(pred_mask & class_mask).item()
            union = torch.sum(pred_mask).item() + torch.sum(class_mask).item()

            if union > 0:  # Avoid division by zero
                dice_scores[i] += 2.0 * intersection / union
                class_total[i] += 1

    # Calculate the average loss
    val_loss /= len(val_loader.dataset)

    # Display Dice scores per class
    print('Validation loss: {:.6f}'.format(val_loss))
    for i in range(num_classes):
        if class_total[i] > 0:
            average_dice = dice_scores[i] / class_total[i]
            print('Dice score for class ' + str(i) + ': ' + str(round(average_dice, 4)))
        else:
            print('Dice score for class'+str(i)+': N/A (no pixels)')

    # Calculate the global average Dice score
    global_dice = sum(dice_scores) / sum(class_total)
    print('Overall Dice score: {:.4f}'.format(global_dice))

    return val_loss, global_dice, dice_scores

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix with an optional normalization option.

    Args:
        cm: Confusion matrix to plot, typically a 2D numpy array.
        classes: List of class names corresponding to the matrix indices.
        normalize: Boolean flag indicating whether to normalize the confusion matrix (default is False).
        title: Title for the plot (default is 'Confusion Matrix').
        cmap: Colormap to use for the plot (default is plt.cm.Blues).
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_confusion_matrix(model, val_loader, device, num_classes=5):
    """
    Evaluates the model's performance on the validation set and displays the confusion matrix.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader for the validation set.
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        num_classes: The number of classes in the classification task, default is 5.
    """
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for data, mask in val_loader:
            data = data.to(device)
            mask = mask.squeeze(1).to(device).long()  # Explicit conversion to integers

            # Model predictions
            output = model(data)
            predicted_classes = torch.argmax(output, dim=1).long()  # Explicit conversion to integers

            # Collect true and predicted values
            y_true.extend(mask.cpu().numpy().flatten().astype(int))  # Explicit conversion to integers
            y_pred.extend(predicted_classes.cpu().numpy().flatten().astype(int))  # Explicit conversion to integers

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    # Display the confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=['Classe ' + str(i) for i in range(num_classes)])
    plt.show()

def display_random_prediction(model, val_loader, device):
    """
    Displays a random image from the validation set along with the ground truth and predicted contours.

    Args:
        model: The trained model to make predictions.
        val_loader: DataLoader for the validation set.
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
    """
    # Set the model to evaluation mode
    model.eval()

    # Select a random batch from the validation loader
    data_iter = iter(val_loader)
    random_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_index):
        next(data_iter)  # Skip to the random index

    # Retrieve a batch of data
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Make predictions
    with torch.no_grad():
        prediction = model(data)
        predicted_classes = torch.argmax(prediction, dim=1)

    # Choose a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()
    predicted_image = predicted_classes[random_image_index].cpu().numpy()

    # Normalize the input image for display
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Add contours to the input image
    image_with_contours = mark_boundaries(data_image, mask_image, color=(0, 1, 0))  # Green contours (ground truth)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image, color=(1, 0, 0))  # Red contours (predictions)

    # Display the image with contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('True contours (green) and predicted contours (red)')
    plt.axis('off')
    plt.show()

def display_comparison(models, val_loader, device, class_names=None):
    """
    Displays a random image with the ground truth mask and predictions from two models.

    Args:
        models: List of trained models [model_1, model_2].
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        class_names: List of class names corresponding to the indices, optional.
    """
    # Set both models to evaluation mode
    for model in models:
        model.eval()

    # Select a random batch from the validation DataLoader
    data_iter = iter(val_loader)
    random_batch_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_batch_index):
        next(data_iter)  # Skip to the selected random batch

    # Get a batch of data and its masks
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Select a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Get predictions from both models
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(data)
            predicted_classes = torch.argmax(prediction, dim=1)
            predictions.append(predicted_classes[random_image_index].cpu().numpy())

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Plot ground truth and predictions
    plt.figure(figsize=(15, 5))

    # Plot the input image
    plt.subplot(1, len(models) + 2, 1)
    plt.imshow(data_image, cmap='gray')
    plt.title('Input image')
    plt.axis('off')

    # Plot the ground truth mask
    plt.subplot(1, len(models) + 2, 2)
    plt.imshow(mask_image, cmap='viridis')
    if class_names:
        plt.title('Ground truth')
    else:
        plt.title('Ground truth mask')
    plt.axis('off')

    # Plot predictions for each model
    for i, prediction in enumerate(predictions):
        plt.subplot(1, len(models) + 2, i + 3)
        plt.imshow(prediction, cmap='viridis')
        title = 'Prediction (Model ' + str(i + 1) + ')'
        if class_names:
            title += '\n' + str(class_names)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_prediction_for_class(model, val_loader, device, target_class):
    """
    Displays a prediction for a specific class, highlighting true and predicted contours,
    and shows the full mask for the target class.
    
    Args:
        model: Trained segmentation model.
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        target_class: Index of the target class to display (e.g., 1 for liver).
    """
    # Set the model to evaluation mode
    model.eval()

    # Iterate through the validation DataLoader to find a batch with the target class
    for data, mask in val_loader:
        data = data.to(device, dtype=torch.float32)
        mask = mask.squeeze(1).to(device, dtype=torch.long)

        # Check if any mask contains the target class
        contains_target_class = (mask == target_class).any(dim=(1, 2))  # Boolean array per image in batch
        if contains_target_class.any():  # If at least one image contains the target class
            break
    else:
        print("No images with class " + str(target_class) + " were found in the validation set.")
        return

    # Select a random image that contains the target class
    indices_with_target_class = torch.where(contains_target_class)[0]
    random_image_index = random.choice(indices_with_target_class)  # Randomly select one
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Make predictions
    with torch.no_grad():
        prediction = model(data)
        predicted_classes = torch.argmax(prediction, dim=1)
        predicted_image = predicted_classes[random_image_index].cpu().numpy()

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Create a smoothed mask for contours to avoid dotted lines (more continuous boundaries)
    smoothed_mask = np.pad(mask_image == target_class, pad_width=1, mode='constant', constant_values=False)
    smoothed_mask = smoothed_mask[1:-1, 1:-1]  # Remove padding for final mask

    smoothed_pred = np.pad(predicted_image == target_class, pad_width=1, mode='constant', constant_values=False)
    smoothed_pred = smoothed_pred[1:-1, 1:-1]  # Remove padding for final predicted mask

    # Add contours for the target class (Ground truth and Prediction)
    image_with_contours = mark_boundaries(data_image, smoothed_mask, color=(0, 1, 0), mode='thick')  # Green for ground truth
    image_with_contours = mark_boundaries(image_with_contours, smoothed_pred, color=(1, 0, 0), mode='thick')  # Red for predictions

    # Create a full mask for the target class
    full_mask = mask_image == target_class
    full_pred_mask = predicted_image == target_class

    # Display the images
    plt.figure(figsize=(12, 12))

    # Plot the original image with contours
    plt.subplot(2, 2, 1)
    plt.imshow(image_with_contours)
    plt.title('Class ' + str(target_class) + ': True contours (green) and predicted contours (red)')
    plt.axis('off')

    # Plot the full ground truth mask for the target class
    plt.subplot(2, 2, 2)
    plt.imshow(full_mask, cmap='gray')
    plt.title('Ground truth mask for class ' + str(target_class))
    plt.axis('off')

    # Plot the full predicted mask for the target class
    plt.subplot(2, 2, 3)
    plt.imshow(full_pred_mask, cmap='gray')
    plt.title('Predicted mask for class'+ str(target_class))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_attention_with_outlines_and_scales(model, data_loader, device):
    """
    Displays attention maps overlaid on the source image with outlined organs,
    and includes a scale bar for each attention map. The images are aligned horizontally.
    The first image also shows the model's prediction in blue contours.

    Args:
        model: The PyTorch model generating the attention maps.
        data_loader: DataLoader containing the data (images and label masks).
        device: Device ('cuda' or 'cpu').
    """
    # Set the model to evaluation mode
    model.eval()

    # Retrieve a batch of data from the DataLoader
    data_iter = iter(data_loader)
    random_index = random.randint(0, len(data_loader) - 1)
    for _ in range(random_index):
        next(data_iter)  # Skip to the random index

    input_tensor, labels = next(data_iter)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    labels = labels.squeeze(1).to(device, dtype=torch.long)  # Ensure labels are in the correct format

    # Pass the data through the model to generate attention maps
    with torch.no_grad():
        output, attention_maps = model(input_tensor, return_attention=True)
        predicted_classes = torch.argmax(output, dim=1)

    # Display for the first image in the batch (index 0)
    image_index = 0
    original_image = input_tensor[image_index].cpu().squeeze(0).numpy()
    label_image = labels[image_index].cpu().numpy()
    predicted_image = predicted_classes[image_index].cpu().numpy()

    # Add contours for labels and predictions
    image_with_labels = mark_boundaries(original_image, label_image, color=(1, 1, 0))  # Contours jaunes
    image_with_predictions = mark_boundaries(image_with_labels, predicted_image, color=(0, 0, 1))  # Contours bleus

    # Create a figure to align images horizontally
    num_attention_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_attention_maps + 1, figsize=(20, 5))

    # Display the original image with labels and prediction
    axes[0].imshow(image_with_predictions, cmap='gray')
    axes[0].set_title('Image, labels (yellow) and prediction (blue)')
    axes[0].axis('off')

    # Display the attention maps
    for i, attention_map in enumerate(attention_maps):
        # Normalize the attention map
        att_map = attention_map[image_index].squeeze(0).cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

        # Add the original image and the attention map
        im = axes[i + 1].imshow(att_map, cmap='jet')  # Overlay the attention map

        # Add a scale bar (legend)
        cbar = plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
        cbar.set_label('Visualization of the attention', rotation=270, labelpad=15)

        axes[i + 1].set_title('Attention map ' + str(i + 1))
        axes[i + 1].axis('off')

    # Adjust layout
    plt.subplots_adjust(wspace=0.2)  # Reduce space between images
    plt.tight_layout()
    plt.show()

def display_random_prediction_two_models(model1, model2, val_loader, device):
    """
    Displays a random prediction comparing two models on a validation batch.

    Args:
        model1: First PyTorch model.
        model2: Second PyTorch model.
        val_loader: DataLoader containing validation data.
        device: Device ('cuda' or 'cpu').
    """
    # Set models to evaluation mode
    model1.eval()
    model2.eval()

    # Select a random batch from the validation loader
    data_iter = iter(val_loader)
    random_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_index):
        next(data_iter)  # Skip to the random index
    # Retrieve a batch of data
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Make predictions for both models
    with torch.no_grad():
        prediction1 = model1(data)
        predicted_classes1 = torch.argmax(prediction1, dim=1)

        prediction2 = model2(data)
        predicted_classes2 = torch.argmax(prediction2, dim=1)

    # Choose a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()
    predicted_image1 = predicted_classes1[random_image_index].cpu().numpy()
    predicted_image2 = predicted_classes2[random_image_index].cpu().numpy()

    # Normalize the input image for display
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Add contours to the input image
    image_with_contours = mark_boundaries(data_image, mask_image, color=(0, 1, 0))  # Green contours (ground truth)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image1, color=(0, 0, 1))  # Blue contours (model 1)
    image_with_contours = mark_boundaries(image_with_contours, predicted_image2, color=(1, 0, 0))  # Red contours (model 2)

    # Display the image with contours
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.title('True mask (green), Classic U-Net (blue), Attention U-Net (red)')
    plt.axis('off')
    plt.show()

def display_comparison(models, val_loader, device, class_names=None):
    """
    Displays a random image with the ground truth mask and predictions from two models.

    Args:
        models: List of trained models [model_1, model_2].
        val_loader: DataLoader for the validation set.
        device: Device to use (CPU or CUDA).
        class_names: List of class names corresponding to the indices, optional.
    """
    # Set both models to evaluation mode
    for model in models:
        model.eval()

    # Select a random batch from the validation DataLoader
    data_iter = iter(val_loader)
    random_batch_index = random.randint(0, len(val_loader) - 1)

    for _ in range(random_batch_index):
        next(data_iter)  # Skip to the selected random batch

    # Get a batch of data and its masks
    data, mask = next(data_iter)
    data = data.to(device, dtype=torch.float32)
    mask = mask.squeeze(1).to(device, dtype=torch.long)

    # Select a random image from the batch
    random_image_index = random.randint(0, data.size(0) - 1)
    data_image = data[random_image_index].cpu().squeeze(0).numpy()
    mask_image = mask[random_image_index].cpu().numpy()

    # Get predictions from both models
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(data)
            predicted_classes = torch.argmax(prediction, dim=1)
            predictions.append(predicted_classes[random_image_index].cpu().numpy())

    # Normalize the input image for visualization
    data_image = (data_image - data_image.min()) / (data_image.max() - data_image.min())

    # Plot ground truth and predictions
    plt.figure(figsize=(15, 5))

    # Plot the input image
    plt.subplot(1, len(models) + 2, 1)
    plt.imshow(data_image, cmap='gray')
    plt.title('Input image')
    plt.axis('off')

    # Plot the ground truth mask
    plt.subplot(1, len(models) + 2, 2)
    plt.imshow(mask_image, cmap='viridis')
    if class_names:
        plt.title('Ground truth')
    else:
        plt.title('Ground truth mask')
    plt.axis('off')

    # Plot predictions for each model
    for i, prediction in enumerate(predictions):
        plt.subplot(1, len(models) + 2, i + 3)
        plt.imshow(prediction, cmap='viridis')
        title = 'Prediction (model ' + str(i + 1) + ')'
        if class_names:
            title += '\n' + str(class_names)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Create a function to plot the two models on the same mask and compare
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="UNet Model Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model 1 (.pth file).")
    parser.add_argument('--model_path2', type=str, required=True, help="Path to the trained model 2 (.pth file).")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for evaluation.")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes in the segmentation task.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument('--image_size', type=int, default=128, help="Size to which images should be resized.")
    parser.add_argument('--criterion', type=str, default="cross_entropy", choices=['cross_entropy'], help="Loss function.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device(args.device)

    # Data loading
    print("Loading data from {}...".format(args.data_dir))
    # Create dataset and DataLoader
    train_loader, test_loader, val_loader = dataLoaderMaking(namefile=args.data_dir, target_shape=(256, 256), batch_size=args.batch_size)

    # Load the model
    print("Loading model 1 from {}...".format(args.model_path))
    model = UNet2d()
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    model = model.to(device)
    print("Evaluation of your first model (Classic U-Net)")

    # Define loss function
    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    # Perform evaluations
    print("Evaluating model...")
    evaluation_pixel(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Evaluating with Dice score...")
    # evaluation_with_dice(model, test_loader, criterion, device, num_classes=args.num_classes)

    print("Generating confusion matrix...")
    # evaluate_confusion_matrix(model, test_loader, device, num_classes=args.num_classes)

    print("Displaying random predictions...")
    # display_random_prediction(model, test_loader, device)

    print("Displaying predictions for each class...")
    print("Class 1")
    # display_prediction_for_class(model, test_loader, device, 1)
    print("Class 2")
    # display_prediction_for_class(model, test_loader, device, 2)
    print("Class 3")
    # display_prediction_for_class(model, test_loader, device, 3)
    print("Class 4")
    # display_prediction_for_class(model, test_loader, device, 4)

    # Load the model 2
    print("Loading model 2 from {}...".format(args.model_path2))
    model2 = UNetAug2D()
    model2.load_state_dict(torch.load((args.model_path2), map_location=torch.device(device)))
    model2 = model2.to(device)
    print("Evaluation of your second model (Attention U-Net)")

    # Define loss function
    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    # Data loading
    print("Loading data from {}...".format(args.data_dir))

    # Perform evaluations
    print("Evaluating model...")
    evaluation_pixel(model2, test_loader, criterion, device, num_classes=args.num_classes)

    print("Evaluating with Dice score...")
    evaluation_with_dice(model2, test_loader, criterion, device, num_classes=args.num_classes)

    print("Generating confusion matrix...")
    evaluate_confusion_matrix(model2, test_loader, device, num_classes=args.num_classes)

    print("Displaying random predictions...")
    display_random_prediction(model2, test_loader, device)

    print("Displaying predictions for each class...")
    print("Class 1")
    display_prediction_for_class(model2, test_loader, device, 1)
    print("Class 2")
    display_prediction_for_class(model2, test_loader, device, 2)
    print("Class 3")
    display_prediction_for_class(model2, test_loader, device, 3)
    print("Class 4")
    display_prediction_for_class(model2, test_loader, device, 4)

    print("Displaying activation map")
    visualize_attention_with_outlines_and_scales(model2, test_loader, device)

    print("Comparison between the two models")

    print("Displaying random predictions of the two models...")
    display_random_prediction_two_models(model, model2, test_loader, device)

    print("Displaying comparison between the two models")
    display_comparison([model, model2], val_loader, device, class_names=None)
