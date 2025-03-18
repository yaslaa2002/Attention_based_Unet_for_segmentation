# Importing libraries 
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from data import dataLoaderMaking
from model import UNet2d,UNetAug2D

# Function to read arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the training dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    return parser.parse_args()



# Modified training function
def training(model, criterion, optimizer, train_loader, val_loader,device,n_epochs,nameFile):
    """
    Trains and validates a model over multiple epochs, saving the model if the validation loss decreases.

    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        device (torch.device): The device to use for training (CPU or CUDA).
        n_epochs (int): The number of epochs to train the model.
        nameFile (str): The file name to save the best model.

    Returns:
        tuple: A tuple containing two lists:
            - train_losses (list): The average training loss for each epoch.
            - valid_losses (list): The average validation loss for each epoch.
    """
    numberSamples = len(train_loader.dataset)
    train_losses, valid_losses = [], []
    valid_loss_min = np.inf
    i = 1
    
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0, 0

        # Training
        model.train()
        for data, label in train_loader:
            data = data.to(device)  # Add the channel dimension
            label = label.squeeze(1).to(device).long()

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            print(epoch,i,":",train_loss)
            i += 1
            

        # Validation
        model.eval()
        for data, label in val_loader:
            data = data.to(device)  # Add the channel dimension
            label = label.squeeze(1).to(device).long()

            with torch.no_grad():
                output = model(data)
            loss = criterion(output, label)
            valid_loss += loss.item() * data.size(0)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        # Save the model if the validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), nameFile)
            valid_loss_min = valid_loss

    return train_losses, valid_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Model Train Script")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for evaluation.")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes in the segmentation task.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument('--image_size', type=int, default=128, help="Size to which images should be resized.")
    parser.add_argument('--criterion', type=str, default="cross_entropy", choices=['cross_entropy'], help="Loss function.")
    args = parser.parse_args()


    # Load the data
    train_loader,test_loader,val_loader = dataLoaderMaking(namefile=args.dataset_path,target_shape = (256, 256),batch_size = args.batch_size)

    # Define the model
    model_class = UNet2d()
    model_augm=UNetAug2D()

    class_weights = torch.tensor([1.04, 30.3, 263.1, 158.7, 270.3]).to(args.device)  # Move the weights to the same device as the model

    # Define the weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam optimizer with a lower learning rate
    optimizer_class = torch.optim.Adam(model_class.parameters(), lr=0.00005)
    optimizer_augm = torch.optim.Adam(model_augm.parameters(), lr=0.00005)

    # Train the model
    train_losses_class, valid_losses_class = training(model_class, criterion, optimizer_class, train_loader, val_loader,args.device,args.epochs,"modelUnetClassque.pt")
    train_losses_augm, valid_losses_caugm = training(model_augm, criterion, optimizer_augm, train_loader, val_loader,args.device,args.epochs,"modelUnetAugmented.pt")
    
