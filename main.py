# Main Script
import argparse
import torch
import subprocess
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate U-Net models")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--train', action='store_true', help="Specify to train models")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--load_classic', type=str, default=None, help="Path to classic U-Net model")
    parser.add_argument('--load_aug', type=str, default=None, help="Path to attention U-Net model")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    python_exe = sys.executable
    if args.train: # If the models are not trained, training them
        args_train=[
            python_exe, 'train.py',
            '--data_dir', str(args.data_dir),
            '--epochs',str(args.epochs),
            '--batch_size', str(args.batch_size),  
            '--device', str(args.device),  
        ]
        result1 = subprocess.run(args_train, capture_output=True, text=True)

        print(result1.stdout)  
        if result1.stderr:
            print("Error output:", result1.stderr)
        
    # Evaluation of the models
    args_eval = [ 
            python_exe, 'evaluation.py', 
            '--model_path', str(args.load_classic),
            '--model_path2', str(args.load_aug),
            '--data_dir', str(args.data_dir),
            '--device', str(args.device), 
            '--num_classes', '5',
            '--batch_size', str(args.batch_size), 
            '--image_size', '128',
            '--criterion', 'cross_entropy'
    ]
        
    result2 = subprocess.run(args_eval, capture_output=True, text=True)

    print(result2.stdout)  
    if result2.stderr:  
        print("Error output:", result2.stderr)
