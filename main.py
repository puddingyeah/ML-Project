import torch
from model import GCN
from data_preparation import load_data, split_data, get_dataloaders
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging
import argparse
import os

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # for data in tqdm(loader, desc="Training"):
    for data in loader:
        data = data.to(device)  
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # for data in tqdm(loader, desc="Testing"):
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def main(args):
    data_list = load_data(args.data_path)
    train_data, test_data = split_data(data_list)
    train_loader, test_loader = get_dataloaders(train_data, test_data, batch_size=args.batch_size)

    # log_path 记录task1 or task 2, 格式：task{args.task}_{args.lr}_{args.epochs}_{args.batch_size}.log
    log_path = os.path.join(args.log_dir, f'task{args.task}_{args.lr}_{args.epochs}_{args.batch_size}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Training parameters: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = GCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = test(model, test_loader, criterion, device)
        logging.info(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # model_path 格式：model_task{args.task}_{args.lr}_{args.epochs}_{args.batch_size}.pt
            model_path = os.path.join(args.model_path, f'model_task{args.task}_{args.lr}_{args.epochs}_{args.batch_size}.pt')
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved new best model at {model_path} with Test Loss: {best_test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Graph Convolutional Network")
    parser.add_argument('--data_path', type=str, default='/path/to/data', help='Path to the dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to log file')
    parser.add_argument('--model_path', type=str, default='./model', help='Directory to save models')
    parser.add_argument('--task', type=int, default=1, help='Task number')
    args = parser.parse_args()

    main(args)
