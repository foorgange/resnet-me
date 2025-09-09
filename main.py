import argparse
from train import train
from model import ResNet18
from evaluate import evaluate
from utils import load_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--data_dir', type=str, default='./datas/cifar-10-python')
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.model_path)
    else:
        model = ResNet18().to(device)
        model = load_model(model, args.model_path, device)
        from data_loader import get_loaders
        test_loader = get_loaders(args.data_dir, batch_size=args.batch_size)[1]
        acc, f1 = evaluate(model, test_loader, device)
        print(f'Evaluation Accuracy: {acc:.4f}, F1-score: {f1:.4f}')

if __name__ == '__main__':
    main()
