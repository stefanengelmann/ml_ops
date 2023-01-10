import argparse
import sys
import pickle
import os
import matplotlib.pyplot as plt
import wandb
import torch
torch.manual_seed(42)

from model import MyAwesomeModel
sys.path.append('./src/data')
from make_dataset import CorruptMnist

wandb.init(project="ml_ops",entity="stefanengelmann10")

def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-b','--batch_size', default=128, type=int)
    parser.add_argument('-e','--epochs', default=5, type=int)
    parser.add_argument('-d','--device')
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    print("Device: ", device)
    
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model = model.to(device)

    wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size
                    }

    os.chdir('./data/processed')

    with open('dataset_train.pt', 'rb') as f:
    # Deserialize the object and recreate it in memory
        train_set = pickle.load(f)

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epoch = args.epochs
    loss_tracker = []
    for epoch in range(n_epoch):
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
            wandb.log({"Train loss":loss.item()})
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")     

    os.chdir('../../models')   
    torch.save(model.state_dict(), 'trained_model.pt')
        
    plt.plot(loss_tracker, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')

    os.chdir('../reports/figures')
    plt.savefig("training_curve.png")

if __name__ == '__main__':
    train()