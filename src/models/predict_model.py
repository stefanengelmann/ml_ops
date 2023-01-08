import argparse
import sys
import pickle
import os
import torch
torch.manual_seed(42)

from model import MyAwesomeModel
sys.path.append('./src/data')
from make_dataset import CorruptMnist

def evaluate():
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluating arguments')
        parser.add_argument('load_model_from', default="",help="Assumes the model is located relative to models/ path")
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
        model.load_state_dict(torch.load("./models/"+args.load_model_from))
        model = model.to(device)

        os.chdir('./data/processed')

        with open('dataset_test.pt', 'rb') as f:
        # Deserialize the object and recreate it in memory
            test_set = pickle.load(f)

        dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(device)).sum().item()
            total += y.numel()
            
        print(f"Test set accuracy {correct/total}")

if __name__ == '__main__':
    evaluate()