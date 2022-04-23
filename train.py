import torch
import sys
sys.path.append('..')

from models.models import CAGNN

import warnings

warnings.filterwarnings('ignore')
from datasets.dataloader import load_data
from utils.utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0, help='cuda id')
parser.add_argument('--dataset', type=str, default='texas', help='dataset')
parser.add_argument('--split_id', type=int, default=0, help='fixed 10 random splits')
parser.add_argument('--add_edges_ratio', type=float, default=0, help='noisy edges scenario')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--num_layer', type=int, default=2, help='Number of models layer')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--conv_type', type=str, default='gcn', help="conv type in cagnn")
parser.add_argument('--gate_type', type=str, default='convex', help="the gate type for cagnn")
parser.add_argument('--norm_type', type=str, default='l2', help="The norm type for cagnn")
args = parser.parse_args()


def train(epoch, model, features, labels, train_mask, val_mask, test_mask,
                lr=1e-2, dur=10, weight_decay=5e-4, loss_fun=torch.nn.CrossEntropyLoss(),
                ):
    set_seed(202)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], []
    best_val_acc, best_test_acc = 0, 0
    dur = epoch / dur
    best_epoch = 0
    for iter in range(epoch):
        model.train()
        output = model(features)
        loss = masked_loss(output, labels, train_mask, loss_fun)
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features)
            val_loss = masked_loss(output, labels, val_mask, loss_fun)
            val_loss_list.append(val_loss.item())
            train_acc, val_acc, test_acc = evaluate(output, labels, train_mask, val_mask, test_mask)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = iter

        if (iter + 1) % dur == 0:
            print(
                "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f},  val_acc {:.4f}, test_acc{:.4f}".format(
                    iter + 1, np.mean(train_loss_list),  np.mean(val_loss_list), train_acc, val_acc, test_acc))
    print("Best at {} epoch, Val Accuracy {:.4f} Test Accuracy {:.4f}".format(best_epoch, best_val_acc, best_test_acc))
    return model


if __name__ == "__main__":
    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id >= 0 else 'cpu'

    data = load_data(args.dataset, device, split_id=args.split_id, add_ratio=args.add_edges_ratio)
    model = CAGNN(data.g, data.x.shape[1], args.hidden, data.num_of_class, num_layers=args.num_layer, dropout=args.dropout,
                 norm_type=args.norm_type, conv_type=args.conv_type, gate_type=args.gate_type).to(device)

    train(args.epochs, model, features=data.x, labels=data.y,
                train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                lr=args.lr, weight_decay=args.weight_decay, dur=10,)
