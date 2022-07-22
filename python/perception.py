import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_mnist_imgs, Dataset_Dyadic


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv_net(outdim, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, outdim),
        nn.LogSoftmax(dim=1)
    )


def auto_enc(outdim, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(outdim, 128),
        nn.ReLU(),
        nn.Linear(128, 784),
        nn.Sigmoid(),
    )


def mlp(indim, outdim, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(indim, 32),
        nn.ReLU(),
        nn.Linear(32, outdim),
        nn.LogSoftmax(dim=1),
    )


class PairCMLP(nn.Module):
    outdim = 10

    def __init__(self, cov_outdim, mlp_outdim):
        super(PairCMLP, self).__init__()
        self.cov_outdim = cov_outdim
        self.mlp_outdim = mlp_outdim
        self.enc = conv_net(cov_outdim)
        self.mlp = mlp(cov_outdim*2, mlp_outdim)

    def forward(self, x):
        encoded0 = self.enc(x[:, 0, :, :])
        encoded1 = self.enc(x[:, 1, :, :])
        encoded = torch.cat((encoded0, encoded1), 1)
        output = self.mlp(encoded)
        return output

    def loss_function(self, pred, y):
        return F.nll_loss(pred, y)
        # return F.binary_cross_entropy(pred, y.view(y.shape[0], -1))


class LSTM(nn.Module):
    """A (Bi)LSTM Model.

    Attributes:
        num_layers: the number of LSTM layers (number of stacked LSTM models) in the network.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
        activation: the activation function.
        bidirectional: the flag for bidirectional LSTM
        dropout: the dropout rate if num_layers > 1
    """

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim,
                 bidirectional=False, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(self.in_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout,
                            batch_first=True)
        fc_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(fc_dim, self.out_dim)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out[:, -1, :])
        outputs = torch.sigmoid(outputs)
        return outputs

    def loss_function(self, pred, y):
        return F.binary_cross_entropy(pred, y.view(y.shape[0], -1))


class Net(nn.Module):
    outdim = 10

    def __init__(self, outdim):
        super(Net, self).__init__()
        self.outdim = outdim
        self.enc = conv_net(outdim)

    def forward(self, x):
        output = self.enc(x)
        return output

    def loss_function(self, pred, y):
        return F.nll_loss(pred, y)


class AutoEncoder(nn.Module):
    outdim = 10

    def __init__(self, outdim):
        super(AutoEncoder, self).__init__()
        self.outdim = outdim
        self.enc = conv_net(outdim)
        self.dec = auto_enc(outdim)

    def forward(self, x):
        x = self.enc(x)
        output = self.dec(x)
        return output

    def loss_function(self, pred, y):
        return F.binary_cross_entropy(pred, y.view(-1, 784), reduction='sum')


def train(model, device, train_loader, optimizer, epoch,
          log_interval=10000, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += model.loss_function(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('-- Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_ae(model, device, train_loader, optimizer, epoch,
             log_interval=10000, dry_run=False):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test_ae(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += model.loss_function(output, data).item()

    test_loss /= len(test_loader.dataset)

    print('-- Test set: Average loss: {:.4f}'.format(test_loss))


def predict_idx_prob(model, img_data, idxs, use_cuda=True):
    img_tensor, _ = get_mnist_imgs(img_data, idxs, use_cuda=use_cuda)
    # print(img_tensor, targets)
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(img_tensor))
    return pred


def predict_idx_labels(model, img_data, idxs, label_names, use_cuda=True):
    img_tensor, _ = get_mnist_imgs(img_data, idxs, use_cuda=use_cuda)
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(img_tensor))
    _, indices = torch.max(pred, 1)
    return [label_names[i] for i in indices]


def pseudo_label_probs(model, indices, all_examples, all_imgs_data, use_cuda=True):
    re = []
    for i in indices:
        sample = all_examples[i]
        idxs = sample.x_idxs
        # predict the probability values
        pred = predict_idx_prob(model, all_imgs_data,
                                idxs, use_cuda=use_cuda).tolist()
        re.append(pred)
    return re


def get_dyadic_mnist_data(dataset, idx_pairs):
    targets = []
    mnist_targets = dataset.targets
    for p in idx_pairs:
        t = 1 if mnist_targets[p[0]] > mnist_targets[p[1]] else 0
        targets.append(t)
    dyadic_data = DyadicDataset(idx_pairs, targets, dataset)

    return dyadic_data


def predict_pair_prob(model, img_data, pairs):
    dyadic_data = get_dyadic_mnist_data(img_data, pairs)
    dyadic_tensor = dyadic_data.get_data()
    targets = dyadic_data.targets.tolist()
    # print(img_tensor, targets)
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(dyadic_tensor)).tolist()
    return pred, targets


def pseudo_label_pairs_probs(model, idxs_pair_groups, all_imgs_data):
    probs = []
    targets = []
    for pairs in idxs_pair_groups:
        pred, tgts = predict_pair_prob(model, all_imgs_data, pairs)
        probs.append(pred)
        targets.append(tgts)
    return probs, targets
