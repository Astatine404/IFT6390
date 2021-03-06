import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Tuple, Callable, List, NamedTuple
import torchvision
import tqdm
import traceback


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    paddings: Tuple[int, ...] = (0, 0, 0)
    dense_hiddens: Tuple[int, ...] = (256, 256)


# Pytorch preliminaries
def gradient_norm(function: Callable, *tensor_list: List[torch.Tensor]) -> float:
    outputs = function(*tensor_list)
    outputs.backward()

    grad = [tensor.grad for tensor in tensor_list]
    grad = torch.stack(grad)

    return torch.linalg.norm(grad).item()



def jacobian_norm(function: Callable, input_tensor: torch.Tensor) -> float:
    jacobian = autograd.functional.jacobian(function, input_tensor)

    return torch.linalg.norm(jacobian).item()



class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 datapath: str = './data',
                 n_classes: int = 10,
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 activation_name: str = "relu",
                 normalization: bool = True):
        self.train, self.valid, self.test = self.load_dataset(datapath)
        if normalization:
            self.train, self.valid, self.test = self.normalize(self.train, self.valid, self.test)
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0][0].shape
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], net_config,
                                           n_classes, activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], net_config, n_classes, activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = 1e-9

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': [],
                           'train_gradient_norm': []}

    @staticmethod
    def load_dataset(datapath: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        trainset = torchvision.datasets.FashionMNIST(root=datapath,
                                                     download=True, train=True)
        testset = torchvision.datasets.FashionMNIST(root=datapath,
                                                    download=True, train=False)

        X_train = trainset.data.view(-1, 1, 28, 28).float()
        y_train = trainset.targets

        X_ = testset.data.view(-1, 1, 28, 28).float()
        y_ = testset.targets

        X_val = X_[:2000]
        y_val = y_[:2000]

        X_test = X_[2000:]
        y_test = y_[2000:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        modules = []

        modules.append(nn.Flatten())

        modules.append(nn.Linear(input_dim, net_config.dense_hiddens[0]))
        modules.append(activation)

        for idx in range(1, len(net_config.dense_hiddens)):
          modules.append(nn.Linear(net_config.dense_hiddens[idx-1], net_config.dense_hiddens[idx]))
          modules.append(activation)

        modules.append(nn.Linear(net_config.dense_hiddens[-1], n_classes))
        modules.append(nn.Softmax(dim=1))

        model = nn.Sequential(*modules)
        
        return model

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        modules = []

        modules.append(nn.Conv2d(in_channels, net_config.n_channels[0], net_config.kernel_sizes[0], padding=net_config.paddings[0], stride=net_config.strides[0]))
        modules.append(activation)

        for idx in range(1, len(net_config.n_channels)):
          modules.append(nn.MaxPool2d(kernel_size=2))
          modules.append(nn.Conv2d(net_config.n_channels[idx-1], net_config.n_channels[idx], net_config.kernel_sizes[idx], padding=net_config.paddings[idx], stride=net_config.strides[idx]))
          modules.append(activation)

        modules.append(nn.AdaptiveMaxPool2d((4, 4)))
        modules.append(nn.Flatten())

        input_size = 4 * 4 * net_config.n_channels[-1]

        modules.append(nn.Linear(input_size, net_config.dense_hiddens[0]))
        modules.append(activation)

        for idx in range(1, len(net_config.dense_hiddens)):
          modules.append(nn.Linear(net_config.dense_hiddens[idx-1], net_config.dense_hiddens[idx]))
          modules.append(activation)

        modules.append(nn.Linear(net_config.dense_hiddens[-1], n_classes))
        modules.append(nn.Softmax(dim=1))

        model = nn.Sequential(*modules)

        return model

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y)

    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        outputs = self.network(X)
        outputs = torch.clip(outputs, min=self.epsilon, max=1-self.epsilon)
        reverted_target = torch.argmax(y, dim=1) 
        log_softmax = torch.log(outputs)
        loss_fn = nn.NLLLoss()
        loss = loss_fn(log_softmax, reverted_target)

        y_pred = torch.argmax(outputs, dim=1)
        acc = (y_pred == reverted_target).float().mean().item()
        
        return loss, acc

    @staticmethod
    def compute_gradient_norm(network: torch.nn.Module) -> float:
        grad_list = []
        for param in network.parameters():
            grad_list.append(param.grad.flatten())
        grads = torch.cat(grad_list)

        return torch.linalg.norm(grads).item()

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss, acc = self.compute_loss_and_accuracy(X_batch, y_batch)
        loss.backward()
        self.optimizer.step()

        return self.compute_gradient_norm(self.network)

    def log_metrics(self, X_train: torch.Tensor, y_train_oh: torch.Tensor,
                    X_valid: torch.Tensor, y_valid_oh: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train_oh)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid_oh)
        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(float(train_loss))
        self.train_logs['validation_loss'].append(float(valid_loss))

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        y_train_oh = self.one_hot(y_train)
        X_valid, y_valid = self.valid
        y_valid_oh = self.one_hot(y_valid)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train_oh[self.batch_size * batch:self.batch_size * (batch + 1), :]
                gradient_norm = self.training_step(minibatchX, minibatchY)
            # Just log the last gradient norm
            self.train_logs['train_gradient_norm'].append(gradient_norm)
            self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        loss, acc = self.compute_loss_and_accuracy(X, self.one_hot(y))
        loss = loss.detach()

        return loss, acc

    @staticmethod
    def normalize(train: Tuple[torch.Tensor, torch.Tensor],
                  valid: Tuple[torch.Tensor, torch.Tensor],
                  test: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor]]:
        mean = train[0].mean(dim=0)
        std =  train[0].std(dim=0)
        X_train = (train[0] - mean) / std
        X_valid = (valid[0] - mean) / std
        X_test = (test[0] - mean) / std

        return (X_train, train[1]), (X_valid, valid[1]), (X_test, test[1])


    def test_equivariance(self):
        from functools import partial
        test_im = self.train[0][0]/255.
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(torchvision.transforms.functional.affine, angle=0,
                        translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(torchvision.transforms.functional.affine, angle=90,
                           translate=(0, 0), scale=1, shear=0)

        # TODO CODE HERE
        pass
