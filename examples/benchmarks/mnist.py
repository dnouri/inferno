"""This script trains a NeuralNetClassifier on MNIST data, once with
skorch, once with pure PyTorch.

Apart from that change, both approaches are as close to eachother as
possible (e.g. performing the same validation steps).

At the end, we assert that the test accuracies are very close (1%
difference) and that skorch is not too much slower (at worst 30%
slower).

Call like this:

```
python examples/benchmarks/mnist.py
python examples/benchmarks/mnist.py --device cuda --num_samples 5000 --amp_enabled true
```

When called the first time, this will download MNIST data to
examples/datasets/mldata (53M).

This script uses similar parameters as this one:
https://github.com/keras-team/keras/blob/0de2adf04b37aa972c955e69caf6917372b70a5b/examples/mnist_cnn.py


"""

import argparse
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import torch
from torch import nn

from skorch.utils import to_device
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring


BATCH_SIZE = 128
LEARNING_RATE = 0.1
MAX_EPOCHS = 12


def get_data(num_samples):
    mnist = fetch_openml('mnist_784')
    torch.manual_seed(0)
    X = mnist.data.astype('float32').reshape(-1, 1, 28, 28)
    y = mnist.target.astype('int64')
    X, y = shuffle(X, y)
    X, y = X[:num_samples], y[:num_samples]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test


class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
        )
        self.out = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.cnn(X)
        X = X.reshape(-1, 64 * 12 * 12)
        X = self.out(X)
        return X


def performance_skorch(
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size,
        device,
        lr,
        max_epochs,
        amp_enabled=False,
):
    torch.manual_seed(0)
    net = NeuralNetClassifier(
        ClassifierModule,
        batch_size=batch_size,
        optimizer=torch.optim.Adadelta,
        lr=lr,
        device=device,
        max_epochs=max_epochs,
        amp_enabled=amp_enabled,
        callbacks=[
            ('tr_acc', EpochScoring(
                'accuracy',
                lower_is_better=False,
                on_train=True,
                name='train_acc',
            )),
        ],
    )
    net.fit(X_train, y_train)
    y_pred = net.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score


def report(losses, batch_sizes, y, y_proba, epoch, time, training=True):
    template = "{} | epoch {:>2} | "

    loss = np.average(losses, weights=batch_sizes)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y, y_pred)

    template += "acc: {:.4f} | loss: {:.4f} | time: {:.2f}"
    print(template.format(
        'train' if training else 'valid', epoch + 1, acc, loss, time))


def train_torch(
        model,
        X,
        X_test,
        y,
        y_test,
        batch_size,
        device,
        lr,
        max_epochs,
        amp_enabled=False,
):
    model = to_device(model, device)

    idx_train, idx_valid = next(iter(StratifiedKFold(
        5, random_state=0).split(np.arange(len(X)), y)))
    X_train, X_valid, y_train, y_valid = (
        X[idx_train], X[idx_valid], y[idx_train], y[idx_valid])
    dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
    )
    dataset_valid = torch.utils.data.TensorDataset(
        torch.tensor(X_valid),
        torch.tensor(y_valid),
    )

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(max_epochs):
        train_out = train_step(
            model,
            dataset_train,
            batch_size=batch_size,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
        )
        report(y=y_train, epoch=epoch, training=True, **train_out)

        valid_out = valid_step(
            model,
            dataset_valid,
            batch_size=batch_size,
            device=device,
            criterion=criterion,
            amp_enabled=amp_enabled,
        )
        report(y=y_valid, epoch=epoch, training=False, **valid_out)

        print('-' * 50)

    return model


def train_step(model, dataset, device, criterion, batch_size, optimizer, amp_enabled=False):
    model.train()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()
    scaler = None if not amp_enabled else torch.cuda.amp.GradScaler()

    for Xi, yi in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        Xi, yi = to_device(Xi, device), to_device(yi, device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            y_pred = model(Xi)
            y_pred = torch.log(y_pred)
            loss = criterion(y_pred, yi)

        if not amp_enabled:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        y_preds.append(y_pred)
        losses.append(loss.item())
        batch_sizes.append(len(Xi))
    toc = time.time()
    return {
        'losses': losses,
        'batch_sizes': batch_sizes,
        'y_proba': torch.cat(y_preds).cpu().detach().numpy(),
        'time': toc - tic,
    }


def valid_step(model, dataset, device, criterion, batch_size, amp_enabled=False):
    model.eval()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()
    with torch.no_grad():
        for Xi, yi in torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
        ):
            Xi, yi = to_device(Xi, device), to_device(yi, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                y_pred = model(Xi)
                y_pred = torch.log(y_pred)
                loss = criterion(y_pred, yi)

            y_preds.append(y_pred)
            loss = loss.item()
            losses.append(loss)
            batch_sizes.append(len(Xi))
    toc = time.time()
    return {
        'losses': losses,
        'batch_sizes': batch_sizes,
        'y_proba': torch.cat(y_preds).cpu().detach().numpy(),
        'time': toc - tic,
    }


def performance_torch(
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size,
        device,
        lr,
        max_epochs,
        amp_enabled=False,
):
    torch.manual_seed(0)
    model = ClassifierModule()
    model = train_torch(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size=batch_size,
        device=device,
        max_epochs=max_epochs,
        lr=0.1,
        amp_enabled=True,
    )

    X_test = torch.tensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().argmax(1)
    return accuracy_score(y_test, y_pred)


def main(device, num_samples, amp_enabled):
    data = get_data(num_samples)
    # trigger potential cuda call overhead
    torch.zeros(1).to(device)

    if True:
        print("\nTesting skorch performance")
        tic = time.time()
        score_skorch = performance_skorch(
            *data,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            amp_enabled=amp_enabled,
        )
        time_skorch = time.time() - tic

    if True:
        print("\nTesting pure torch performance")
        tic = time.time()
        score_torch = performance_torch(
            *data,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            amp_enabled=amp_enabled,
        )
        time_torch = time.time() - tic

    print("time skorch: {:.4f}, time torch: {:.4f}".format(
        time_skorch, time_torch))
    print("score skorch: {:.4f}, score torch: {:.4f}".format(
        score_skorch, score_torch))

    assert np.isclose(score_skorch, score_torch, rtol=0.01), "Scores are not close enough."
    assert np.isclose(time_skorch, time_torch, rtol=0.3), "Times are not close enough."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="skorch MNIST benchmark")
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (e.g. "cuda", "cpu")')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='total number of samples to use')
    parser.add_argument('--amp_enabled', type=bool, default=False,
                        help='whether to enable automatic mixed precision')
    args = parser.parse_args()
    main(device=args.device, num_samples=args.num_samples, amp_enabled=args.amp_enabled)
