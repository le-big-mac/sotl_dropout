import torch
from torch import nn
from copy import deepcopy
import numpy as np
from math import sqrt


device = "cpu"


class LinearModel(nn.Module):
    def __init__(self, dim, prior_std, p=0):
        super(LinearModel, self).__init__()

        self.dr = nn.Dropout(p) if p > 0 else lambda x: x
        self.w = nn.Linear(dim, 1, bias=True)
        nn.init.normal_(self.w.weight, mean=0.0, std=prior_std)
        nn.init.normal_(self.w.bias, mean=0.0, std=prior_std)

    def forward(self, x):
        x = self.dr(x)
        return self.w(x)


def sample_then_optimize(p, X_train, y_train, X_test, y_test, k=10, weight_decay=False):
    n = len(X_train)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_train.to(device)
    y_train.to(device)
    d = len(X_train[0])

    prior_std = sqrt(1/(d+1))

    # final model is model trained on all data
    test_losses = []
    naive_sotls = []
    for i in range(k):
        final_model = LinearModel(d, prior_std, p).double()
        final_model, n_sotl = train_to_convergence(final_model, X_train, y_train, weight_decay=weight_decay,
                                                   early_stopping=False)

        naive_sotls.append(n_sotl)

        final_model.eval()
        y_pred = final_model(X_test)

        test_losses.append(nn.MSELoss(reduction="mean")(y_pred.flatten(), y_test).item())

    test_loss = (1/k) * sum(test_losses)
    naive_sotl = (1/k) * sum(naive_sotls)

    print("Naive SoTL: {}".format(', '.join(map(str, naive_sotls))))
    print("Test Losses: {}".format(', '.join(map(str, test_losses))))

    # return naive_sotl for final model trained on all data
    return naive_sotl, test_loss


def train_to_convergence(model, X, y, step_size=0.01, num_steps=500, weight_decay=False, early_stopping=False):
    model.train()
    initial_weights = deepcopy(model.w.weight).detach()

    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best = {"epoch": 0, "loss": np.inf, "naive_sotl": 0, "model": deepcopy(model)}
    s = 0
    naive_sotl = 0
    last_loss = np.inf

    def condition():
        if early_stopping:
            return s-50 < best["epoch"] and s < num_steps and last_loss > 1e-3
        else:
            return s < num_steps

    # end training if we have not improved in 50 steps or reach 500 steps
    while condition():
        optimizer.zero_grad()

        y_pred = model(X)

        loss = nn.MSELoss(reduction="mean")(y_pred.flatten(), y) if not weight_decay else \
            nn.MSELoss(reduction="sum")(y_pred.flatten(), y) + (model.w.weight - initial_weights).pow(2).sum() / 10

        last_loss = loss.item()
        naive_sotl -= loss.item()

        if early_stopping and loss.item() < best["loss"]:
            best = {"epoch": s, "loss": loss.item(), "naive_sotl": naive_sotl, "model": deepcopy(model)}

        loss.backward()
        optimizer.step()
        scheduler.step()

        s += 1

    # print("Steps: {}".format(s))
    if early_stopping:
        return best["model"], best["naive_sotl"]
    else:
        return model, naive_sotl
