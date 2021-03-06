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
    noise_std = 1

    sotl_list = []
    mc_sotl_list = []

    for i in range(n-1):
        for j in range(k):
            # sample new weights each time
            model = LinearModel(d, prior_std, p).double()
            if i > 0:
                model, _ = train_to_convergence(model, X_train[:i], y_train[:i], weight_decay=weight_decay,
                                                early_stopping=True)

            model.eval()
            neg_loss = - (model(X_train[i]) - y_train[i]).item()**2 / (2*noise_std**2)  # - 1/2 * np.log(2*np.pi*noise_variance)
            sotl_list.append(neg_loss)

            if j == 0:
                # draw k Monte Carlo dropout samples to approximate expectation of P(D|theta)
                # draw samples from approximating dropout distribution
                # put model in train to get loss with Monte Carlo dropout
                model.train()
                for e in range(k):
                    neg_mc_loss = -(model(X_train[i]) - y_train[i]).item()**2 / (2*noise_std**2)  # - 1/2 * np.log(2*np.pi*noise_variance)
                    mc_sotl_list.append(neg_mc_loss)

        print("SoTL D<{}: {}".format(i, ', '.join(map(str, sotl_list[-k:]))))
        print("MC SoTL D<{}: {}".format(i, ', '.join(map(str, mc_sotl_list[-k:]))))

    # L^(D) from Bayesian perspective
    sotl = (1/k)*sum(sotl_list)
    mc_sotl = (1/k)*sum(mc_sotl_list)

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_test.to(device)
    y_test.to(device)

    # final model is model trained on all data
    early_stop_naive_sotls = []
    early_stop_test_losses = []

    naive_sotls = []
    test_losses = []

    for i in range(k):
        for early_stop in (True, False):
            final_model = LinearModel(d, prior_std, p).double()
            final_model, n_sotl = train_to_convergence(final_model, X_train, y_train, weight_decay=weight_decay,
                                                       early_stopping=early_stop)

            final_model.eval()
            y_pred = final_model(X_test)
            loss = nn.MSELoss(reduction="mean")(y_pred.flatten(), y_test).item()

            if early_stop:
                early_stop_naive_sotls.append(n_sotl)
                early_stop_test_losses.append(loss)
            else:
                naive_sotls.append(n_sotl)
                test_losses.append(loss)

    early_stop_test_loss = sum(early_stop_test_losses) / k
    early_stop_naive_sotl = sum(early_stop_naive_sotls) / k

    test_loss = sum(test_losses) / k
    naive_sotl = sum(naive_sotls) / k

    print("Naive SoTL: {}".format(', '.join(map(str, naive_sotls))))
    print("Test Losses: {}".format(', '.join(map(str, test_losses))))

    # return naive_sotl for final model trained on all data
    return sotl, mc_sotl, early_stop_naive_sotl, early_stop_test_loss, naive_sotl, test_loss,


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
