import torch
from torch import nn
from copy import deepcopy
from math import sqrt
import numpy as np


device = "cpu"


class LinearModel(nn.Module):
    def __init__(self, dim, prior_variance=1.0, p=0):
        super(LinearModel, self).__init__()

        self.dr = nn.Dropout(p) if p > 0 else lambda x: x
        self.w = nn.Linear(dim, 1, bias=True)
        nn.init.normal_(self.w.weight, mean=0.0, std=sqrt(prior_variance))
        nn.init.normal_(self.w.bias, mean=0.0, std=sqrt(prior_variance))

    def forward(self, x):
        x = self.dr(x)
        return self.w(x)


def sample_then_optimize(p, X_train, y_train, X_test, y_test, prior_variance=1.0, noise_variance=1.0, k=10):
    n = len(X_train)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_train.to(device)
    y_train.to(device)
    d = len(X_train[0])

    sotl_list = []
    mc_sotl_list = []

    for i in range(n-1):
        for j in range(k):
            # sample new weights each time
            model = LinearModel(d, prior_variance, p).double()
            if i > 0:
                model, _ = train_to_convergence(model, X_train[:i], y_train[:i], p)

            model.eval()
            neg_loss = - (model(X_train[i]) - y_train[i]).item()**2 / (2*noise_variance) # - 1/2 * np.log(2*np.pi*noise_variance)
            sotl_list.append(neg_loss)

            if j == 0:
                # draw k Monte Carlo dropout samples to approximate expectation of P(D|theta)
                # draw samples from approximating dropout distribution
                # put model in train to get loss with Monte Carlo dropout
                model.train()
                for e in range(k):
                    neg_mc_loss = -(model(X_train[i]) - y_train[i]).item()**2 / (2*noise_variance) # - 1/2 * np.log(2*np.pi*noise_variance)
                    mc_sotl_list.append(neg_mc_loss)

    # L^(D) from Bayesian perspective
    sotl = (1/k)*sum(sotl_list)
    mc_sotl = (1/k)*sum(mc_sotl_list)

    print("SoTL: {}".format(', '.join(map(str, sotl_list))))
    print("MC SoTL: {}".format(', '.join(map(str, mc_sotl_list))))

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_test.to(device)
    y_test.to(device)

    # final model is model trained on all data
    test_losses = []
    naive_sotls = []
    for i in range(k):
        final_model = LinearModel(d, prior_variance, p).double()
        final_model, n_sotl = train_to_convergence(final_model, X_train, y_train, p)

        final_model.eval()
        y_pred = final_model(X_test)

        naive_sotls.append(n_sotl)
        test_losses.append(nn.MSELoss()(y_pred.flatten(), y_test).item())

    test_loss = (1/k) * sum(test_losses)
    naive_sotl = (1/k) * sum(naive_sotls)

    print("Naive SoTL: {}".format(', '.join(map(str, naive_sotls))))
    print("Test Losses: {}".format(', '.join(map(str, test_losses))))

    # return naive_sotl for final model trained on all data
    return sotl, mc_sotl, naive_sotl, test_loss


def train_to_convergence(model, X, y, step_size=0.001, num_steps=500):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

    best = {"epoch": 0, "loss": np.inf, "naive_sotl": 0, "model": deepcopy(model)}
    s = 0
    naive_sotl = 0

    # end training if we have not improved in 50 steps or reach 500 steps
    while s-50 < best["epoch"] and s < num_steps:
        optimizer.zero_grad()

        y_pred = model(X)

        loss = nn.MSELoss(reduction='mean')(y_pred.flatten(), y)

        naive_sotl -= loss.item()

        if loss.item() < best["loss"]:
            best = {"epoch": s, "loss": loss.item(), "naive_sotl": naive_sotl, "model": deepcopy(model)}

        loss.backward()
        optimizer.step()

        s += 1

    # print("Steps: {}".format(s))
    return best["model"], best["naive_sotl"]
