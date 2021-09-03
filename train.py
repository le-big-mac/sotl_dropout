import numpy as np
import torch
from torch import nn
from copy import deepcopy


device = "cpu"


class LinearModel(nn.Module):
    def __init__(self, w: torch.tensor, p=0):
        super(LinearModel, self).__init__()

        self.dr = nn.Dropout(p) if p > 0 else lambda x: x
        self.w = nn.Linear(len(w), 1, bias=False)
        with torch.no_grad():
            self.w.weight.copy_(w)

    def forward(self, x):
        x = self.dr(x)
        return self.w(x)


def sample_then_optimize(p, X_train, y_train, X_test, y_test, prior_variance=1.0, noise_variance=1.0, k=10):
    n = len(X_train)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_train.to(device)
    y_train.to(device)

    n_losses = []

    for i in range(n-1):
        # sample new weights each time (maybe not necessary)
        w = torch.normal(0, prior_variance, (len(X_train[0]),))

        if i == 0:
            model = LinearModel(w, p).double()
        else:
            model, _ = train_to_convergence(w, X_train[:i], y_train[:i], p)

        # put model in train to get loss with Monte Carlo dropout
        model.train()
        k_losses = []

        # draw k Monte Carlo dropout samples to approximate expectation of P(D|theta)
        for j in range(k):
            # draw samples from approximating dropout distribution
            l = - (model(X_train[i]) - y_train[i])**2 / (2*noise_variance) - 1/2 * np.log(2*np.pi*noise_variance)
            k_losses.append(l.item())

        n_losses.append(k_losses)

    # L^(D) from Bayesian perspective
    sotl = 1 / k * sum(map(sum, n_losses))

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_test.to(device)
    y_test.to(device)

    # final model is model trained on all data
    w = torch.normal(0, prior_variance, (len(X_train[0]),))
    final_model, naive_sotl = train_to_convergence(w, X_train, y_train, p)

    final_model.eval()
    y_pred = final_model(X_test)

    # higher gen is better
    neg_test_loss = -nn.MSELoss()(y_pred.flatten(), y_test)

    # return naive_sotl for final model trained on all data
    return sotl, naive_sotl, neg_test_loss


def train_to_convergence(w, X, y, p=0, step_size=0.001, num_steps=500):
    model = LinearModel(w, p).double()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

    naive_sotl = 0
    best = {"epoch": 0, "loss": np.inf, "naive_sotl": 0, "model": deepcopy(model)}
    s = 0
    # end training if we reach max number of steps or have not improved in 50
    while s-50 < best["epoch"] and s < num_steps:
        optimizer.zero_grad()

        y_pred = model(X)

        loss = nn.MSELoss(reduction='mean')(y_pred.flatten(), y)

        naive_sotl -= loss.item()
        # print(loss.item())
        if loss.item() < best["loss"]:
            best = {"epoch": s, "loss": loss.item(), "naive_sotl": naive_sotl, "model": deepcopy(model)}

        loss.backward()
        optimizer.step()

        s += 1

    # print("Steps: {}".format(s))
    return best["model"], best["naive_sotl"]


# class OptType(Enum):
#     OSBAND = 1
#     SAMPLE_SGD = 2
#     NAIVE_SGD = 3
#
#
# def opt_type_train(opt_type, w, w_prior, X, y, prior_variance=1.0, noise_variance=1.0, step_size=0.001):
#     if opt_type[0] == OptType.OSBAND:
#         return get_osband_weights(w_prior, X, y, prior_variance, noise_variance)
#     elif opt_type[0] == OptType.SAMPLE_SGD:
#         return train_to_convergence(w, w_prior, X, y, p=opt_type[1], prior_variance=prior_variance,
#                                     noise_variance=noise_variance, step_size=step_size, regularization=True)[0]
#
#
# def get_naive_sotl(X_train, y_train, X_test, y_test, p=0, prior_variance=1.0):
#     w_init = np.random.normal(0, prior_variance, (len(X_train[0]),))
#
#     w_final, naive_sotl = train_to_convergence(w_init, w_init, X_train, y_train, p=p, prior_variance=prior_variance,
#                                                regularization=False)
#
#     X_test = torch.tensor(X_test)
#     y_test = torch.tensor(y_test)
#
#     w = torch.tensor(w_final, requires_grad=True)
#     model = LinearModel(w).double()
#     model.eval()
#
#     y_pred = model(X_test)
#
#     gen = -nn.MSELoss()(y_pred.flatten(), y_test).item()
#
#     return naive_sotl, gen
#
#
# def sample_then_optimize(opt_type, X_train, y_train, X_test, y_test, prior_variance=1.0,
#                          noise_variance=1.0, k=10):
#     n = len(X_train)
#     k_losses = []
#     k_ws = []
#
#     for j in range(k):
#         # print("Sample: {}".format(j))
#
#         w = np.random.normal(0, prior_variance, (len(X_train[0]),))
#         w_init = w.copy()
#         ws = [w]
#
#         y_train += np.random.normal(0, noise_variance, y_train.shape)
#
#         losses = []
#
#         for i in range(n-1):
#             # print("Datapoint: {}".format(i))
#
#             # compute logP(D_i|w)
#             l = - (w @ X_train[i] - y_train[i])**2 / (2*noise_variance) - 1/2 * np.log(2*np.pi*noise_variance)
#             losses.append(l)
#
#             w = opt_type_train(opt_type, w, w_init, X_train[:i+1], y_train[:i+1], prior_variance=prior_variance,
#                                noise_variance=noise_variance)
#             # print(w)
#             ws.append(w)
#
#             # print()
#
#         k_losses.append(losses)
#         k_ws.append(ws)
#
#         # print()
#
#     k_losses = list(zip(*k_losses))
#     sotl = 1/k * sum(map(sum, k_losses))
#
#     # get generalization performance
#     X_test = torch.tensor(X_test)
#     y_test = torch.tensor(y_test)
#
#     gens = []
#     for i in range(k):
#         w_final = k_ws[i][-1]
#         w = torch.tensor(w_final, requires_grad=True)
#
#         model = LinearModel(w).double()
#         model.eval()
#
#         y_pred = model(X_test)
#
#         # higher gen is better
#         gens.append(-nn.MSELoss()(y_pred.flatten(), y_test).item())
#
#     av_gen = 1/k * sum(gens)
#
#     return sotl, av_gen
#
#
# def train_to_convergence(w, w_prior, X, y, p=0, prior_variance=1.0, noise_variance=1.0, step_size=0.001, num_steps=500,
#                          regularization=True):
#     w = torch.tensor(w, requires_grad=True)
#     w_prior = torch.tensor(w_prior)
#
#     model = LinearModel(w, p).double()
#     model.train()
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
#
#     X = torch.tensor(X)
#     y = torch.tensor(y)
#
#     X = X.to(device)
#     y = y.to(device)
#
#     naive_sotl = 0
#     # (index, best train loss, sotl at best train loss, parameters at best train loss)
#     best = (0, np.inf, 0, next(model.w.cpu().parameters()))
#     s = 0
#     while s-50 < best[0] and s < num_steps:
#         optimizer.zero_grad()
#
#         y_pred = model(X)
#
#         loss = nn.MSELoss(reduction='mean')(y_pred.flatten(), y)
#         if regularization:
#             loss += 1/len(X) * noise_variance/prior_variance * torch.norm(torch.sub(w_prior, next(model.w.parameters())))
#
#         naive_sotl -= loss.item()
#         # print(loss.item())
#         if loss.item() < best[1]:
#             best = (s, loss.item(), naive_sotl, next(model.w.cpu().parameters()))
#
#         loss.backward()
#         optimizer.step()
#
#         s += 1
#
#     # print("Steps: {}".format(s))
#     return best[3].detach().numpy().flatten(), best[2]
#
#
# def get_osband_weights(w_prior, X, y, prior_variance=1.0, noise_variance=1.0):
#     d = len(X[0])
#     w_osband = \
#         inv(matrix.transpose(X) @ X + noise_variance / prior_variance * np.identity(d)) @ \
#         (noise_variance / prior_variance * w_prior + matrix.transpose(X) @ y)
#
#     return w_osband

