import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import List

from sklearn.svm import SVC


def JSDiv(p, q):
    m = (p+q)/2
    return 0.5*F.kl_div(torch.log(p), m) + 0.5*F.kl_div(torch.log(q), m)


def ZRFScore(model: Module, retrain: Module, forget: DataLoader):
    model_preds = []
    retrain_preds = []

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)

    with torch.no_grad():
        for inputs, labels in forget:
            inputs = inputs.to(DEVICE)
            model_output = model(inputs)
            retrain_output = retrain(inputs)

            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            retrain_preds.append(F.softmax(retrain_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, dim=0)
    retrain_preds = torch.cat(retrain_preds, dim=0)

    return 1 - JSDiv(model_preds, retrain_preds)

class MIA():
    def __init__(self, models: List, retain: DataLoader, forget: DataLoader, test: DataLoader):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = models
        self.retain_prob = self.collect_prob(retain)
        self.forget_prob = self.collect_prob(forget)
        self.test_prob = self.collect_prob(test)

    def collect_prob(self, loader: DataLoader) -> torch.Tensor:
        prob = []
        for model in self.models:
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.DEVICE)
                    outputs = model(inputs)
                    prob.append(*F.softmax(outputs, dim=1))
            prob = torch.cat(prob, dim=0)
        return prob

    def entropy(self, p: torch.Tensor, dim=-1, keepdim=False):
        return -torch.where(p>0, p * p.log(), p.new([0.])).sum(dim=dim, keepdim=keepdim)

    def get_MIA_data(self):
        self.X_r = torch.cat([self.entropy(self.retain_prob), self.entropy(self.forget_prob)], dim=0).cpu().numpy().reshape(-1, 1)
        self.Y_r = np.concatenate([np.ones(len(self.retain_prob)), np.zeros(len(self.test_prob))])

        self.X_f = self.entropy(self.forget_prob).cpu().numpy().reshape(-1, 1)
        self.Y_f = np.concatenate([np.ones(len(self.forget_prob))])

    def get_MIA_prob(self):
        clf = SVC(C=3, gamma='auto', kernel='rbf')

        clf.fit(self.X_r, self.Y_r)

        result = clf.predict(self.X_f)
        return result.mean()


