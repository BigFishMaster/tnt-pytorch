from copy import deepcopy


class AccumWeights:
    def __init__(self, model):
        self.avg_weights = deepcopy(list(p.data for p in model.parameters()))

    def update_avg(self, model):
        for p, avg_p in zip(model.parameters(), self.avg_weights):
            avg_p.mul_(0.999).add_(0.001, p.data)

    def update_cur(self, model):
        for p, new_p in zip(model.parameters(), self.avg_weights):
            p.data.copy_(new_p)
