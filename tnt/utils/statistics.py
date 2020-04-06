import time
import math
import sys

from tnt.utils.logging import logger
from collections import Counter


class Statistics(object):
    def __init__(self, loss=0, n=0, topk=(1,), vals=None):
        self.loss = loss
        self.n = n
        if vals:
            self.n_correct = Counter({k: v for k, v in zip(topk, vals)})
        else:
            self.n_correct = Counter({k: 0 for k in topk})

        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n += stat.n
        self.n_correct.update(stat.n_correct)

    def acc(self, topk=1):
        """ compute accuracy """
        return 100 * (self.n_correct[topk] / self.n)

    def avgloss(self):
        """ compute cross entropy """
        return self.loss / self.n

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def print(self, mode, step, num_steps, lr, start):
        t = self.elapsed_time()
        step_fmt = "%.4d/%.4d" % (step, num_steps)
        acc = " ".join(["top-%s:%6.2f" % (k, self.acc(k)) for k in self.n_correct.keys()])
        logger.info(
            ("(%s) step %s; acc: %s; ppl: %5.2f; avgloss: %4.4f; " +
             "lr: %7.7f; %3.0f src/sec; %6.0f sec")
            % (mode,
               step_fmt,
               acc,
               self.ppl(),
               self.avgloss(),
               lr,
               self.n / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, writer, lr, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/avgloss", self.avgloss(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        for k in self.n_correct:
            name = "top-" + str(k)
            writer.add_scalar(prefix + "/acc/"+name, self.acc(k), step)
        writer.add_scalar(prefix + "/src_per_sec", self.n / (t + 1e-5), step)
        writer.add_scalar(prefix + "/lr", lr, step)
