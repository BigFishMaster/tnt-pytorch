import torch


def test_standard():
    torch.random.manual_seed(1)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 4.0])
    layer1 = torch.nn.Linear(3, 2)
    layer2 = torch.nn.Linear(2, 1)
    loss1 = layer2(layer1(a))
    loss2 = layer2(layer1(b))
    loss1.backward(retain_graph=True)
    loss2.backward()
    print("In standard:\nlayer1-weight:\n{}\nlayer2-weight:\n{}\n"
          "layer1-grad:\n{}\nlayer2-grad:\n{}".format(
        layer1.weight.data, layer2.weight.data,
        layer1.weight.grad, layer2.weight.grad))


def test_multiforward():
    torch.random.manual_seed(1)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 4.0])
    layer1 = torch.nn.Linear(3, 2)
    layer2 = torch.nn.Linear(2, 1)
    batch = [a, b]
    loss = 0
    for c in batch:
        loss += layer2(layer1(c))
    loss.backward()
    print("In multiforward:\nlayer1-weight:\n{}\nlayer2-weight:\n{}\n"
          "layer1-grad:\n{}\nlayer2-grad:\n{}".format(
        layer1.weight.data, layer2.weight.data,
        layer1.weight.grad, layer2.weight.grad))


def test2():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    z = y * 4
    output1 = z.mean()
    output2 = z.sum()
    output1.backward(retain_graph=True)
    output2.backward()


def test():
    test_standard()
    print("=="*20)
    test_multiforward()


if __name__ == "__main__":
    test()
