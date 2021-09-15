from torch import nn


def train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    model.train()
    for i, batch in enumerate(train_dataloader):

        source, target = batch.src.to(device), batch.trg.to(device)

        # Compute prediction error
        pred = model(source)
        loss = loss_fn(pred, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(source)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
