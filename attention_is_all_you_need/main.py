import torch

from data_loader import Multi30kDataLoader
from model import Transformer
import train
import test
from loss import LabelSmoothingLoss


batch_size = 128
epochs = 1
learning_rate = 5e-4
label_smoothing_value = 0.1
d_model = 512
num_heads = 8
d_k = d_model // num_heads


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    multi_30k_dataloader = Multi30kDataLoader(batch_size, device)
    train_dataloader = multi_30k_dataloader.get_train_loader()
    valid_dataloader = multi_30k_dataloader.get_valid_loader()
    test_dataloader = multi_30k_dataloader.get_test_loader()

    source_vocab_size = multi_30k_dataloader.source_vocab_size
    target_vocab_size = multi_30k_dataloader.target_vocab_size

    model = Transformer().to(device)
    print(model)

    loss_fn = LabelSmoothingLoss(
        label_smoothing_value, len(multi_30k_dataloader.TARGET.vocab))

    # lrate = (d ** 0.5)_model · min(step_num ** −0.5, step_num · warmup_steps ** −1.5)
    # origin warmup_steps = 4000
    # origin theta_1 = 0.9, theta_2 = 0.98, epsilon=10 ** -9
    optimizer = torch.optim.Adam(lr=learning_rate, amsgrad=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, valid_dataloader,
              model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")
    pass


if __name__ == "main":
    main()
