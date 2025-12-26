import torch 
import sys
import os
import time
import warnings
import torch.nn.functional as F
import yaml 
import models
from torch.utils.tensorboard import SummaryWriter

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spikingjelly')))

# import spikingjelly
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import parametric_lif_net, train_classify, tv_ref_classify



def main():
    ckpt = torch.load(
        "./save/cifar10_snn/model_best.pth.tar",
        map_location="cpu",
        weights_only=False   # 你这个文件需要
    )

    state_dict = ckpt["state_dict"]

    log_dir = "./runs/snn/w_hist"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name.endswith(".w"):
                writer.add_histogram(
                    tag=f"params/{name}",
                    values=tensor.cpu(),
                    global_step=0
                )

    writer.close()

    


if __name__ == '__main__':
    main()
