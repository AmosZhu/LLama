"""
Author: Dizhong Zhu
Date: 20/02/2024
"""

from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from llama import Llama, Dialog
import torch


class SimpleConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        return self.relu(a)


if __name__ == '__main__':
    # example_args = (torch.randn(1, 3, 256, 256),)
    # pre_autograd_aten_dialect = capture_pre_autograd_graph(SimpleConv(), example_args)
    # print("Pre-Autograd ATen Dialect Graph")
    # print(pre_autograd_aten_dialect)

    ckpt_dir = 'llama-2-7b-chat'
    tokenizer_path = 'tokenizer.model'
    max_seq_len = 4096
    max_batch_size = 6

    generator = Llama.build_simple(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    llama_model = generator.model
    example_args = (torch.ones(size=(max_batch_size, max_seq_len)), 0)
    pre_autograd_aten_dialect = capture_pre_autograd_graph(llama_model, example_args)
