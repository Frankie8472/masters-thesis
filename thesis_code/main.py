import gc
import torch


def main():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
