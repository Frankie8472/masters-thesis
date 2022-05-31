import gc
import torch


def main():
    # file is used for local testing, has no purpose apart from that
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
