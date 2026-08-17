import torch

from ResNet import resnet18_cbam
from config import Config
from iCaRL import iCaRLmodel


def main():
    cfg = Config()
    print(cfg)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    feature_extractor = resnet18_cbam(pretrained=cfg.pretrained)
    model = iCaRLmodel(numclass=cfg.task_sizes[0],
                       feature_extractor=feature_extractor,
                       batch_size=cfg.batch_size,
                       task_size=cfg.task_sizes[0],
                       memory_size=cfg.memory_size,
                       epochs=cfg.epochs,
                       learning_rate=cfg.learning_rate,
                       device=device,
                       config=cfg)

    for t, tsize in enumerate(cfg.task_sizes):
        model.numclass = sum(cfg.task_sizes[:t + 1])
        model.task_size = tsize
        print('==== start task %d (classes %d..%d) ====' %
              (t, model.numclass - tsize, model.numclass))
        model.beforeTrain()
        accuracy = model.train()
        model.afterTrain(accuracy, t)
        print('==== task %d done, softmax acc %.3f ====' % (t, accuracy))


if __name__ == '__main__':
    main()
