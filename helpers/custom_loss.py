class CrossEntropyLossForOneHot(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = "cross_entropy_loss"
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        _ , labels = y_true.max(dim=1)
        return nn.CrossEntropyLoss()(y_pred, labels)