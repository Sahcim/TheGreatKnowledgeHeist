from torchmetrics import Accuracy, F1Score

GET_ACCURACY_METRIC = {
    "amazon_polarity": Accuracy(num_classes=2, average="macro"),
    "acronym_identification": Accuracy(num_classes=5, average="macro"),
    "swag": Accuracy(num_classes=4, average="macro", mdmc_average="global"),
}

GET_F1_METRIC = {
    "amazon_polarity": F1Score(num_classes=2, average="macro"),
    "acronym_identification": F1Score(num_classes=5, average="macro"),
    "swag": F1Score(num_classes=4, average="macro", mdmc_average="global"),
}
