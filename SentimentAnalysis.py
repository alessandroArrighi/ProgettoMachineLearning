from NetRunner import NetRunner
from CustomDatasetSentiment import CustomDatasetSentiment
from ConfigChecker import ConfigChecker
from DatasetHendler import DatasetHendler

if __name__ == "__main__":
    conf = ConfigChecker('./conf/conf.json', './conf/conf_schema.json')

    dataset = DatasetHendler('./data/reviews.txt', './data/labels.txt', './dataset')
    classes = CustomDatasetSentiment(conf.data.te_labels_path, conf.data.te_data_path)

    runner = NetRunner(conf, classes.classes, len(dataset.vocab) + 1)
    runner.train()
    runner.test()