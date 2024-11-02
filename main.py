from ultralytics import YOLO
import ultralytics.data.build as build
from ultralytics.data.dataset import YOLODataset
import numpy as np
import argparse


class BalancedDataset(YOLODataset):
    """
    Dataset implementation for unbalanced ZVZ dataset
    """

    def __init__(self, *args, **kwargs):
        super(BalancedDataset, self).__init__(*args, **kwargs)

        self.trainMode = "train" in self.prefix
        self.samplesDistribution = self.calculateDistribution()

    def calculateLabelsFreq(self):
        """
        Calculates frequency of each label
        """
        frequency = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for idx in cls:
                frequency[idx] += 1

        return np.array(frequency)

    def calculateSamplesWeights(self, labelsFrequency):
        """
        Calculates weights for each sample according to it's labels.
        """
        labelsWeights = np.sum(labelsFrequency) / labelsFrequency
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1)
                continue

            weight = np.max(labelsWeights[cls])
            weights.append(weight)
        return weights

    def calculateDistribution(self):
        """
        Calculates sampling distribution for each sample.
        """
        samplesWeights = self.calculateSamplesWeights(self.calculateLabelsFreq())
        totalWeight = sum(samplesWeights)
        distribution = [w / totalWeight for w in samplesWeights]
        return distribution

    def __getitem__(self, index):
        if not self.trainMode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.samplesDistribution)
            return self.transforms(self.get_image_and_label(index))


def main(skipTraining, pathToModel):
    """
    Trains a model (optional) and evaluates quality on the infer dataset.

    Args:
        skipTraining: skip the training process and run evaluation only.
        pathToModel: path to pretrained model. Used as pretrained model for
                     training and evaluation model if training is skipped.
    """

    build.YOLODataset = BalancedDataset

    model = YOLO(pathToModel)

    if not skipTraining:
        model.train(
            data=r".\dataset\data.yaml",
            imgsz=640,
            epochs=20,
            cache=True,
            device="cpu",
            workers=8,
            pretrained=True,
            optimizer="AdamW",
            close_mosaic=1,
            freeze=11,
            lr0=0.004,
            lrf=0.01,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0,
            scale=0.3,
            shear=5,
            perspective=0.0001,
            fliplr=0.01,
            copy_paste=0.3,
        )

    model.val(split="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for training and testing a model."
    )
    parser.add_argument(
        "--pathToModel",
        default="yolo11n-seg.pt",
        help="Optional path to existing model. Used as pretrained model for"
        " training and evaluation model if training is skipped.",
    )
    parser.add_argument(
        "--skipTraining",
        action="store_true",
        help="Skips trainig process and evaluates current final model.",
    )
    args = parser.parse_args()
    main(args.skipTraining, args.pathToModel)
