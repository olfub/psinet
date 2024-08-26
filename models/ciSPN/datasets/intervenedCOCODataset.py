import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection

coco_categories = [
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "food",
    "furniture",
    "electronic",
    "appliance",
    "indoor",
]

# coco labels: [id][paper, coco2014, coco2017, super category]
# info: coco2014 and coco2017 labels are the same
coco_labels = [
    ["-", "-", "-", "person"],
    ["person", "person", "person", "person"],
    ["bicycle", "bicycle", "bicycle", "vehicle"],
    ["car", "car", "car", "vehicle"],
    ["motorcycle", "motorcycle", "motorcycle", "vehicle"],
    ["airplane", "airplane", "airplane", "vehicle"],
    ["bus", "bus", "bus", "vehicle"],
    ["train", "train", "train", "vehicle"],
    ["truck", "truck", "truck", "vehicle"],
    ["boat", "boat", "boat", "vehicle"],
    ["traffic light", "traffic light", "traffic light", "outdoor"],
    ["fire hydrant", "fire hydrant", "fire hydrant", "outdoor"],
    ["street sign", "-", "-", "outdoor"],
    ["stop sign", "stop sign", "stop sign", "outdoor"],
    ["parking meter", "parking meter", "parking meter", "outdoor"],
    ["bench", "bench", "bench", "outdoor"],
    ["bird", "bird", "bird", "animal"],
    ["cat", "cat", "cat", "animal"],
    ["dog", "dog", "dog", "animal"],
    ["horse", "horse", "horse", "animal"],
    ["sheep", "sheep", "sheep", "animal"],
    ["cow", "cow", "cow", "animal"],
    ["elephant", "elephant", "elephant", "animal"],
    ["bear", "bear", "bear", "animal"],
    ["zebra", "zebra", "zebra", "animal"],
    ["giraffe", "giraffe", "giraffe", "animal"],
    ["hat", "-", "-", "accessory"],
    ["backpack", "backpack", "backpack", "accessory"],
    ["umbrella", "umbrella", "umbrella", "accessory"],
    ["shoe", "-", "-", "accessory"],
    ["eye glasses", "-", "-", "accessory"],
    ["handbag", "handbag", "handbag", "accessory"],
    ["tie", "tie", "tie", "accessory"],
    ["suitcase", "suitcase", "suitcase", "accessory"],
    ["frisbee", "frisbee", "frisbee", "sports"],
    ["skis", "skis", "skis", "sports"],
    ["snowboard", "snowboard", "snowboard", "sports"],
    ["sports ball", "sports ball", "sports ball", "sports"],
    ["kite", "kite", "kite", "sports"],
    ["baseball bat", "baseball bat", "baseball bat", "sports"],
    ["baseball glove", "baseball glove", "baseball glove", "sports"],
    ["skateboard", "skateboard", "skateboard", "sports"],
    ["surfboard", "surfboard", "surfboard", "sports"],
    ["tennis racket", "tennis racket", "tennis racket", "sports"],
    ["bottle", "bottle", "bottle", "kitchen"],
    ["plate", "-", "-", "kitchen"],
    ["wine glass", "wine glass", "wine glass", "kitchen"],
    ["cup", "cup", "cup", "kitchen"],
    ["fork", "fork", "fork", "kitchen"],
    ["knife", "knife", "knife", "kitchen"],
    ["spoon", "spoon", "spoon", "kitchen"],
    ["bowl", "bowl", "bowl", "kitchen"],
    ["banana", "banana", "banana", "food"],
    ["apple", "apple", "apple", "food"],
    ["sandwich", "sandwich", "sandwich", "food"],
    ["orange", "orange", "orange", "food"],
    ["broccoli", "broccoli", "broccoli", "food"],
    ["carrot", "carrot", "carrot", "food"],
    ["hot dog", "hot dog", "hot dog", "food"],
    ["pizza", "pizza", "pizza", "food"],
    ["donut", "donut", "donut", "food"],
    ["cake", "cake", "cake", "food"],
    ["chair", "chair", "chair", "furniture"],
    ["couch", "couch", "couch", "furniture"],
    ["potted plant", "potted plant", "potted plant", "furniture"],
    ["bed", "bed", "bed", "furniture"],
    ["mirror", "-", "-", "furniture"],
    ["dining table", "dining table", "dining table", "furniture"],
    ["window", "-", "-", "furniture"],
    ["desk", "-", "-", "furniture"],
    ["toilet", "toilet", "toilet", "furniture"],
    ["door", "-", "-", "furniture"],
    ["tv", "tv", "tv", "electronic"],
    ["laptop", "laptop", "laptop", "electronic"],
    ["mouse", "mouse", "mouse", "electronic"],
    ["remote", "remote", "remote", "electronic"],
    ["keyboard", "keyboard", "keyboard", "electronic"],
    ["cell phone", "cell phone", "cell phone", "electronic"],
    ["microwave", "microwave", "microwave", "appliance"],
    ["oven", "oven", "oven", "appliance"],
    ["toaster", "toaster", "toaster", "appliance"],
    ["sink", "sink", "sink", "appliance"],
    ["refrigerator", "refrigerator", "refrigerator", "appliance"],
    ["blender", "-", "-", "appliance"],
    ["book", "book", "book", "indoor"],
    ["clock", "clock", "clock", "indoor"],
    ["vase", "vase", "vase", "indoor"],
    ["scissors", "scissors", "scissors", "indoor"],
    ["teddy bear", "teddy bear", "teddy bear", "indoor"],
    ["hair drier", "hair drier", "hair drier", "indoor"],
    ["toothbrush", "toothbrush", "toothbrush", "indoor"],
    ["hair brush", "-", "-", "indoor"],
]


class IntervenedCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", image_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform

        # self.bad_ids = 0

        if split == "train":
            self.coco = CocoDetection(
                root_dir + "train2017/",
                root_dir + "annotations/instances_train2017.json",
            )
        elif split == "val":
            self.coco = CocoDetection(
                root_dir + "val2017/", root_dir + "annotations/instances_val2017.json"
            )
        else:
            raise RuntimeError(f"Unknown dataset split {split}")

        # fix intervention category for each smaple
        self._rng = np.random.default_rng(12345)
        self.interventions = self._rng.integers(0, 1000, size=len(self.coco))

        self.intervention_count = len(coco_categories)

        self.num_observed_variables = 0

        self.num_hidden_variables = len(coco_labels)
        self.intervention_shape = 1  # len(coco_categories)

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, item):
        image, target = self.coco[item]

        # select one label and intervene on all labels with its category
        if len(target) == 0:
            # self.bad_ids += 1
            intervention_cat = len(coco_categories)
            intervened_classes = np.zeros((len(coco_labels)))
        else:
            intervention_idx = self.interventions[item] % len(target)
            intervention_cat = coco_categories.index(
                coco_labels[target[intervention_idx]["category_id"]][-1]
            )

            # collect intervened classes
            intervened_classes = np.zeros((len(coco_labels)))
            intervened_annotations = []
            for i, annotation in enumerate(target):
                ann_class = annotation["category_id"]
                ann_cat = coco_categories.index(coco_labels[ann_class][-1])

                if ann_cat == intervention_cat:
                    intervened_annotations.append(i)
                    intervened_classes[ann_class] = 1

            # black out annotations
            image = np.array(image, dtype=np.uint8)
            for i in intervened_annotations:
                annotation = target[i]
                x, y, w, h = annotation["bbox"]
                image[round(y) : round(y + h), round(x) : round(x + w), :] = 0

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "intervention": intervention_cat,
            "target": intervened_classes,
        }
