import numpy as np
import attr
from typing import List
import random
from utils.imgproc import PixelCoord


@attr.s
class AnnotationEntry:
    category_name: str = ''
    binary_mask: np.ndarray = np.ndarray(shape=[])
    bbox_topleft: PixelCoord = PixelCoord()
    bbox_bottomright: PixelCoord = PixelCoord()


@attr.s
class ImageWithAnnotation:
    rgb_image: np.ndarray = np.ndarray(shape=[])
    # There might be multiple annotation in a image
    annotation_list: List[AnnotationEntry] = []


class AbstractMaskDatabase(object):
    """
    This class serves as an thin interface between real dataset
    such as spartan multi-view or synthetic and the processor that
    transform these dataset into COCO format.
    To avoid save temp binary mask and bounding box, the data exchange
    is raw image and mask (instead of path to image and mask)
    """
    def __int__(self):
        pass

    def __getitem__(self, idx: int) -> ImageWithAnnotation:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def get_random_entry(self) -> ImageWithAnnotation:
        idx = random.randint(0, self.__len__() - 1)
        return self.__getitem__(idx)
