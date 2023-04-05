import json
import attr
import os
from typing import List
import tqdm
import yaml
import cv2
from utils.imgproc import PixelCoord, mask2bbox
from dataproc.abstract_db import AbstractMaskDatabase, ImageWithAnnotation, AnnotationEntry


@attr.s
class CapSingleObjMaskDatabaseConfig:
    # ${pdc_data_root}/logs_proto/2018-10....
    pdc_data_root: str = ''

    # A list of file indicates which logs will be used for dataset.
    scene_list_filepath: str = ''

    # The name of object category
    category_name_key: str = ''

    # The name of the yaml file with pose annotation
    # Relative to the "${pdc_data_root}/logs_proto/2018-10..../processed" folder
    # Should be in ${pdc_data_root}/logs_proto/2018-10..../processed/${keypoint_yaml_name}
    pose_yaml_name: str = 'images/pose_data.yaml'

    # Simple flag
    verbose: bool = True


@attr.s
class CapSingleObjMaskDatabaseEntry:
    # The path to rgb
    rgb_image_path = ''

    # The path to depth image
    depth_image_path = ''

    # The path to mask image
    binary_mask_path = ''

    # The bounding box is tight
    bbox_top_left = PixelCoord()
    bbox_bottom_right = PixelCoord()


class CapSingleObjMaskDatabase(AbstractMaskDatabase):

    def __init__(self, config: CapSingleObjMaskDatabaseConfig):
        super(CapSingleObjMaskDatabase, self).__init__()
        self._config = config  # Not actually use it, but might be useful
        self._image_entry_list: List[CapSingleObjMaskDatabaseEntry] = []

        # For each scene
        scene_list = sorted(os.listdir(config.pdc_data_root))
        scene_list = filter(lambda x: os.path.isdir(os.path.join(config.pdc_data_root, x)), scene_list)
        scene_list = filter(lambda x: os.path.exists(os.path.join(config.pdc_data_root, x, 'scene_camera.json')), scene_list)
        for scene_dir in tqdm.tqdm(scene_list):

            rgb_dir = os.path.join(config.pdc_data_root, scene_dir, 'rgb')
            depth_dir = os.path.join(config.pdc_data_root, scene_dir, 'depth')
            mask_visib_dir = os.path.join(config.pdc_data_root, scene_dir, 'mask_visib')
            with open(os.path.join(config.pdc_data_root, scene_dir, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(config.pdc_data_root, scene_dir, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(config.pdc_data_root, scene_dir, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)
            
            image_list = sorted(os.listdir(rgb_dir))
            for image_name in image_list:
                image_id = int(image_name.split('.')[0])
                for obj_id in range(2):
                    entry = CapSingleObjMaskDatabaseEntry()
                    entry.rgb_image_path = os.path.join(rgb_dir, image_name)
                    entry.depth_image_path = os.path.join(depth_dir, image_name.replace('jpg', 'png'))
                    entry.binary_mask_path = os.path.join(mask_visib_dir, image_name.split('.')[0] + f'_{obj_id:06d}.png')
                    entry.bbox_top_left = PixelCoord()
                    entry.bbox_bottom_right = PixelCoord()
                    entry.bbox_top_left.x = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][0]
                    entry.bbox_top_left.y = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][1]
                    entry.bbox_bottom_right.x = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][0] + scene_gt_info[str(image_id)][obj_id]['bbox_obj'][2]
                    entry.bbox_bottom_right.y = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][1] + scene_gt_info[str(image_id)][obj_id]['bbox_obj'][3]
                    self._image_entry_list.append(entry)

        # Simple info
        print('The number of images is %d' % len(self._image_entry_list))

    def __len__(self):
        return len(self._image_entry_list)

    @property
    def path_entry_list(self) -> List[CapSingleObjMaskDatabaseEntry]:
        return self._image_entry_list

    def __getitem__(self, idx: int) -> ImageWithAnnotation:
        image_path_entry = self._image_entry_list[idx]
        # The returned type
        result = ImageWithAnnotation()

        # The raw RGB image
        result.rgb_image = cv2.imread(image_path_entry.rgb_image_path, cv2.IMREAD_ANYCOLOR)

        # The annotation, there is only one object
        annotation = AnnotationEntry()
        annotation.category_name = self._config.category_name_key
        annotation.binary_mask = cv2.imread(image_path_entry.binary_mask_path, cv2.IMREAD_GRAYSCALE)
        annotation.bbox_topleft = image_path_entry.bbox_top_left
        annotation.bbox_bottomright = image_path_entry.bbox_bottom_right

        # Append to result and return
        result.annotation_list = [annotation]
        return result


# The debugging method
def path_entry_sanity_check(entry):
    if len(entry.rgb_image_path) < 1 or (not os.path.exists(entry.rgb_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.depth_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.binary_mask_path)):
        return False

    if (not entry.bbox_top_left.is_valid()) or (not entry.bbox_bottom_right.is_valid()):
        return False

    # OK
    return True


def test_cap_singleobj_database():
    import utils.imgproc as imgproc
    config = CapSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/wei/data/pdc'
    config.scene_list_filepath = '/home/wei/Code/mrcnn_integrate/dataset_config/mugs_all.txt'
    config.category_name_key = 'mug'
    database = CapSingleObjMaskDatabase(config)
    path_entry_list = database.path_entry_list
    for entry in path_entry_list:
        assert path_entry_sanity_check(entry)

    # Write the image and annotation
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Iterate over the dataset
    for i in range(len(database)):
        entry = database[i]
        # Write the rgb image
        rgb_img = entry.rgb_image
        rgb_img_path = os.path.join(tmp_dir, 'img_%d_rgb.png' % i)
        cv2.imwrite(rgb_img_path, rgb_img)

        # The mask image
        assert len(entry.annotation_list) == 1
        mask_img = entry.annotation_list[0].binary_mask
        mask_img_path = os.path.join(tmp_dir, 'img_%d_mask.png' % i)
        cv2.imwrite(mask_img_path, imgproc.get_visible_mask(mask_img))

        # the bounding box
        bbox_img = imgproc.draw_bounding_box(480, 640,
                                  entry.annotation_list[0].bbox_topleft, entry.annotation_list[0].bbox_bottomright)
        bbox_img_path = os.path.join(tmp_dir, 'img_%d_bbox.png' % i)
        cv2.imwrite(bbox_img_path, bbox_img)


if __name__ == '__main__':
    test_cap_singleobj_database()
