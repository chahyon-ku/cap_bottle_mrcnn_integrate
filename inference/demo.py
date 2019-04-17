from maskrcnn_benchmark.config import cfg
from inference.predictor import COCODemo
import cv2


def main():
    config_file = "/home/wei/Coding/mrcnn/maskrcnn-" \
                  "benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread('/home/wei/data/mankey_pdc_data/mug/0_rgb.png', cv2.IMREAD_COLOR)
    predictions = coco_demo.run_on_opencv_image(image)
    cv2.imshow('window', predictions)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()