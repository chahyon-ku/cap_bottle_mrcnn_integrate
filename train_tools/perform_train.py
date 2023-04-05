import os
import argparse

# The global parameter
parser = argparse.ArgumentParser()
parser.add_argument('--maskrcnn_benchmark_root', type=str,
                    default='/home/rpm/Lab/cap_bottle/kpam/mrcnn/maskrcnn-benchmark',
                    help='Path to maskrcnn_benchmark, the package should be INSTALLED')
parser.add_argument('--config_path', type=str,
                    default='/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_mug.yaml',
                    help='The absolute path to config file')

# Parse the argument and get result
args = parser.parse_args()
maskrcnn_benchmark_dir = args.maskrcnn_benchmark_root
config_path = args.config_path


def main():
    # The python script as exe
    train_executable = os.path.join(maskrcnn_benchmark_dir, "tools", "train_net.py")

    # Check the existence
    assert os.path.exists(train_executable)

    # The project root for this repo
    proj_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    proj_root = os.path.abspath(proj_root)

    # The config file name
    paths_catalog_path = os.path.join(proj_root, 'train_tools/paths_catalog_pdc.py')
    assert os.path.exists(paths_catalog_path)

    # The output dir
    output_dir = os.path.join(proj_root, 'train_tools/tmp')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert os.path.exists(output_dir)

    # Construct output directory and path catalog
    args = ""
    args += "OUTPUT_DIR \"%s\" " % output_dir
    args += "PATHS_CATALOG \"%s\" " % paths_catalog_path

    # The parameter for training
    args += 'SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0005 ' \
            'SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 ' \
            'SOLVER.CHECKPOINT_PERIOD 2500'
    cmd = "python %s --config-file %s %s" % (train_executable, config_path, args)

    # The command
    print('The command is ')
    print(cmd)

    # Run it
    os.system(cmd)


if __name__ == '__main__':
    main()
