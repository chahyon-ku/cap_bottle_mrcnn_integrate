import os
from dataproc.spartan_singleobj_database import SpartanSingleObjMaskDatabaseConfig, SpartanSingleObjMaskDatabase
from dataproc.cap_singleobj_database import CapSingleObjMaskDatabaseConfig, CapSingleObjMaskDatabase
from dataproc.abstract_db import AbstractMaskDatabase
from dataproc.patch_paste_db import PatchPasteDatabase, PatchPasteDatabaseConfig
from dataproc.coco_formatter import COCODatasetFormatter, COCODatasetFormatterConfig
from typing import List, Tuple


def build_singleobj_database() -> (SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig):
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/media/rpm/Data/pdc'
    config.scene_list_filepath = '/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mankey_ros/mankey/config/mugs_up_with_flat_logs.txt'
    config.category_name_key = 'mug'
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def make_singleobj_database(scene_list_path: str, category_name: str) -> Tuple[SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig]:
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/media/rpm/Data/pdc'
    config.scene_list_filepath = scene_list_path
    config.category_name_key = category_name
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def make_capsingleobj_database(scene_list_path: str, category_name: str) -> Tuple[CapSingleObjMaskDatabase, CapSingleObjMaskDatabaseConfig]:
    config = CapSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/rpm/Lab/cap_bottle/data_generation/datasets/train_5k'
    config.scene_list_filepath = scene_list_path
    config.category_name_key = category_name
    database = CapSingleObjMaskDatabase(config)
    return database, config


def build_patch_database(
        database_in: List[AbstractMaskDatabase],
        nominal_size: int) -> (PatchPasteDatabase, PatchPasteDatabaseConfig):
    patch_db_config = PatchPasteDatabaseConfig()
    patch_db_config.db_input_list = database_in
    patch_db_config.nominal_size = nominal_size
    patch_db_config.max_instance = 2
    patch_db = PatchPasteDatabase(patch_db_config)
    return patch_db, patch_db_config


def build_peghole_coco_dataset():
    # Build the database
    raw_db_hole, _ = make_singleobj_database('/home/wei/Coding/mrcnn_integrate/dataset_config/printed_hole.txt', 'hole')
    raw_db_peg, _ = make_singleobj_database('/home/wei/Coding/mrcnn_integrate/dataset_config/printed_peg.txt', 'peg')
    raw_db_siemens_peg, _ = make_singleobj_database('/home/wei/Coding/mrcnn_integrate/dataset_config/siemens_peg.txt', 'peg')
    patch_db, _ = build_patch_database([raw_db_peg, raw_db_hole, raw_db_siemens_peg], 50000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = 'lego_db'
    formatter_config.base_folder = '/home/wei/data/coco/lego_db'
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])


def build_coco_dataset():
    # Build the database
    # raw_db, _ = make_singleobj_database('/home/rpm/Lab/cap_bottle/kpam/cap_bottle_mankey_ros/mankey/config/mugs_up_with_flat_logs.txt', 'shoe')
    raw_db, _ = make_capsingleobj_database('', 'shoe')
    patch_db, _ = build_patch_database([raw_db], 5000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = 'cap_5k'
    formatter_config.base_folder = '/media/rpm/Data/pdc/coco/cap_5k'
    os.makedirs(formatter_config.base_folder, exist_ok=True)
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])


if __name__ == '__main__':
    build_coco_dataset()
