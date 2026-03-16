# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, UnifiedSIIterableDataset, UnifiedSIIterableDatasetv2
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'si': UnifiedSIIterableDataset,
    'si_v2': UnifiedSIIterableDatasetv2,
    'rule_base': UnifiedSIIterableDataset,
    'rule_base_v2': UnifiedSIIterableDatasetv2
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/mnt/shared-storage-user/zhuchenming/data/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/mnt/shared-storage-user/zhuchenming/data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '/mnt/shared-storage-user/zhuchenming/data/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/mnt/shared-storage-user/zhuchenming/data/bagel_example/vlm/images',
			'jsonl_path': '/mnt/shared-storage-user/zhuchenming/data/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'si':{
        'scannetpp': {
			'data_dir': '/mnt/inspurfs/efm_t/zhuchenming/packed_scene_data',
			'jsonl_path': '/mnt/inspurfs/efm_t/zhuchenming/packed_scene_data/scannetpp_dataset.jsonl',
			'num_total_samples': 1333
		},
    },
    'si_v2':{
        'scannetpp': {
			'data_dir': '/mnt/inspurfs/efm_t/zhuchenming/packed_scene_data',
			'jsonl_path': '/mnt/inspurfs/efm_t/zhuchenming/packed_scene_data/scannetpp_dataset.jsonl',
			'num_total_samples': 1333
		},
    },
    'rule_base':{
        'dl3dv': {
			'data_dir': '/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv',
			'jsonl_path': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/annotation_for_dl3dv.jsonl',
			'num_total_samples': 19640
		},
        'matterport3d':{
            'data_dir': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data',
			'jsonl_path': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/annotation_matterport3d_1pos_0neg_wi_z_0129_train.json',
			'num_total_samples': 1780
        },
        'scannet':{
            'data_dir': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data',
			'jsonl_path': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/annotation_scannet_1pos_0neg_wi_z_0129_train.json',
			'num_total_samples': 8336
        },
    },
    'rule_base_v2':{
        'dl3dv': {
			'data_dir': '/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv',
			'jsonl_path': '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/annotation_for_dl3dv.jsonl',
			'num_total_samples': 19640
		},
    }
}