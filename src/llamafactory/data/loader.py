# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union
import json
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
import webdataset as wds
import wids
from PIL import Image
import io
import random
import torch

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset, get_dataset_converter
from .data_utils import merge_dataset, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)
from .webdataset_adapter import WebDatasetAdapter


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def load_webdataset(
        path: str,
        dataset_attr: "DatasetAttr",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        cache_dir: Optional[str] = None
    ) -> wds.WebDataset:
    logger.info_rank0(f"Loading WebDataset from pattern: {path}")
    
    # Create WebDataset directly
    # wds_dataset = wds.WebDataset(webdataset_pattern, shardshuffle=True, resampled=True)
    # we get /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/VisualSketchpad/llama_factory/mc_question_perception_test/v0.3correct_top1/v0.3correct_top1-{000000..000003}.tar
    # get all the shards by expansing the { } part

    # wds_dataset = wids.ShardListDataset(
    #     path, 
    #     cache_dir=cache_dir, 
    #     # cache_size=10, # the number of shards to keep in the cache
    #     keep=True
    # )
    # from https://github.com/webdataset/webdataset/issues/250
    wds_dataset = wds.WebDataset(
        path, 
        resampled=True,
        nodesplitter=wds.split_by_node,
        shardshuffle=True,
        # workersplitter=wds.split_by_worker,
    ).shuffle(
        # 1000
        1000
    # ).decode(
    #     # 'pil'
    # # ).to_tuple(
    # #     "groundlevel.jpg", "overhead.jpg", "metadata.json","__key__"
    )
    world_size = 1
    try:
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = torch.distributed.group.WORLD
            world_size = torch.distributed.get_world_size(group=group)
    except ModuleNotFoundError:
        pass
    if not world_size > 1:
        iterator = iter(wds_dataset)
        # iterator = torch.utils.data.DataLoader(wds_dataset, num_workers=2)
        # THis breaks if worldsize > 1 as the loader doesn't have workers!
        try:
            first_sample = next(iterator)
            print(f"{first_sample=}")
        except StopIteration:
            print("The dataset is empty!")
    
    # We now apply the shuffling with the sampler
    # And use DistributedChunkedSampler for distributed training
    # - The converter is called as usual by align_dataset
    
    
    # Sample counter for logging
    sample_count = [0]
    valid_count = [0]
    error_count = [0]
    
    # Process sample function with better error handling
    def process_sample(sample):
        # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(342) {sample=}")
        sample_count[0] += 1
        
        # # Log progress periodically
        # if sample_count[0] % 100 == 0 and training_args.local_process_index == 0:
        #     logger.info_rank0(f"Processed {sample_count[0]} samples, {valid_count[0]} valid, {error_count[0]} errors")
        
        # # Log sample keys for debugging
        # if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):  # Log first 5 samples and ~1% of others
        #     logger.info_rank0(f"WebDataset sample keys: {list(sample.keys())}")
        
        # Use output.json instead of json
        if "output.json" not in sample:
            if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                logger.warning_rank0(f"Sample missing 'output.json' key: {list(sample.keys())}")
                raise ValueError(f"Sample missing 'output.json' key: {list(sample.keys())}")
            error_count[0] += 1
            return None
        
        # try:
        # WebDataset's decode() should have already converted the JSON string to a Python object
        # But handle both cases for robustness
        if isinstance(sample["output.json"], str):
            try:
                json_data = json.loads(sample["output.json"])
            except json.JSONDecodeError as e:
                if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                    logger.warning_rank0(f"JSON decode error: {str(e)}")
                    raise ValueError(f"JSON decode error: {str(e)}")
                error_count[0] += 1
                return None
        elif isinstance(sample["output.json"], bytes):
            json_data = json.loads(sample["output.json"].decode("utf-8"))
        else:
            json_data = sample["output.json"]
        
        # Ensure the decoded data has the expected structure
        if not isinstance(json_data, dict):
            if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                logger.warning_rank0(f"JSON data is not a dictionary: {type(json_data)}")
                raise ValueError(f"JSON data is not a dictionary: {type(json_data)}")
            error_count[0] += 1
            return None
        
        # Process images if they exist
        images_key = dataset_attr.images or "images"
        if images_key in json_data and isinstance(json_data[images_key], list):
            # Check if we have binary image data in the sample
            img_keys = [k for k in sample.keys() if k.startswith("image_") or k.endswith((".jpg", ".png", ".jpeg"))]
            
            # Log image keys for debugging
            # if training_args.local_process_index == 0 and sample_count[0] <= 5:
            #     logger.info_rank0(f"Sample {sample_count[0]} image keys: {img_keys}")
            #     logger.info_rank0(f"Sample {sample_count[0]} JSON images: {json_data[images_key]}")
            
            # Create a new list for processed images
            processed_images = []
            
            # Process all found images
            for img_path in json_data[images_key]:
                # try:
                # First, check if the image path is a key in the sample
                if img_path in sample:
                    # Image data is directly in the sample
                    img = Image.open(io.BytesIO(sample[img_path]))
                    processed_images.append(img)
                else:
                    # Try to find the image by its basename or other patterns
                    img_basename = os.path.basename(img_path)
                    matching_keys = [k for k in img_keys if img_basename in k or k in img_path]
                    
                    if matching_keys:
                        # Use the first matching key
                        img_key = matching_keys[0]
                        if isinstance(sample[img_key], bytes):
                            img = Image.open(io.BytesIO(sample[img_key]))
                        else:
                            img = sample[img_key]  # It's already a PIL image
                        processed_images.append(img)
                    else:
                        # As a last resort, try to load from filesystem
                        # (this should rarely happen with properly formatted WebDatasets)
                        try:
                            img_path_with_dir = os.path.join(data_args.media_dir, img_path)
                            if os.path.exists(img_path_with_dir):
                                img = Image.open(img_path_with_dir)
                                processed_images.append(img)
                            else:
                                logger.warning_rank0(f"Could not find image {img_path} in sample or filesystem")
                                raise ValueError(f"Could not find image {img_path} in sample or filesystem")
                                # Add None as a placeholder
                                processed_images.append(None)
                        except Exception as e:
                            raise ValueError(f"Failed to load image {img_path} from filesystem: {str(e)}")
                            logger.warning_rank0(f"Failed to load image {img_path} from filesystem: {str(e)}")
                            # Add None as a placeholder
                            processed_images.append(None)
            json_data[images_key] = processed_images
        # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(437) {json_data=}")
        return json_data
    
    # Apply the processing function to the WebDataset
    dataset = wds_dataset.map(
        process_sample
    # ).batched(
    #     # self.args.train_batch_size
    #     1 # is this global or per-gpu batch size?
    # ).with_epoch(
    #     # I think here is number of batches, since we batch just before?
    #     4676 # this is the number of samples per epoch
    )

    # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(484) {dataset=}")
    return dataset
    

def load_shardlistdataset(
        path: str,
        dataset_attr: "DatasetAttr",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        cache_dir: Optional[str] = None
    ) -> wids.ShardListDataset:
    logger.info_rank0(f"Loading WebDataset from pattern: {path}")
    
    # Create WebDataset directly
    # wds_dataset = wds.WebDataset(webdataset_pattern, shardshuffle=True, resampled=True)
    # we get /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/VisualSketchpad/llama_factory/mc_question_perception_test/v0.3correct_top1/v0.3correct_top1-{000000..000003}.tar
    # get all the shards by expansing the { } part

    wds_dataset = wids.ShardListDataset(
        path, 
        cache_dir=cache_dir, 
        # cache_size=10, # the number of shards to keep in the cache
        keep=True
    )


    assert len(wds_dataset) > 0, f"WebDataset has no shards: {path}"
    
    
    
    # Sample counter for logging
    sample_count = [0]
    valid_count = [0]
    error_count = [0]
    
    # Process sample function with better error handling
    def process_sample(sample):
        # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(342) {sample=}")
        sample_count[0] += 1
        
        # # Log progress periodically
        # if sample_count[0] % 100 == 0 and training_args.local_process_index == 0:
        #     logger.info_rank0(f"Processed {sample_count[0]} samples, {valid_count[0]} valid, {error_count[0]} errors")
        
        # # Log sample keys for debugging
        # if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):  # Log first 5 samples and ~1% of others
        #     logger.info_rank0(f"WebDataset sample keys: {list(sample.keys())}")
        
        # Use output.json instead of json
        if ".output.json" not in sample:
            if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                logger.warning_rank0(f"Sample missing '.output.json' key: {list(sample.keys())}")
                raise ValueError(f"Sample missing '.output.json' key: {list(sample.keys())}")
            error_count[0] += 1
            return None
        
        # try:
        # WebDataset's decode() should have already converted the JSON string to a Python object
        # But handle both cases for robustness
        if isinstance(sample[".output.json"], str):
            try:
                json_data = json.loads(sample[".output.json"])
            except json.JSONDecodeError as e:
                if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                    logger.warning_rank0(f"JSON decode error: {str(e)}")
                error_count[0] += 1
                return None
        else:
            json_data = sample[".output.json"]
        
        # Ensure the decoded data has the expected structure
        if not isinstance(json_data, dict):
            if training_args.local_process_index == 0 and (sample_count[0] <= 5 or random.random() < 0.01):
                logger.warning_rank0(f"JSON data is not a dictionary: {type(json_data)}")
            error_count[0] += 1
            return None
        
        # Process images if they exist
        images_key = dataset_attr.images or "images"
        if images_key in json_data and isinstance(json_data[images_key], list):
            # Check if we have binary image data in the sample
            img_keys = [k for k in sample.keys() if k.startswith("image_") or k.endswith((".jpg", ".png", ".jpeg"))]
            
            # Log image keys for debugging
            # if training_args.local_process_index == 0 and sample_count[0] <= 5:
            #     logger.info_rank0(f"Sample {sample_count[0]} image keys: {img_keys}")
            #     logger.info_rank0(f"Sample {sample_count[0]} JSON images: {json_data[images_key]}")
            
            # Create a new list for processed images
            processed_images = []
            
            # Process all found images
            for img_path in json_data[images_key]:
                # try:
                # First, check if the image path is a key in the sample
                if img_path in sample:
                    # Image data is directly in the sample
                    img = Image.open(io.BytesIO(sample[img_path]))
                    processed_images.append(img)
                else:
                    # Try to find the image by its basename or other patterns
                    img_basename = os.path.basename(img_path)
                    matching_keys = [k for k in img_keys if img_basename in k or k in img_path]
                    
                    if matching_keys:
                        # Use the first matching key
                        img_key = matching_keys[0]
                        if isinstance(sample[img_key], bytes):
                            img = Image.open(io.BytesIO(sample[img_key]))
                        else:
                            img = sample[img_key]  # It's already a PIL image
                        processed_images.append(img)
                    else:
                        # As a last resort, try to load from filesystem
                        # (this should rarely happen with properly formatted WebDatasets)
                        try:
                            img_path_with_dir = os.path.join(data_args.media_dir, img_path)
                            if os.path.exists(img_path_with_dir):
                                img = Image.open(img_path_with_dir)
                                processed_images.append(img)
                            else:
                                logger.warning_rank0(f"Could not find image {img_path} in sample or filesystem")
                                raise ValueError(f"Could not find image {img_path} in sample or filesystem")
                                # Add None as a placeholder
                                processed_images.append(None)
                        except Exception as e:
                            raise ValueError(f"Failed to load image {img_path} from filesystem: {str(e)}")
                            logger.warning_rank0(f"Failed to load image {img_path} from filesystem: {str(e)}")
                            # Add None as a placeholder
                            processed_images.append(None)
            json_data[images_key] = processed_images
        # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(437) {json_data=}")
        return json_data
    
    dataset = wds_dataset.add_transform(process_sample)

    # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/loader.py:(484) {dataset=}")
    return dataset
    


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    
    if dataset_attr.load_from in "webdataset":
        data_path = dataset_attr.webdataset_pattern
        data_files = [data_path]
    elif dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.formatting == "webdataset_sharegpt":
        # breakpoint() # pass cache dir etc?
        assert len(data_files) == 1, f"WebDataset should have exactly one file pointing to the shards, but got: {data_files}"
        dataset = load_webdataset(
            data_files[0],
            dataset_attr,
            data_args,
            training_args,
            cache_dir=model_args.cache_dir,
        )
    elif dataset_attr.formatting == "shardlistdataset_sharegpt":
        # breakpoint() # pass cache dir etc?
        assert len(data_files) == 1, f"WebDataset should have exactly one file pointing to the shards, but got: {data_files}"
        dataset = load_shardlistdataset(
            data_files[0],
            dataset_attr,
            data_args,
            training_args,
            cache_dir=model_args.cache_dir,
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    merge: bool = True,
) -> Optional[Union["Dataset", "IterableDataset", Dict[str, "Dataset"]]]:
    r"""
    Returns the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
        
        # Check if we have a WebDataset
        if isinstance(datasets[dataset_name], wds.WebDataset):
            logger.info_rank0(f"Dataset {dataset_name} is a WebDataset")
            # Try to get shard information
            try:
                if hasattr(datasets[dataset_name], "pipeline") and hasattr(datasets[dataset_name].pipeline[0], "urls"):
                    urls = datasets[dataset_name].pipeline[0].urls
                    logger.info_rank0(f"WebDataset shards: {len(urls)} shards found")
                    if len(urls) > 0:
                        logger.info_rank0(f"First few shards: {urls[:5]}")
            except Exception as e:
                logger.info_rank0(f"Error getting shard information: {str(e)}")
    
    if merge:
        # Check if any dataset is a WebDataset
        has_webdataset = any(isinstance(ds, wds.WebDataset) for ds in datasets.values())
        if has_webdataset:
            logger.info_rank0("Cannot merge WebDataset with other datasets, returning the first one")
            # Return the first WebDataset
            for ds in datasets.values():
                if isinstance(ds, wds.WebDataset):
                    return ds
        
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)
    else:
        return datasets


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""
    Returns the corresponding dataset processor.
    """
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None
        
    # Check if we have a WebDatasetAdapter
    if isinstance(dataset, WebDatasetAdapter):
        logger.info_rank0("Processing WebDatasetAdapter with native WebDataset pipeline")
        
        # Get the dataset processor
        dataset_processor = _get_dataset_processor(
            data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
        )
        
        # Apply operations in the correct order:
        # 1. Ensure the dataset is decoded - wids skips this
        # if hasattr(dataset, 'is_decoded') and not dataset.is_decoded:
        #     dataset = dataset.decode()
        #     logger.info_rank0("Applied decoding to WebDatasetAdapter")
        
        # 2. Apply shuffling if needed - this will be handled by the sampler
        
        # 3. Batch the samples
        batch_size = data_args.preprocessing_batch_size or 32
        batched_dataset = dataset.batched(batch_size)
        logger.info_rank0(f"Applied batching with batch size: {batch_size}")
        
        # 4. Apply the dataset processor to the batched dataset
        processed_dataset = batched_dataset.map(dataset_processor.preprocess_dataset)
        logger.info_rank0("Applied dataset processor to batched WebDatasetAdapter")
        
        # 5. Set up a custom sampler for distributed training
        # Store the dataset and sampler information for later use in the trainer
        if hasattr(processed_dataset, 'set_sampler_builder'):
            def sampler_builder(dataset):
                return wids.DistributedChunkedSampler(
                    dataset, 
                    chunksize=128,  # Adjust as needed
                    shuffle=True,
                    seed=training_args.seed
                )
            processed_dataset.set_sampler_builder(sampler_builder)
            logger.info_rank0("Set up DistributedChunkedSampler for WebDataset")
        
        # Print an example if should_log is True
        if training_args.should_log:
            try:
                print("eval example:" if is_eval else "training example:")
                # Get the first batch
                first_batch = next(iter(processed_dataset))
                # Print the first example in the batch
                dataset_processor.print_data_example({
                    key: value[0] if isinstance(value, list) and len(value) > 0 else value
                    for key, value in first_batch.items()
                })
            except StopIteration:
                if stage == "pt":
                    raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
                else:
                    raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")
        
        return processed_dataset
    
    # For regular datasets, use the original implementation
    
    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    

    
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    if isinstance(dataset, wids.ShardListDataset):
        assert len(kwargs) == 0, f"kwargs is not empty for ShardListDataset: {kwargs=}"
        def transform_removing_columns(example, func, remove_columns=None):
            """
            Processes a single example by first removing specified columns before passing it to
            a mapping function. The function's returned dict is then used to update the example.
            
            This mimics the behavior of Hugging Face Datasets' map function when using the
            remove_columns argument.
            
            Args:
            example: A dictionary representing one data example.
            func: A function that takes a dictionary and returns a dictionary of updates.
            remove_columns: A list (or set) of keys to remove from the example before applying func.
            
            Returns:
            A new dictionary representing the processed example.
            """
            # Create a copy to avoid modifying the original.
            new_example = example.copy()
            
            # Remove specified columns BEFORE applying the mapping function.
            if remove_columns:
                for col in remove_columns:
                    new_example.pop(col, None)  # remove the key if it exists
            
            # Call the mapping function on the pruned example.
            updates = func(new_example)
            
            # If the mapping function returns some updates, update the example.
            if updates is not None:
                new_example.update(updates)
            print(f"After processing: {new_example.keys()=}")
            return new_example
        from functools import partial
        dataset = dataset.add_transform(
            # TODO: figure out why we need to add brackets here. Is it because our process_sample function doesn't?
            # The brackets cause some issues later, where they need to be removed
            # This fails because important columns are actually removed! Not sure why map() works here 
            # partial(transform_removing_columns, 
            #     func= lambda x: dataset_processor.preprocess_dataset(
            #         {k: [v] for k, v in x.items()}
            #     ),
            #     remove_columns=column_names
            # )
            # Remove a selection of columns while doing the mapping. Columns will be removed before updating the examples with the output of function, i.e. if function is adding columns with names in remove_columns, these columns will be kept.
            lambda x: dataset_processor.preprocess_dataset(
                    {k: [v] for k, v in x.items()}
            )
        )
    elif isinstance(dataset, wds.WebDataset):
        dataset = dataset.map(
            dataset_processor.preprocess_dataset,
        )
    else:
        # Get column names from the first example
        column_names = list(next(iter(dataset)).keys())
        dataset = dataset.map(
            dataset_processor.preprocess_dataset,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=column_names,
            **kwargs,
        )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            if not (isinstance(dataset, wds.WebDataset) and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
                dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset



def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data: Union["Dataset", "DatasetDict"] = load_from_disk(data_args.tokenized_path)
            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")

            dataset_module: Dict[str, "Dataset"] = {}
            if isinstance(tokenized_data, DatasetDict):
                if "train" in tokenized_data:
                    dataset_module["train_dataset"] = tokenized_data["train"]

                if "validation" in tokenized_data:
                    dataset_module["eval_dataset"] = tokenized_data["validation"]

            else:  # single dataset
                dataset_module["train_dataset"] = tokenized_data

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        # this only loads paths, not actual video (or text) tokens
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset, model_args, data_args, training_args, stage, merge=training_args.do_predict
        )

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming and not isinstance(dataset, wids.ShardListDataset):
                    if isinstance(dataset, wds.WebDataset):
                        # Use WebDataset's shuffle method
                        dataset = dataset.shuffle(size=data_args.buffer_size, seed=training_args.seed)
                    else:
                        # Use HuggingFace's shuffle method
                        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    dataset_dict.update({f"validation_{name}": data for name, data in eval_dataset.items()})
                else:
                    if data_args.streaming and not isinstance(eval_dataset, wids.ShardListDataset):
                        if isinstance(eval_dataset, wds.WebDataset):
                            # Use WebDataset's shuffle method
                            eval_dataset = eval_dataset.shuffle(size=data_args.buffer_size, seed=training_args.seed)
                        else:
                            # Use HuggingFace's shuffle method
                            eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                    dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:  # save tokenized dataset to disk and exit
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please restart the training with `tokenized_path: {data_args.tokenized_path}`.")

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]
        else:
            eval_dataset = {}
            for key in dataset_dict.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset_dict[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

        return dataset_module
