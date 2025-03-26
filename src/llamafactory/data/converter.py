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
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union

from ..extras import logging
from .data_utils import Role

import wids
import webdataset as wds


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr

logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union[Any, Sequence[Any]]) -> Optional[List[Any]]:
        r"""
        Optionally concatenates media path to media dir when loading from local disk.
        """
        if not isinstance(medias, list):
            medias = [medias] if medias is not None else []
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"] and isinstance(medias[0], str):
            for i in range(len(medias)):
                if os.path.isfile(os.path.join(self.data_args.media_dir, medias[i])):
                    medias[i] = os.path.join(self.data_args.media_dir, medias[i])
                else:
                    logger.warning_rank0_once(f"Media {medias[i]} does not exist in `media_dir`. Use original path.")

        return medias

    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Converts a single example in the dataset to the standard format.
        """
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        
        logger = logging.get_logger(__name__)
        assert isinstance(example, dict), f"example is not a dict: {type(example)=}"
        # logger.info_rank0(f"SharegptDatasetConverter example keys: {list(example.keys())}")
        
        # Check if the messages key is missing
        if self.dataset_attr.messages not in example:
            logger.warning_rank0(f"Missing messages key '{self.dataset_attr.messages}' in example")
            return {
                "_prompt": [],
                "_response": [],
                "_system": "",
                "_tools": "",
                "_images": [],
                "_videos": [],
                "_audios": [],
            }
            # raise ValueError(f"Missing messages key '{self.dataset_attr.messages=}' in example")
        
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True
   
        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        # still a list at this point
        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class WebDatasetSharegptConverter(SharegptDatasetConverter):
    """
    Converter for WebDataset in sharegpt format.
    This is essentially the same as SharegptDatasetConverter but ensures compatibility
    with the WebDataset format.
    """
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # Debug logging
        logger = logging.get_logger(__name__)
        # logger.info_rank0(f"WebDatasetSharegptConverter example keys: {list(example.keys())}")
        
        # Check if this is a dummy example (only has __dummy__ key)
        if set(example.keys()) == {"__dummy__"}:
            logger.warning_rank0("Received example with only __dummy__ key, returning empty example")
            return {
                "_prompt": [],
                "_response": [],
                "_system": "",
                "_tools": "",
                "_images": [],
                "_videos": [],
                "_audios": [],
            }
        
        # logger.info_rank0(f"WebDatasetSharegptConverter dataset_attr.messages: {self.dataset_attr.messages}")
        
        # Check if we need to map keys
        if self.dataset_attr.messages and self.dataset_attr.messages not in example:
            # Try to find the correct key for messages
            if "conversations" in example:
                logger.info_rank0(f"Using 'conversations' instead of '{self.dataset_attr.messages}'")
                example["messages"] = example["conversations"]
            elif "messages" in example:
                logger.info_rank0(f"Using 'messages' as fallback")
                # If dataset_attr.messages is not "messages", create a mapping
                if self.dataset_attr.messages != "messages":
                    example[self.dataset_attr.messages] = example["messages"]
        
        # TODO: For some reason we need to remove batch dim ... hacky
        messages_key = self.dataset_attr.messages or "messages"
        if messages_key in example:
            messages = example[messages_key]
            if isinstance(messages, list) and len(messages) == 1 and \
                len(messages[0]) > 1 and isinstance(messages[0][0], dict):
                example[messages_key] = messages[0]
                messages = example[messages_key]
                example['images'] = example['images'][0]
                if 'videos' in example:
                    example['videos'] = example['videos'][0]
                if 'audios' in example:
                    example['audios'] = example['audios'][0]

        # Call the parent class implementation
        # here returns passes example which is single dict with 'messages'  and 'images' 
        
        return super().__call__(example)


@dataclass
class ShardListDatasetSharegptConverter(SharegptDatasetConverter):
    """
    Converter for ShardListDataset in sharegpt format.
    This is essentially the same as SharegptDatasetConverter but ensures compatibility
    with the ShardListDataset format.
    """
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # Debug logging
        logger = logging.get_logger(__name__)
        # logger.info_rank0(f"ShardListDatasetSharegptConverter example keys: {list(example.keys())}")
        
        # Check if this is a dummy example (only has __dummy__ key)
        if set(example.keys()) == {"__dummy__"}:
            logger.warning_rank0("Received example with only __dummy__ key, returning empty example")
            return {
                "_prompt": [],
                "_response": [],
                "_system": "",
                "_tools": "",
                "_images": [],
                "_videos": [],
                "_audios": [],
            }
        
        # logger.info_rank0(f"ShardListDatasetSharegptConverter dataset_attr.messages: {self.dataset_attr.messages}")
        
        # Check if we need to map keys
        if self.dataset_attr.messages and self.dataset_attr.messages not in example:
            # Try to find the correct key for messages
            if "conversations" in example:
                logger.info_rank0(f"Using 'conversations' instead of '{self.dataset_attr.messages}'")
                example["messages"] = example["conversations"]
            elif "messages" in example:
                logger.info_rank0(f"Using 'messages' as fallback")
                # If dataset_attr.messages is not "messages", create a mapping
                if self.dataset_attr.messages != "messages":
                    example[self.dataset_attr.messages] = example["messages"]
        
        # Check if messages exist and have the right format
        messages_key = self.dataset_attr.messages or "messages"
        if messages_key in example:
            messages = example[messages_key]
            
            # Check if messages need to be reformatted
            if isinstance(messages, list) and len(messages) > 0:
                # Check if messages have the expected structure
                if isinstance(messages[0], dict):
                    # Check if we need to map role/content keys
                    if self.dataset_attr.role_tag not in messages[0] or self.dataset_attr.content_tag not in messages[0]:
                        # Try to find alternative keys
                        if "role" in messages[0] and "content" in messages[0]:
                            logger.info_rank0(f"Mapping 'role'/'content' to '{self.dataset_attr.role_tag}'/{self.dataset_attr.content_tag}")
                            # Create a new list with the correct keys
                            new_messages = []
                            for msg in messages:
                                new_msg = {
                                    self.dataset_attr.role_tag: msg["role"],
                                    self.dataset_attr.content_tag: msg["content"]
                                }
                                new_messages.append(new_msg)
                            example[messages_key] = new_messages
        
        # Call the parent class implementation
        # here returns passes example which is single dict with 'messages'  and 'images' 
        return super().__call__(example)


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "webdataset_sharegpt": WebDatasetSharegptConverter,
    "shardlistdataset_sharegpt": ShardListDatasetSharegptConverter,
}


def register_dataset_converter(name: str, dataset_converter: Type["DatasetConverter"]) -> None:
    r"""
    Register a new dataset converter.
    """
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""
    Gets a dataset converter.
    """
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
        _audios: [],
    """
    # print(f"in /pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/hg52wuli/workspace/LLaMA-Factory/src/llamafactory/data/converter.py:(343) {next(iter(dataset))=}")
    if isinstance(dataset, wids.ShardListDataset):
        # data_iter = iter(wids.DistributedChunkedSampler(dataset))
        # breakpoint()
        next_data = dataset[0]
        # next_data = dataset[next(data_iter)]
    elif isinstance(dataset, wds.WebDataset):
        world_size = 1
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = torch.distributed.group.WORLD
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
        if world_size > 1:
            next_data = {}
        else:
            next_data = next(iter(dataset))
    else:
        next_data = next(iter(dataset))
    column_names = list(next_data.keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    if isinstance(dataset, wids.ShardListDataset):
        # breakpoint()
        assert len(kwargs) == 0, f"kwargs is not empty for ShardListDataset: {kwargs=}"
        dataset = dataset.add_transform(
            dataset_converter,
            # batched=False,
            # TODO: can we just ignore this?
            # remove_columns=column_names, 
        )
    elif isinstance(dataset, wds.WebDataset):
        dataset = dataset.map(
            dataset_converter
        )
    else:
        dataset = dataset.map(
            dataset_converter,
            batched=False,
            remove_columns=column_names,
            **kwargs,
        )
    return dataset
