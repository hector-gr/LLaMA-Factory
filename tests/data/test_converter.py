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

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


def test_webdataset_sharegptv_converter():
    dataset_attr = DatasetAttr("webdataset", "test-preference-dataset")
    dataset_attr.formatting = "webdataset_sharegptv"
    dataset_attr.ranking = True
    data_args = DataArguments()
    
    example = {
        "conversations": {
            "messages": [
                {"role": "user", "content": "What color is the bucket in the image?"}
            ],
            "images": ["image_0000.jpg", "image_0001.jpg"]
        },
        "chosen": {
            "messages": [
                {"role": "assistant", "content": "The bucket in the image is white."}
            ],
            "images": ["image_0002_chosen.jpg", "image_0003_chosen.jpg"]
        },
        "rejected": {
            "messages": [
                {"role": "assistant", "content": "I cannot clearly see the buckets in the image."}
            ],
            "images": ["image_0002_rejected.jpg"]
        }
    }
    
    dataset_converter = get_dataset_converter("webdataset_sharegptv", dataset_attr, data_args)
    result = dataset_converter(example)
    
    # Check the structure
    assert len(result["_prompt"]) == 1
    assert result["_prompt"][0]["role"] == Role.USER.value
    assert result["_prompt"][0]["content"] == "What color is the bucket in the image?"
    
    # Check responses (chosen + rejected)
    assert len(result["_response"]) == 2
    assert result["_response"][0]["role"] == Role.ASSISTANT.value
    assert result["_response"][0]["content"] == "The bucket in the image is white."
    assert result["_response"][1]["role"] == Role.ASSISTANT.value
    assert result["_response"][1]["content"] == "I cannot clearly see the buckets in the image."
    
    # Check images are combined
    assert len(result["_images"]) == 5  # 2 from conversations + 2 from chosen + 1 from rejected
    assert result["_images"] == ["image_0000.jpg", "image_0001.jpg", "image_0002_chosen.jpg", "image_0003_chosen.jpg", "image_0002_rejected.jpg"]
    
    assert result["_system"] == ""
    assert result["_tools"] == ""
    assert result["_videos"] is None
    assert result["_audios"] is None
