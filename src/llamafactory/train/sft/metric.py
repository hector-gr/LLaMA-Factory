# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Dict, Optional, Any, Tuple, List

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
        

@dataclass
class ComputeIoU:
    """
    A callable class that computes mIoU and R1@{0.3,0.5,0.7} over text intervals extracted
    from decoded predictions and labels. It supports batch evaluation.

    On each call:
    - If compute_result=False, just accumulate partial results.
    - If compute_result=True, return the final aggregated results and reset internal state.

    Assumes access to `self.tokenizer` for decoding.
    """

    tokenizer: Any  # The tokenizer must be provided when initializing this class.

    def __post_init__(self):
        self._dump()  # Initialize/reset score dictionary at the start

    def _dump(self) -> Optional[Dict[str, float]]:
        """
        Returns the aggregated results if available, then resets the accumulators.
        If no data is accumulated, returns None.
        """
        result = None
        if hasattr(self, "score_dict"):
            # Compute final metrics
            num_examples = self.score_dict["num_examples"]
            if num_examples > 0:
                mIoU = float(np.mean(self.score_dict["ious"])) if len(self.score_dict["ious"]) > 0 else 0.0
                r1_0_3 = self.score_dict["hits@0.3"] / num_examples
                r1_0_5 = self.score_dict["hits@0.5"] / num_examples
                r1_0_7 = self.score_dict["hits@0.7"] / num_examples
                result = {
                    "mIoU": mIoU,
                    "R1@0.3": r1_0_3,
                    "R1@0.5": r1_0_5,
                    "R1@0.7": r1_0_7
                }

        # Reset the dictionaries for the next evaluation
        self.score_dict = {
            "ious": [],
            "hits@0.3": 0,
            "hits@0.5": 0,
            "hits@0.7": 0,
            "num_examples": 0
        }
        return result

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        """
        Accumulate or finalize IoU and R1 metrics.

        Args:
            eval_preds (EvalPrediction): Contains `predictions` and `label_ids` arrays.
            compute_result (bool): If False, accumulate batch results.
                                   If True, return final results and reset state.

        Returns:
            If compute_result=True: a dictionary with {"mIoU", "R1@0.3", "R1@0.5", "R1@0.7"}.
            If compute_result=False: returns None (partial accumulation).
        """
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # If predictions are logits (3D: batch_size, seq_len, vocab_size), convert to IDs via argmax
        if preds.ndim == 3:
            pred_ids = np.argmax(preds, axis=-1)
        else:
            pred_ids = preds

        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels_filtered = np.where(labels == IGNORE_INDEX, self.tokenizer.pad_token_id, labels)
        label_texts = self.tokenizer.batch_decode(labels_filtered, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Regex to extract intervals: "From frames X to Y"
        pattern = re.compile(r"From frames (\d+)\s*to\s*(\d+)", re.IGNORECASE)

        def extract_windows(text: str) -> List[Tuple[int, int]]:
            intervals = []
            for match in pattern.finditer(text):
                start = int(match.group(1))
                end = int(match.group(2))
                if start <= end:
                    intervals.append((start, end))
                else:
                    intervals.append((end, start))
            return intervals

        def iou(interval_a: Tuple[int, int], interval_b: Tuple[int, int]) -> float:
            a_start, a_end = interval_a
            b_start, b_end = interval_b
            intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
            a_len = a_end - a_start
            b_len = b_end - b_start
            union = a_len + b_len - intersection
            return intersection / union if union > 0 else 0.0

        # Ensure score_dict is initialized if not
        if not hasattr(self, "score_dict"):
            self._dump()

        # Accumulate results for this batch
        for pred_text, label_text in zip(pred_texts, label_texts):
            pred_windows = extract_windows(pred_text)
            label_windows = extract_windows(label_text)

            self.score_dict["num_examples"] += 1
            if len(pred_windows) == 0 or len(label_windows) == 0:
                # No intervals means IoU = 0
                if len(label_windows) == 0:
                    print(f"WARNING: no intervals found in ground truth {label_text=} ")
                self.score_dict["ious"].append(0.0)
                continue

            # Compute max IoU for this example
            example_iou_scores = [
                iou(p_win, l_win) for p_win in pred_windows for l_win in label_windows
            ]
            max_iou = max(example_iou_scores) if example_iou_scores else 0.0
            self.score_dict["ious"].append(max_iou)

            # Update hits for thresholds
            if max_iou >= 0.3:
                self.score_dict["hits@0.3"] += 1
            if max_iou >= 0.5:
                self.score_dict["hits@0.5"] += 1
            if max_iou >= 0.7:
                self.score_dict["hits@0.7"] += 1

        # If not final result, just return None
        if not compute_result:
            return None

        # On final call, return aggregated results and reset
        return self._dump()


@dataclass
class ComputeClassificationAccuracy:
    """
    Computes accuracy for multiple choice classification tasks.
    Supports both letter-based (A/B/C...) and text-based answer matching.
    Supports batch evaluation metrics.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        """Returns the accumulated metrics and resets the state."""
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "accuracy": [],
            "letter_match": [],
            "text_match": []
        }
        return result

    def __post_init__(self):
        self._dump()

    def _extract_answer(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts both the letter choice and answer text from a response.
        Returns tuple of (letter_choice, answer_text).
        """
        # Match pattern like (A) or (B) answer text
        letter_pattern = r'\(([A-Z])\)'
        letter_match = re.search(letter_pattern, text)
        letter_choice = letter_match.group(1) if letter_match else None

        # Extract text after the letter choice
        answer_text = None
        if letter_match:
            answer_text = text[letter_match.end():].strip()
        
        return letter_choice, answer_text

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        """
        Computes accuracy by matching either letter choices or answer text.
        
        Args:
            eval_preds: Contains predictions and label_ids
            compute_result: If True, returns final metrics. If False, accumulates results.
        """
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # Decode the predictions and labels
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract answers from both prediction and label
            pred_letter, pred_text = self._extract_answer(pred)
            label_letter, label_text = self._extract_answer(label)

            # Compute letter match
            letter_correct = int(bool(pred_letter and label_letter and pred_letter.upper() == label_letter.upper()))
            self.score_dict["letter_match"].append(letter_correct)

            # Compute text match (if both texts are available)
            text_correct = 0
            if pred_text and label_text:
                # Normalize both texts (lowercase, remove extra whitespace)
                pred_text = " ".join(pred_text.lower().split())
                label_text = " ".join(label_text.lower().split())
                text_correct = int(pred_text == label_text)
            self.score_dict["text_match"].append(text_correct)

            # Overall accuracy (correct if either letter or text matches)
            self.score_dict["accuracy"].append(int(bool(letter_correct or text_correct)))

        if compute_result:
            return self._dump()
