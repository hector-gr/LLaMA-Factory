"""
WebDataset adapter for LLaMA-Factory.
"""

import random
import webdataset as wds
from ..extras import logging

logger = logging.get_logger(__name__)

class WebDatasetAdapter(wds.WebDataset):
    """
    Adapter class that wraps WebDataset to make it compatible with LLaMA-Factory.
    Inherits from wds.WebDataset to ensure compatibility with WebDataset's methods.
    """
    
    def __init__(self, wds_dataset, dataset_attr, data_args, training_args):
        """
        Initialize the WebDatasetAdapter.
        
        Args:
            wds_dataset: The WebDataset to wrap
            dataset_attr: Dataset attributes
            data_args: Data arguments
            training_args: Training arguments
        """
        # Initialize the parent class with the same pipeline
        super().__init__(wds_dataset.pipeline[0].urls)
        
        # Copy the pipeline from the original dataset
        self.pipeline = wds_dataset.pipeline
        
        # Store additional attributes
        self.wds_dataset = wds_dataset
        self.dataset_attr = dataset_attr
        self.data_args = data_args
        self.training_args = training_args
        self.sample_count = 0
        self.error_count = 0
        self.valid_count = 0
        
        # Log initial information about the dataset
        logger.info_rank0(f"Created WebDatasetAdapter with {dataset_attr.formatting} formatting")
        
        # Try to get shard information
        try:
            if hasattr(wds_dataset, "pipeline") and hasattr(wds_dataset.pipeline[0], "urls"):
                urls = wds_dataset.pipeline[0].urls
                logger.info_rank0(f"WebDataset shards: {len(urls)} shards found")
                if len(urls) > 0:
                    logger.info_rank0(f"First few shards: {urls[:5]}")
        except Exception as e:
            logger.warning_rank0(f"Could not retrieve shard information: {str(e)}")
    
    def __iter__(self):
        """
        Iterate through the dataset, validating samples and logging progress.
        """
        # Log start of iteration
        if self.training_args.local_process_index == 0:
            logger.info_rank0("Starting iteration through WebDatasetAdapter")
        
        # Track if we've yielded any valid samples
        yielded_any = False
        
        # Iterate through the dataset
        for sample in self.wds_dataset:
            self.sample_count += 1
            
            # Log progress periodically
            if self.sample_count % 100 == 0 and self.training_args.local_process_index == 0:
                logger.info_rank0(f"Yielded {self.valid_count} samples from WebDatasetAdapter (processed: {self.sample_count}, errors: {self.error_count})")
            
            # Print sample keys for debugging (first few samples)
            if self.sample_count <= 3 and self.training_args.local_process_index == 0:
                print(f"##################################################")
                print(f"WebDatasetAdapter sample {self.sample_count} keys: {list(sample.keys())}")
                if "_prompt" in sample:
                    print(f"_prompt length: {len(sample['_prompt'])}")
                if "_response" in sample:
                    print(f"_response length: {len(sample['_response'])}")
                print(f"##################################################")
            
            # Validate sample
            if not isinstance(sample, dict):
                self.error_count += 1
                if self.training_args.local_process_index == 0:
                    if self.sample_count <= 3 or random.random() < 0.01:  # Log first 3 samples and ~1% of errors
                        logger.warning_rank0(f"Sample {self.sample_count} is not a dictionary: {type(sample)}")
                continue
            
            # Check if sample has required keys for processing
            # Note: We're checking for _prompt and _response here, not input_ids etc.,
            # because tokenization happens later in the pipeline
            if "_prompt" not in sample or "_response" not in sample:
                self.error_count += 1
                if self.training_args.local_process_index == 0:
                    if self.sample_count <= 3 or random.random() < 0.01:  # Log first 3 samples and ~1% of errors
                        logger.warning_rank0(f"Sample {self.sample_count} missing required fields: {list(sample.keys())}")
                continue
            
            # Sample is valid
            self.valid_count += 1
            yielded_any = True
            yield sample
        
        # If we didn't yield any valid samples, raise an error
        if not yielded_any:
            error_msg = "No valid samples found in WebDatasetAdapter. Check your dataset and processing pipeline."
            if self.training_args.local_process_index == 0:
                logger.error_rank0(error_msg)
            raise ValueError(error_msg)
        
        # Log end of iteration
        if self.training_args.local_process_index == 0:
            logger.info_rank0(f"Finished iteration through WebDatasetAdapter: {self.sample_count} total samples, {self.valid_count} valid, {self.error_count} errors")
    
    def compose(self, *args, **kw):
        """
        Override compose to ensure it works with our adapter.
        """
        result = super().compose(*args, **kw)
        # Make sure our custom attributes are preserved
        result.dataset_attr = self.dataset_attr
        result.data_args = self.data_args
        result.training_args = self.training_args
        result.sample_count = self.sample_count
        result.error_count = self.error_count
        result.valid_count = self.valid_count
        return result 