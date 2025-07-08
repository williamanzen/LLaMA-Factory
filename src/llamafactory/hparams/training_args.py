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

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict

from ..extras.misc import use_ray

# Configure maximum logging for training_args module
logger = logging.getLogger(__name__)

# Set up comprehensive logging configuration
def setup_max_logging():
    """Configure maximum logging level for debugging workload settings."""
    # Create formatter with detailed information
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] [PID:%(process)d] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger to DEBUG level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Add handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(console_handler)
    
    # Configure specific logger for this module
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)
    
    return module_logger

# Initialize maximum logging
logger = setup_max_logging()
logger.info("="*80)
logger.info("LLAMAFACTORY TRAINING_ARGS MODULE LOADED")
logger.info(f"Module path: {__file__}")
logger.info(f"Process ID: {os.getpid()}")
logger.info(f"Python executable: {sys.executable}")
logger.info("="*80)


@dataclass
class RayArguments:
    r"""Arguments pertaining to the Ray training."""

    ray_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The training results will be saved at `<ray_storage_path>/ray_run_name`."},
    )
    ray_storage_path: str = field(
        default="./saves",
        metadata={"help": "The storage path to save training results to"},
    )
    ray_storage_filesystem: Optional[Literal["s3", "gs", "gcs"]] = field(
        default=None,
        metadata={"help": "The storage filesystem to use. If None specified, local filesystem will be used."},
    )
    ray_num_workers: int = field(
        default=1,
        metadata={"help": "The number of workers for Ray training. Default is 1 worker."},
    )
    resources_per_worker: Union[dict, str] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={"help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."},
    )
    placement_strategy: Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"] = field(
        default="PACK",
        metadata={"help": "The placement strategy for Ray training. Default is PACK."},
    )
    ray_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "The arguments to pass to ray.init for Ray training. Default is None."},
    )

    def __post_init__(self):
        logger.info("="*60)
        logger.info("RayArguments.__post_init__ method called")
        logger.info(f"ray_storage_filesystem: {self.ray_storage_filesystem}")
        logger.info(f"ray_storage_path: {self.ray_storage_path}")
        logger.info(f"ray_run_name: {self.ray_run_name}")
        logger.info(f"ray_num_workers: {self.ray_num_workers}")
        logger.info(f"resources_per_worker (before processing): {self.resources_per_worker}")
        logger.info(f"placement_strategy: {self.placement_strategy}")
        logger.info("="*60)
        
        self.use_ray = use_ray()
        logger.info(f"Ray usage detected: {self.use_ray}")
        
        if isinstance(self.resources_per_worker, str) and self.resources_per_worker.startswith("{"):
            logger.info("Processing resources_per_worker string to dict conversion")
            logger.debug(f"Original resources_per_worker string: {self.resources_per_worker}")
            self.resources_per_worker = _convert_str_dict(json.loads(self.resources_per_worker))
            logger.info(f"Converted resources_per_worker to dict: {self.resources_per_worker}")
        
        if self.ray_storage_filesystem is not None:
            logger.info(f"Processing ray_storage_filesystem: {self.ray_storage_filesystem}")
            if self.ray_storage_filesystem not in ["s3", "gs", "gcs"]:
                error_msg = f"ray_storage_filesystem must be one of ['s3', 'gs', 'gcs'], got {self.ray_storage_filesystem}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            import pyarrow.fs as fs
            import os
            logger.info("Successfully imported pyarrow.fs and os modules")
            logger.info("ENTERING POST_INIT METHOD - RAY STORAGE FILESYSTEM CONFIGURATION")
            
            if self.ray_storage_filesystem == "s3":
                logger.info("ENTERING S3 FILESYSTEM BRANCH")
                aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL", None)
                logger.info(f"AWS_ENDPOINT_URL environment variable: {aws_endpoint_url}")
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", None)
                logger.info(f"AWS_ACCESS_KEY_ID environment variable: {aws_access_key}")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
                logger.info(f"AWS_SECRET_ACCESS_KEY environment variable: {aws_secret_key}")
                logger.debug("Creating S3FileSystem with endpoint override")
                self.ray_storage_filesystem = fs.S3FileSystem(
                    access_key=aws_access_key,
                    secret_key=aws_secret_key,
                    endpoint_override=aws_endpoint_url
                )
                logger.info("Successfully configured S3FileSystem")
                logger.debug(f"S3FileSystem object: {self.ray_storage_filesystem}")
            elif self.ray_storage_filesystem == "gs" or self.ray_storage_filesystem == "gcs":
                logger.info(f"ENTERING GCS FILESYSTEM BRANCH (filesystem type: {self.ray_storage_filesystem})")
                logger.debug("Creating GcsFileSystem")
                self.ray_storage_filesystem = fs.GcsFileSystem()
                logger.info("Successfully configured GcsFileSystem")
                logger.debug(f"GcsFileSystem object: {self.ray_storage_filesystem}")
        else:
            logger.info("No ray_storage_filesystem specified, using local filesystem")
        
        logger.info("RayArguments.__post_init__ completed successfully")
        logger.info("="*60)


@dataclass
class TrainingArguments(RayArguments, Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    def __post_init__(self):
        logger.info("="*80)
        logger.info("TrainingArguments.__post_init__ method called")
        logger.info("Calling parent class __post_init__ methods...")
        logger.info("="*80)
        
        logger.info("Calling Seq2SeqTrainingArguments.__post_init__()")
        Seq2SeqTrainingArguments.__post_init__(self)
        logger.info("Seq2SeqTrainingArguments.__post_init__() completed")
        
        logger.info("Calling RayArguments.__post_init__()")
        RayArguments.__post_init__(self)
        logger.info("RayArguments.__post_init__() completed")
        
        logger.info("="*80)
        logger.info("TrainingArguments initialization completed successfully")
        logger.info("Final configuration summary:")
        logger.info(f"  - use_ray: {getattr(self, 'use_ray', 'Not set')}")
        logger.info(f"  - ray_storage_filesystem: {getattr(self, 'ray_storage_filesystem', 'Not set')}")
        logger.info(f"  - ray_storage_path: {getattr(self, 'ray_storage_path', 'Not set')}")
        logger.info(f"  - ray_num_workers: {getattr(self, 'ray_num_workers', 'Not set')}")
        logger.info("="*80)

# Log completion of module loading
logger.info("="*80)
logger.info("LLAMAFACTORY TRAINING_ARGS MODULE FULLY LOADED AND READY")
logger.info("All classes and functions have been defined")
logger.info("Module import completed successfully")
logger.info("="*80)
