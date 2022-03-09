import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import numpy as np
import torch

sys.path.insert(0, '.')

from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers.modeling_utils import PreTrainedModel as BertImgForPreTraining
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from datasets.build import make_data_loader
from utils.misc import mkdir, get_rank
from utils.metric_logger import TensorboardLogger

logger=logging.getLogger(__name__)
ALL_MODELS=sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())
MODEL_CLASSES = {'bert': (BertConfig, BertImgForPreTraining, BertTokenizer),}

def main():
    