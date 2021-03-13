from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from logger import logger
import ml.util as util

# features_directory	= '../features/features_20180520-005740_processed_99_standard'
features_directory	= '../features/raw_1k'
saves_directory		= './saves_with_ids'
log_dir				= './logs/'

feature_set_lookup_file_name = 'feature_names_by_type.pkl'

