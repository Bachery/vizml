from tqdm import tqdm
from os.path import join
import pandas as pd
import numpy as np
import pickle

from logger import logger
import ml.util as util

# features_directory	= '../features/features_20180520-005740_processed_99_standard'
features_directory	= '../features/raw_1k'
saves_directory	= './saves_with_ids'
log_dir			= './logs/'
num_datapoints		= None
tasks = [None,
		{'outcome_variable_name': 'all_one_trace_type',
		'prediction_task': 'two',
		'sampling_mode': 'over',
		'pref_id': 1,
		'dataset': 'dataset'},
		{'outcome_variable_name': 'all_one_trace_type',
		'prediction_task': 'three',
		'sampling_mode': 'over',
		'pref_id': 2,
		'dataset': 'dataset'},
		{'outcome_variable_name': 'all_one_trace_type', 'prediction_task': 'six',
		'sampling_mode': 'over', 'pref_id': 3, 'dataset': 'dataset'},
		{'outcome_variable_name': 'has_single_src', 'prediction_task': 'two',
		'sampling_mode': 'over', 'pref_id': 4, 'dataset': 'dataset'},
		{'outcome_variable_name': 'num_x_axes', 'prediction_task': 'numeric',
		'sampling_mode': 10000, 'pref_id': 5, 'dataset': 'dataset'},         #10000
		{'outcome_variable_name': 'num_y_axes', 'prediction_task': 'numeric',
		'sampling_mode': 10000, 'pref_id': 6, 'dataset': 'dataset'},          #10000
		{'outcome_variable_name': 'trace_type', 'prediction_task': 'two',
		'sampling_mode': 'over', 'pref_id': 7, 'dataset': 'field'},
		{'outcome_variable_name': 'trace_type', 'prediction_task': 'three',
		'sampling_mode': 'over', 'pref_id': 8, 'dataset': 'field'},
		{'outcome_variable_name': 'trace_type', 'prediction_task': 'six',
		'sampling_mode': 'over', 'pref_id': 9, 'dataset': 'field'},
		{'outcome_variable_name': 'is_single_src', 'prediction_task': 'two',
		'sampling_mode': 'over', 'pref_id': 10, 'dataset': 'field'},
		{'outcome_variable_name': 'is_x_or_y', 'prediction_task': 'two',
		'sampling_mode': 'over', 'pref_id': 11, 'dataset': 'field'},
]

# feature_set_lookup_file_name = 'feature_names_by_type.pkl'
# feature_names_by_type = pickle.load(
# 		open(join(features_directory, feature_set_lookup_file_name), 'rb'))


def load_testset(task_id):
	feature_names = load_feature_names()
	X_test = np.load(saves_directory + '/task_' + str(task_id) + '_X_test_' + str(num_datapoints) + '.npy', 
					allow_pickle=True)
	y_test = np.load(saves_directory + '/task_' + str(task_id) + '_y_test_' + str(num_datapoints) + '.npy', 
					allow_pickle=True)
				
	X_test_df = pd.DataFrame(X_test)
	y_test_df = pd.DataFrame(y_test)
	# print(X_test_df.shape)
	# print(y_test_df.shape)

	if task_id <= 6:	X_test_df.columns = feature_names[0]
	else:			X_test_df.columns = feature_names[2]
	y_test_df.columns = [tasks[task_id]['outcome_variable_name']]
	if task_id <= 6:	task_df = pd.concat([y_test_df, X_test_df['fid']], axis=1)
	else:			task_df = pd.concat([y_test_df, X_test_df['fid'], X_test_df['field_id']], axis=1)
	return task_df


def statistic():
	tasks_df = []
	for task_id in [2, 4, 8, 10, 11]:
		tasks_df.append(load_testset(task_id))
	dataset_df = pd.merge(tasks_df[0], tasks_df[1], on='fid', how='inner')
	field_df = pd.merge(tasks_df[2], tasks_df[3], on='field_id', how='inner')
	field_df = pd.merge(field_df, tasks_df[4], on='field_id', how='inner')
	print(dataset_df.shape, field_df.shape)
	print(dataset_df['fid'])
	print(field_df['fid'].value_counts())

def save_feature_names():
	dataset_features_df = pd.read_csv(
		join(features_directory, 'features_aggregate_single_pairwise.csv'),
		nrows=10)
	dataset_features_names = [name for name in dataset_features_df.columns]

	dataset_outcomes_df = pd.read_csv(
		join(features_directory, 'chart_outcomes.csv'),
		nrows=10)
	dataset_outcomes_names = [name for name in dataset_outcomes_df.columns]

	field_features_df = pd.read_csv(
		join(features_directory, 'field_level_features.csv'),
		nrows=10)
	field_features_names = [name for name in field_features_df.columns]

	field_outcomes_df = pd.read_csv(
		join(features_directory, 'field_level_outcomes.csv'),
		nrows=10)
	field_outcome_names = [name for name in field_outcomes_df.columns]

	print(len(dataset_features_names))
	print(len(dataset_outcomes_names))
	print(len(field_features_names))
	print(len(field_outcome_names))

	np.save(features_directory + '/names_dataset_features.npy', np.array(dataset_features_names, dtype=np.str))
	np.save(features_directory + '/names_dataset_outcomes.npy', np.array(dataset_outcomes_names, dtype=np.str))
	np.save(features_directory + '/names_field_features.npy', np.array(field_features_names, dtype=np.str))
	np.save(features_directory + '/names_field_outcomes.npy', np.array(field_outcome_names, dtype=np.str))


def load_feature_names():
	return [np.load(features_directory + '/names_dataset_features.npy', allow_pickle=True),
			np.load(features_directory + '/names_dataset_outcomes.npy', allow_pickle=True),
			np.load(features_directory + '/names_field_features.npy', allow_pickle=True),
			np.load(features_directory + '/names_field_outcomes.npy', allow_pickle=True)]


if __name__ == '__main__':
	# save_feature_names()
	# load_testset(1)
	statistic()