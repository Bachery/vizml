# ML utils
import ml.evaluate as evaluate
import ml.util as util
import ml.train as train
from logger import logger
# paper tasks variables and fucntions
from paper_tasks import dataset_indices, field_indices
import paper_tasks
# packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import argparse
import pickle
import os
import sys
from os.path import join
sys.path.insert(0, '..')
# helper utils
from helpers.processing import *
from helpers.analysis import *

'''
Hyper parameters
'''
# features_directory	= '../features/features_20180520-005740_processed_99_standard'
features_directory	= '../features/raw_1k'
models_directory	= './models_full/'
saves_directory		= './saves_with_ids'
log_dir				= './logs/'
feature_set			= 3
RANDOM_STATE 		= 42
num_datapoints		= None
model_prf_for_tasks	= {
	'1':	'paper_dataset_1',
	'2':	'paper_dataset_2',
	'3':	'paper_dataset_3',
	'4':	'paper_dataset_4',
	'5':	'save_task_5_sample10000',
	'6':	'save_task_6_sample10000',
	'7':	'paper_filed_7',
	'8':	'paper_filed_8',
	'9':	'paper_filed_9',
	'10':	'paper_filed_10',
	'11':	'paper_filed_11',
}
model_suf_for_tasks	= {
	'1':	'39',
	'2':	'39',
	'3':	'38',
	'4':	'38',
	'5':	'32',
	'6':	'35',
	'7':	'44',
	'8':	'43',
	'9':	'47',
	'10':	'68',
	'11':	'40',
}
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

def parse_args():
	def s2b(v): return v.lower() in ('true', '1')	# str2bool
	parser = argparse.ArgumentParser('Model')
	
	parser.add_argument('--batch_size',		type=int,	default=200,	help='Batch Size during training [default: 200]')
	parser.add_argument('--num_epochs',		type=int,	default=100,	help='Epoch to run [default: 100]')
	parser.add_argument('--hidden_sizes',	type=int,	default=[1000, 1000, 1000],	nargs='+',
																		help='sizes of network\'s hidden layers [default: 1000*3]')
	parser.add_argument('--learning_rate',	type=float,	default=5e-4,	help='Initial learning rate [default: 5e-4]')
	parser.add_argument('--dropout',		type=float,	default=0.00,	help='')
	parser.add_argument('--patience',		type=int,	default=10,		help='')
	parser.add_argument('--threshold',		type=float,	default=1e-3,	help='')

	parser.add_argument('--only_train',		type=s2b,	default=False,	help='')
	parser.add_argument('--save_model',		type=s2b,	default=True,	help='Whether to save trained models [default: True]')
	parser.add_argument('--print_test',		type=s2b,	default=True,	help='Whether to print test accuracies [default: True]')
	# for constructing learning curves
	parser.add_argument('--dataset_ratios',	type=float,	default=[0.01, 0.1, 0.5, 1.0], nargs='+',
																		help='')
	parser.add_argument('--test_best',		type=s2b,	default=True,	help='')
	# GPU Setting
	parser.add_argument('--use_cuda',		type=s2b,	default=False,	help='')
	parser.add_argument('--gpu',			type=str,	default='0',	help='GPU to use [default: GPU 0]')
	# Note
	parser.add_argument('--note',			type=str,	default=None,	help='')
	# Tasks to use
	parser.add_argument('--tasks',			type=int,	default=[1],	nargs='+',
																		help='')

	return vars(parser.parse_args())


'''
redo features and outcomes loading to save fid
'''
# modified from helpers.analysis.join_features_and_outcomes
def join_data_and_keep_id(features_df, outcomes_df, on='fid'):
    print('Joining feature and outcome DFs')
    final_df = pd.merge(features_df, outcomes_df, on=on, how='inner')
    # final_df = final_df.drop(['fid'], axis=1, inplace=False, errors='ignore')
    # if on != 'fid':
    #     final_df = final_df.drop([on], axis=1, inplace=False, errors='ignore')
    return final_df


# modified from neural_network.paper_tasks.load_features
def load_features_and_save_id(task, logger):
	
	# settings for tasks
	dataset_prediction_task_to_outcomes = {
		'all_one_trace_type': {
			'two': ['line', 'bar'],
			'three': ['line', 'scatter', 'bar'],
			'six': ['line', 'scatter', 'bar', 'box', 'histogram', 'pie'],
		},
		'has_single_src': {
			'two': [True, False]
		},
		'num_x_axes': {
			'numeric': [i for i in range(5)]
		},
		'num_y_axes': {
			'numeric': [i for i in range(5)]
		}
	}
	field_prediction_task_to_outcomes = {
		'trace_type': {
			'two': ['line', 'bar'],
			'three': ['line', 'scatter', 'bar'],
			'six': ['line', 'scatter', 'bar', 'box', 'histogram', 'heatmap'],
		},
		'is_xsrc': {
			'two': [True, False]
		},
		'is_ysrc': {
			'two': [True, False]
		},
		'is_x_or_y': {
			'two': ['x', 'y']
		},
		'is_single_src': {
			'two': [True, False]
		}
	}
	if task['dataset'] == 'dataset':
		task['features_df_file_name'] = 'features_aggregate_single_pairwise.csv'
		task['outcomes_df_file_name'] = 'chart_outcomes.csv'
		task['id_field'] = 'fid'
		prediction_task_to_outcomes = dataset_prediction_task_to_outcomes
	else:
		assert task['dataset'] == 'field'
		task['features_df_file_name'] = 'field_level_features.csv'
		task['outcomes_df_file_name'] = 'field_level_outcomes.csv'
		task['id_field'] = 'field_id'
		prediction_task_to_outcomes = field_prediction_task_to_outcomes
	
	# read original feature and outcome files
	features_df = pd.read_csv(
		join(features_directory, task['features_df_file_name']),
		nrows=num_datapoints)
	outcomes_df = pd.read_csv(
		join(features_directory, task['outcomes_df_file_name']),
		nrows=num_datapoints)
	# feature_names_by_type = pickle.load(
	# 	open(join(features_directory, feature_set_lookup_file_name), 'rb'))
	logger.log('Initial Features: ' + str(features_df.shape))
	logger.log('Initial Outcomes: ' + str(outcomes_df.shape))
	
	# deal with outcomes
	if task['dataset'] == 'field':
		def is_x_or_y(is_xsrc, is_ysrc):
			if is_xsrc and pd.isnull(is_ysrc): return 'x'
			if is_ysrc and pd.isnull(is_xsrc): return 'y'
			else:                              return None
		outcomes_df['is_x_or_y'] = np.vectorize(is_x_or_y)(outcomes_df['is_xsrc'], outcomes_df['is_ysrc'])
		outcomes_df['is_single_src'] = outcomes_df['is_single_xsrc'] | outcomes_df['is_single_ysrc']

	outcomes_df_subset = paper_tasks.format_outcomes_df(logger, outcomes_df, 
							task['outcome_variable_name'],
							prediction_task_to_outcomes[ task['outcome_variable_name'] ] [task['prediction_task'] ],
							id_field=task['id_field'])
 
	# join features and outcomes by the fid/field_id
	final_df = join_data_and_keep_id(features_df, outcomes_df_subset, on=task['id_field'])
	last_index = final_df.columns.get_loc(task['outcome_variable_name'])
	X = final_df.iloc[:, :last_index]
	y = final_df.iloc[:, last_index]
	logger.log('Final DF Shape: ' + str(final_df.shape))
	logger.log('Last Index: ' + str(last_index))
	logger.log('Intermediate Features: ' + str(X.shape))
	logger.log('Index of fid in X: ' + str(X.columns.get_loc('fid')))
	if task['dataset']=='field': 
		logger.log('Index of field in X: ' + str(X.columns.get_loc('field_id')))
	logger.log('Intermediate Outcomes: ' + str(y.shape))
	logger.log('Value counts: \n' + str(y.value_counts()))
	del final_df, outcomes_df	# delete variables to save memory

	# formatting outputs
	y = pd.get_dummies(y).values.argmax(1)

	# sampling data
	if task['sampling_mode'] == 'over':
		res = RandomOverSampler(random_state=RANDOM_STATE)
		X, y = res.fit_sample(X, y)
	elif task['sampling_mode'] == 'under':
		res = RandomUnderSampler(random_state=RANDOM_STATE)
		X, y = res.fit_sample(X, y)
	elif isinstance(task['sampling_mode'], int):
		X_resampled_arrays, y_resampled_arrays = [], []
		for outcome in np.unique(y):
			outcome_mask = (y == outcome)
			X_resampled_outcome, y_resampled_outcome = resample(
				X[outcome_mask],
				y[outcome_mask],
				n_samples=task['sampling_mode'],
				random_state=RANDOM_STATE)
			X_resampled_arrays.append(X_resampled_outcome)
			y_resampled_arrays.append(y_resampled_outcome)
		X = np.concatenate(X_resampled_arrays)	#.astype(np.float64)
		y = np.concatenate(y_resampled_arrays)
	else:
		X, y = X.values.astype(np.float64), y

	logger.log('Final Features:' + str(X.shape))
	logger.log('Final Outcomes:' + str(y.shape))
	unique, counts = np.unique(y, return_counts=True)
	logger.log('Value counts after sampling:')
	logger.log_dict(dict(zip(unique, counts)))
	logger.log('\n')

	X, y = util.unison_shuffle(X, y)
	return X, y


# calling load_features_and_save_id for each task
def load_features_for_all_tasks():
	log_file = log_dir + 'loading_all_tasks_to_save_fid.txt'
	load_logger = logger(log_file, {'MISSION': 'Save fid for all tasks'})
	for task_id in range(1, 12):
		task = tasks[task_id]
		prefix = 'task_' + str(task_id)
		X, y = load_features_and_save_id(task, load_logger)
		util.save_matrices_to_disk(
			X, y, [0.2, 0.2], saves_directory, prefix, num_datapoints)


'''
Evaluation of multiple tasks
'''
def evaluate_and_save_results():
	pass


def eval_multi_tasks(parameters):
	# GPU setting
	if parameters['use_cuda']: 
		os.environ['CUDA_VISIBLE_DEVICES'] = parameters['gpu']
	
	# Log setting
	log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	if parameters['note']: log_suffix = parameters['note'] + '_' + log_suffix
	log_file = log_dir + 'multi_eval_' + log_suffix + '.txt'
	eval_logger = logger(log_file, parameters)
	eval_logger.log('\n')
	parameters['logger'] = eval_logger

	# evaluate each task
	for task_id in parameters['tasks']:
		task = tasks[task_id]
		data_prefix = 'paper_' + task['dataset'] + '_' + str(task['pref_id'])
		parameters['logger'].log_dict(task)
		parameters['logger'].log('dataset prefix: ' + data_prefix)
		# Dataset
		X_train, y_train, X_val, y_val, X_test, y_test = util.load_matrices_from_disk(
			saves_directory, data_prefix, num_datapoints)
		if task['dataset'] == 'dataset':
			X_train	= X_train	[:, dataset_indices[feature_set]]
			X_val	= X_val		[:, dataset_indices[feature_set]]
			X_test	= X_test	[:, dataset_indices[feature_set]]
		else:
			assert task['dataset'] == 'field'
			X_train	= X_train	[:, field_indices[feature_set]]
			X_val	= X_val		[:, field_indices[feature_set]]
			X_test	= X_test	[:, field_indices[feature_set]]
		parameters['logger'].log('loaded test size: ' + str(X_teset.shape))
		_, _, test_dataloader = train.load_datasets(
				X_train, y_train, X_val, y_val, parameters, X_test, y_test, parameters['logger'])
		# model
		parameters['model_prefix'] = model_prf_for_tasks[str(task_id)]
		parameters['model_suffix'] = model_suf_for_tasks[str(task_id)]
		# eval
		evaluate.evaluate(
            parameters['model_suffix'], test_dataloader, parameters, models_directory)


if __name__ == '__main__':
	# parameters = parse_args()
	# print(parameters)
	
	load_features_for_all_tasks()