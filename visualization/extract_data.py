# packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import time
import json
import sys
import os

# from instance import Instance
sys.path.insert(0, '..')
from neural_network.logger import logger
# from feature_extraction.general_helpers import clean_chunk
# from feature_extraction.outcomes.chart_outcomes import extract_chart_outcomes
# from feature_extraction.outcomes.field_encoding_outcomes import extract_field_outcomes

class Extracter(object):
	def __init__(self, data_file, parallelize=True, chunk_size=1000, features_files=[]):
		if not os.path.exists('./logs/'): os.mkdir('./logs/')
		log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
		log_file = './logs/extracter_log_' + log_suffix + '.txt'
		self.logger 		= logger(log_file, {'MISSION': 'extracting original data'})
		self.data_file		= data_file
		self.parallelize	= parallelize
		self.chunk_size		= chunk_size

		self.save_folder	= './saves/'
		self.MAX_FIELDS		= 25
		self.first_batch	= True
		if len(features_files) != 0:
			self.dataset_feature_file	= features_files[0]
			self.dataset_outcome_file	= features_files[1]
			self.field_feature_file		= features_files[2]
			self.field_outcome_file		= features_files[3]
			self.dataset_level_index	= None
			self.field_level_index		= None
		if not os.path.exists(self.save_folder):
			os.mkdir(self.save_folder)


### deal with provided features
	def load_dataset_features_and_outcomes(self):
		features_df = pd.read_csv(self.dataset_feature_file)
		outcomes_df = pd.read_csv(self.dataset_outcome_file)
		self.logger.log('Read Dataset-level Features: ' + str(features_df.shape))
		self.logger.log('Read Dataset-level Outcomes: ' + str(outcomes_df.shape))
		features_df['dataset_f_index'] = features_df.index
		outcomes_df['dataset_o_index'] = outcomes_df.index
		features_df = features_df[['fid', 'dataset_f_index']]
		outcomes_df = outcomes_df[['fid', 'dataset_o_index']]
		dataset_level_index = pd.merge(features_df, outcomes_df, on='fid', how='inner') # 删除没有common fid的项
		self.logger.log('Merged Dataset-level Indexes: ' + str(dataset_level_index.shape))
		self.dataset_level_index = dataset_level_index


	def load_field_features_and_outcomes(self):
		features_df = pd.read_csv(self.field_feature_file)
		outcomes_df = pd.read_csv(self.field_outcome_file)
		self.logger.log('Read Field-level Features: ' + str(features_df.shape))
		self.logger.log('Read Field-level Outcomes: ' + str(outcomes_df.shape))
		features_df['level_f_index'] = features_df.index
		outcomes_df['level_o_index'] = outcomes_df.index
		features_df = features_df[['fid', 'field_id', 'level_f_index']]
		outcomes_df = outcomes_df[[       'field_id', 'level_o_index']]
		field_level_index = pd.merge(features_df, outcomes_df, on='field_id', how='inner') # 删除没有common field_id的项
		# field_level_index = field_level_index.sort_values(by='fid', axis=0)
		self.logger.log('Merged Field-level Indexes: ' + str(field_level_index.shape))
		group_field_id_col = field_level_index.groupby('fid').apply(lambda x: x['field_id'].tolist())
		group_f_index_col = field_level_index.groupby('fid').apply(lambda x: x['level_f_index'].tolist())
		group_o_index_col = field_level_index.groupby('fid').apply(lambda x: x['level_o_index'].tolist())
		self.logger.log('Grouped Field-level features by fid and get {} charts'.format(len(group_f_index_col)))
		field_level_index = pd.DataFrame({
			'level_f_indexes': group_f_index_col,
			'level_o_indexes': group_o_index_col,
			'level_field_ids': group_field_id_col})
		field_level_index.insert(0, 'fid', field_level_index.index)
		self.field_level_index = field_level_index


	def save_indexes(self):
		self.dataset_level_index.to_csv(self.save_folder + 'dataset_level_index.csv', index=False)
		self.field_level_index.to_csv(self.save_folder + 'field_level_index.csv', index=False)


	def load_indexes(self):
		self.dataset_level_index = pd.read_csv(self.save_folder + 'dataset_level_index.csv')
		self.field_level_index = pd.read_csv(self.save_folder + 'field_level_index.csv')
		self.merged_feature_index = pd.merge(self.dataset_level_index, self.field_level_index, on='fid', how='inner')
		self.logger.log('merge features indexes :' + str(self.merged_feature_index.shape))
		self.merged_feature_index.to_csv(self.save_folder + 'merged_feature_index.csv', index=False)


### deal with raw data
	def load_raw_data(self, data_file, chunk_size=1000):
		self.logger.log('Loading raw data from ' + data_file)
		df = pd.read_csv(
			data_file,
			sep='\t',
			error_bad_lines=False,
			chunksize=chunk_size,
			encoding='utf-8'
		)
		return df


	def enumerate_chunks(self):
		raw_df_chunks = self.load_raw_data(self.data_file, chunk_size=self.chunk_size)
		self.logger.log('Start traversing chunks')
		self.logger.log('Parallelization: ' + str(self.parallelize))
		
		# 并行
		if self.parallelize:
			batch_size = multiprocessing.cpu_count()	# n_jobs
			self.logger.log('Number of jobs: ' + str(batch_size))
			self.logger.log('')
			# start
			batch_num	= 1
			chunk_batch	= []
			for i, chunk in enumerate(raw_df_chunks):
				chunk_num = i + 1
				chunk_batch.append(chunk)
				if chunk_num == (batch_size * batch_num):
					self.logger.log('Start batch {} [chunk {} ~ {}]'.format(
								batch_num, chunk_num-batch_size+1, chunk_num))
					pool = multiprocessing.Pool(batch_size)
					batch_start_time = time.time()
					batch_results = pool.map_async(self.extract_chunk_data, chunk_batch).get(9999999)
					self.write_batch_results(batch_results)
					batch_time_cost = time.time() - batch_start_time
					self.logger.log('Finish batch {}'.format(batch_num))
					self.logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
					self.logger.log('')
					batch_num	+= 1
					chunk_batch	= []
					pool.close()
				if chunk_num == 2101: break		# 文件第2101979行会报错，暂时没找到解决方案
			# process left overs
			if len(chunk_batch) != 0:
				self.logger.log('Start last batch {} [chunk {} ~ {}]'.format(
							batch_num, batch_size*(batch_num-1)+1, chunk_num))
				pool = multiprocessing.Pool(batch_size)
				batch_start_time = time.time()
				remaining_batch_results = pool.map_async(self.extract_chunk_data, chunk_batch).get(9999999)
				self.write_batch_results(remaining_batch_results)
				batch_time_cost = time.time() - batch_start_time
				self.logger.log('Finish last batch {}'.format(batch_num))
				self.logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
				self.logger.log('')
				pool.close()
		
		# 不并行
		else:
			self.logger.log('')
			for i, chunk in enumerate(raw_df_chunks):
				chunk_num = i + 1
				self.logger.log('Start chunk {}'.format(chunk_num))
				chunk_start_time = time.time()
				chunk_result = self.extract_chunk_data(chunk)
				self.write_batch_results([chunk_result])
				chunk_time_cost = time.time() - chunk_start_time
				self.logger.log('Finish chunk {}'.format(chunk_num))
				self.logger.log('Time cost: {:.1f}s'.format(chunk_time_cost))
				self.logger.log('')
				if chunk_num == 2101: break		# 文件第2101979行会报错，暂时没找到解决方案


	def extract_chunk_data(self, chunk):
		'''modified from feature_extraction.extract.extract_chunk_features'''
		# chunk_df = clean_chunk(chunk)
		# num_valid_charts = chunk_df.shape[0]
		# for chart_num, chart_obj in chunk_df.iterrows():
		# 	fid = chart_obj.fid
		# 	table_data = chart_obj.table_data
		# 	fields = table_data[list(table_data.keys())[0]]['cols']
		# 	sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
		# 	num_fields = len(sorted_fields)
		# 	if num_fields > self.MAX_FIELDS:
		# 		num_valid_charts -= 1
		# 		self.logger.log('Skip chart [{}] with {} fields'.format(fid, num_fields))
		# 		continue
		# 	# self.extract_chart_data(chart_obj)
		
		# 	# chart_outcomes = extract_chart_outcomes(chart_obj)
		# 	# field_outcomes = extract_field_outcomes(chart_obj)
		# 	# instance = Instance(fid, sorted_fields, chart_outcomes, field_outcomes)
		# 	# instance.to_single_html()
		# return {'num_valid_charts': num_valid_charts}

		# if self.dataset_level_index == None:
		# 	self.load_dataset_features_and_outcomes()
		chunk = chunk[['fid', 'table_data']]
		return pd.merge(self.dataset_level_index, chunk, on='fid', how='inner')


	def extract_chart_data(self, chart_obj):
		pass


	def write_batch_results(self, batch_results):
		output_file_name = self.save_folder + 'fid_dataset_index.csv'
		for df in batch_results:
			df.to_csv(output_file_name, mode='a', index=False, header=self.first_batch)
		self.first_batch = False



if __name__ == '__main__':

	# dataset_feature_file	= '../features/features_20180520-005740_processed_99_standard/features_aggregate_single_pairwise.csv'
	# dataset_outcome_file	= '../features/features_20180520-005740_processed_99_standard/chart_outcomes.csv'
	# field_feature_file		= '../features/features_20180520-005740_processed_99_standard/field_level_features.csv'
	# field_outcome_file		= '../features/features_20180520-005740_processed_99_standard/field_level_outcomes.csv'
	dataset_feature_file	= '../features/raw_1k/features_aggregate_single_pairwise.csv'
	dataset_outcome_file	= '../features/raw_1k/chart_outcomes.csv'
	field_feature_file		= '../features/raw_1k/field_level_features.csv'
	field_outcome_file		= '../features/raw_1k/field_level_outcomes.csv'

	features_files = [dataset_feature_file, dataset_outcome_file, field_feature_file, field_outcome_file]
	ex = Extracter('../data/plot_data.tsv', chunk_size=1000, parallelize=False, features_files=features_files)
	
	# ex.load_dataset_features_and_outcomes()
	# ex.load_field_features_and_outcomes()
	# ex.save_indexes()
	
	ex.load_indexes()
	# ex.enumerate_chunks()