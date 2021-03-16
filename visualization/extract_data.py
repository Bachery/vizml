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

from instance import Instance
sys.path.insert(0, '..')
from neural_network.logger import logger
from feature_extraction.general_helpers import clean_chunk
from feature_extraction.outcomes.chart_outcomes import extract_chart_outcomes
from feature_extraction.outcomes.field_encoding_outcomes import extract_field_outcomes

class Extracter(object):
	def __init__(self, data_file, parallelize=True, chunk_size=1000):
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

		if not os.path.exists(self.save_folder):
			os.mkdir(self.save_folder)


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

		self.logger.log('Finish. Total skipped charts num: {}'.format(self.num_charts_exceeding_max_fields))


	def extract_chunk_data(self, chunk):
		'''modified from feature_extraction.extract.extract_chunk_features'''
		chunk_df = clean_chunk(chunk)
		num_valid_charts = chunk_df.shape[0]
		for chart_num, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			table_data = chart_obj.table_data
			fields = table_data[list(table_data.keys())[0]]['cols']
			sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
			num_fields = len(sorted_fields)
			if num_fields > self.MAX_FIELDS:
				num_valid_charts -= 1
				self.logger.log('Skip chart [{}] with {} fields'.format(fid, num_fields))
				continue
			# self.extract_chart_data(chart_obj)
		return {'num_valid_charts': num_valid_charts}
			# chart_outcomes = extract_chart_outcomes(chart_obj)
			# field_outcomes = extract_field_outcomes(chart_obj)
			# instance = Instance(fid, sorted_fields, chart_outcomes, field_outcomes)
			# instance.to_single_html()


	def extract_chart_data(self, chart_obj):
		pass


	def write_batch_results(self, batch_results):
		if self.first_batch:
			self.logger.log('fucccccccccck')
		self.first_batch = False


if __name__ == '__main__':

	ex = Extracter('../data/plot_data.tsv', chunk_size=10, parallelize=True)
	ex.enumerate_chunks()