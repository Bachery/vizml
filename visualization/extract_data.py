
# packages
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import h5py
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
	def __init__(self, data_file):
		log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
		log_file = './logs/extracter_log_' + log_suffix + '.txt'
		self.logger = logger(log_file, {'MISSION': 'extract data for vis'})
		self.data_file		= data_file
		self.parallelize	= False
		self.chunk_size		= 1000
		self.save_folder	= './saves/'
		self.MAX_FIELDS		= 25


	def load_raw_data(self, data_file, chunk_size=1000):
		self.logger.log('Loading raw data from ' + data_file)
		df = pd.read_table(
			data_file,
			error_bad_lines=False,
			chunksize=chunk_size,
			encoding='utf-8'
		)
		return df


	def extract_chunk_data(self, chunk_df):
		'''modified from feature_extraction.extract.extract_chunk_features'''
		for chart_num, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			table_data = chart_obj.table_data
			fields = table_data[list(table_data.keys())[0]]['cols']
			sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
			num_fields = len(sorted_fields)
			if num_fields > self.MAX_FIELDS:
				self.logger.log('Skip chart ' + str(chart_num) + ' (' + fid + ')' + \
								'with ' + str(num_fields) + ' fields')
				continue
			chart_outcomes = extract_chart_outcomes(chart_obj)
			field_outcomes = extract_field_outcomes(chart_obj)
			instance = Instance(fid, sorted_fields, chart_outcomes, field_outcomes)
			save_file = self.save_folder + str(chart_num) + '_{0}_{1}.h5'.format(*fid.split(':'))
			with h5py.File(save_file, 'w') as f:
				f['instance'] = instance


	def main_process(self):
		raw_df_chunks = self.load_raw_data(self.data_file, chunk_size=self.chunk_size)
		if self.parallelize: pass
		else:
			self.logger.log('Start traversing chunks')
			for i, chunk in enumerate(raw_df_chunks):
				chunk_num = i + 1
				self.logger.log('Chunk Num: ' + str(chunk_num))
				df = clean_chunk(chunk)
				self.extract_chunk_data(df)



if __name__ == '__main__':

	ex = Extracter('../data/plot_data.tsv')
	ex.main_process()