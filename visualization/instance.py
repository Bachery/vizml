
# packages
from collections import OrderedDict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import json
import os


class Field(object):
	def __init__(self):
		self.field_outcome_feature_names = [
			'fid',
			'field_id',
			'trace_type',
			'is_xsrc',
			'is_ysrc',
			'is_zsrc',
			'is_locationsrc',
			'is_labelsrc',
			'is_valuesrc',
			# The only src of that type
			'is_single_xsrc',
			'is_single_ysrc',
			'is_single_zsrc',
			'is_single_locationsrc',
			'is_single_labelsrc',
			'is_single_valuesrc',
			# 'is_x_axis_src',
			# 'is_y_axis_src',
			# Multiple occurrences          # 是第几个该src，包含重复
			'num_xsrc',
			'num_ysrc',
			'num_zsrc',
			'num_locationsrc',
			'num_labelsrc',
			'num_valuesrc',
		]
		outcomes = OrderedDict([(f, None) for f in field_outcome_feature_names])




class Instance(object):
	def __init__(self, fid, fields, c_outcomes, f_outcomes):
		self.fid		= fid
		self.fields		= fields
		self.num_fields	= len(fields)
		self.chart_outcomes	= c_outcomes
		self.field_outcomes = f_outcomes
	
	def get_link(self):
		return 'https://plot.ly/~{0}/{1}'.format(*self.fid.split(':'))

	def to_single_html(self):
		pass