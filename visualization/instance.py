
# packages
from pyecharts.charts import Bar, Line, Scatter, Pie, Grid, Page
from pyecharts import options as opts
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


class Trace(object):
	pass



class Instance(object):
	def __init__(self, fid, fields, c_outcomes, f_outcomes):
		self.fid				= fid
		self.fields				= fields
		self.num_fields			= len(fields)
		self.num_traces 		= c_outcomes['num_traces']
		self.trace_types		= c_outcomes['trace_types']
		self.fields_by_trace	= c_outcomes['fields_in_each_trace']
		self.chart_outcomes		= c_outcomes
		self.field_outcomes 	= f_outcomes

		self.charts = []
		self.charts_dir = './html/'
		self.name = self.fid.split(':')[0] + '_' + self.fid.split(':')[1]
	

	def get_link(self):
		return 'https://plot.ly/~{0}/{1}'.format(*self.fid.split(':'))


	def find_field_from_uid(self, uid):
		for field in self.fields:
			if field[1]['uid'] == uid:
				return field
		print('No such field with uid ', uid)


	def generate_views(self):
		def generate_a_view(data):
			# 设置图标基本属性
			margin = '5%'
			if data['chart'] == 'bar':
				chart = (Bar().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
							.set_global_opts(title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
											xaxis_opts=opts.AxisOpts(name=data['x_name']),
											yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
			elif data['chart'] == 'pie': 
				chart = (Pie().set_global_opts(title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin)))
			elif data['chart'] == 'line': 
				chart = (Line().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
							.set_global_opts(title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
												xaxis_opts=opts.AxisOpts(name=data['x_name']),
												yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
			elif data['chart']== 'scatter': 
				chart = (Scatter().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
								.set_global_opts(title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
												xaxis_opts=opts.AxisOpts(type_='value', name=data['x_name'], splitline_opts=opts.SplitLineOpts(is_show=True)),
												yaxis_opts=opts.AxisOpts(type_='value', name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
			else :
				print ("not valid chart")
			# 添加数据
			attr = data["x_data"] # 横坐标
			val = data["y_data"] # 纵坐标
			if data['chart'] == 'bar':       
				chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
			elif data['chart'] == 'line':    
				chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
			elif data['chart'] == 'pie':     
				chart.add("", [list(z) for z in zip(attr, val)])
			elif data['chart'] == 'scatter': 
				if isinstance(attr[0], str):
					attr = [x for x in attr if x != '']
					attr = list(map(float, attr))
				if isinstance(val[0], str):
					val = [x for x in val if x != '']
					val = list(map(float, val))
				chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
			return chart

		for trace_id in range(self.num_traces):
			fields_src = self.fields_by_trace[trace_id]
			if len(fields_src) != 2: continue
			# xsrc = '{}:{}'.format(fields_src[0].split(':')[0], fields_src[0].split(':')[-1])
			# ysrc = '{}:{}'.format(fields_src[1].split(':')[0], fields_src[1].split(':')[-1])
			x_uid = fields_src[0].split(':')[-1]
			y_uid = fields_src[1].split(':')[-1]
			x_field = self.find_field_from_uid(x_uid)
			y_field = self.find_field_from_uid(y_uid)
			
			view_info = {}
			view_info['chartname']	= str(trace_id)
			view_info['describe']	= ''
			view_info['x_name']		= x_field[0]
			view_info['y_name']		= y_field[0]
			view_info['chart']		= self.trace_types[trace_id]
			view_info['x_data']		= x_field[1]['data']
			view_info['y_data']		= y_field[1]['data']
			
			self.charts.append(generate_a_view(view_info))


	def to_single_html(self):
		if len(self.charts) == 0:
			self.generate_views()
		page = Page()
		self.page = Page()
		for chart in self.charts:
			grid = Grid()
			grid.add(chart, grid_opts=opts.GridOpts(pos_bottom='20%', pos_top='20%'))
			self.page.add(grid)
		self.page.render(self.charts_dir + self.name + '.html')