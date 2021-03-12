import logging

class logger():
	def __init__(self, log_file, parameters):
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(message)s')
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		self.logger.addHandler(file_handler)
		self.log('PARAMETERS:')
		self.log_dict(parameters)

	def log(self, info):
		self.logger.info(info)
		print(info)

	def log_dict(self, dict):
		for k, v in dict.items():
			if type(v) != str:
				v = str(v)
			self.log(k + ': ' + v)
		# for kv in dict.items():
		# 	self.log(kv)
		self.log('\n')
