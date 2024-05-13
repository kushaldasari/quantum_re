
import os

file_path = os.path.dirname(__file__)
data_path = os.path.realpath(os.path.join(file_path, '../data/')) + '/'
tf_save_path = os.path.realpath(os.path.join(file_path, '../tf_save/')) + '/'
tf_log_path = os.path.realpath(os.path.join(file_path, '../tf_log/')) + '/'
