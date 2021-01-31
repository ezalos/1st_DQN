from utils import DotDict
from functools import reduce

config = DotDict()

config.q_table_size_array = [4,4,5,5] #4 * 4
config.q_table_size = reduce((lambda x, y: x * y), config.q_table_size_array)
config.discretisation_method = "sqrt"