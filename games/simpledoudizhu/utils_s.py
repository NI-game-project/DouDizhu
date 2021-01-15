''' Doudizhu utils
'''

import os
import json
from collections import OrderedDict


# Read required docs
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())
