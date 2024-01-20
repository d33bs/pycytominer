from .aggregate import aggregate
from .annotate import annotate
from .feature_select import feature_select
from .normalize import normalize
from .consensus import consensus
from pycytominer import __about__

import pandas as pd

# set copy_on_write for pandas to True
# see here for more details:
# https://pandas.pydata.org/pandas-docs/version/2.2.0/whatsnew/v2.2.0.html#copy-on-write
pd.options.mode.copy_on_write = True
