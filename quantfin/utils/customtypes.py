from typing import Union
from pandas import Timestamp
from datetime import date, datetime
from numpy.core import datetime64

Date = Union[str, Timestamp, date, datetime, datetime64]
