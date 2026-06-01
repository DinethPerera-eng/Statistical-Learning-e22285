from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from google.colab import files
except:
    files = None

from scipy.stats import chi2_contingency, pointbiserialr, f_oneway
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
