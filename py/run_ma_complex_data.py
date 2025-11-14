import json
import numpy as np
from argparse import Namespace
import sys
import os
import io
from contextlib import redirect_stdout

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.ma import run

# Generate moderately complex data
np.random.seed(42) # For reproducibility
n_points = 100
base_value = 100
trend = np.linspace(0, 50, n_points)
seasonality = 10 * np.sin(np.linspace(0, 3 * np.pi, n_points))
noise = np.random.normal(0, 3, n_points)

data_values = base_value + trend + seasonality + noise
data = [{"sales": float(v)} for v in data_values]
data_json = json.dumps(data)

# Define args for ma.py training run
args = Namespace(
    data=data_json,
    train_start=0,
    train_end=69,
    eval_start=70,
    eval_end=99,
    window=7,
    save_model=None, # Not saving model for this run
    load_model=None,
    pretty=True # For readable output
)

# Run ma.py and capture output
output_io = io.StringIO()
with redirect_stdout(output_io):
    run(args)

print(output_io.getvalue())
