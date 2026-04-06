"""Global Constants for the Project"""

import os
from environ.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
PAPER_FIGURE_PATH = PROJECT_ROOT / "lido-bank" / "figures"
PAPER_TABLE_PATH = PROJECT_ROOT / "lido-bank" / "tables"

for directory in [DATA_PATH, PROCESSED_DATA_PATH, PAPER_FIGURE_PATH, PAPER_TABLE_PATH]:
    os.makedirs(directory, exist_ok=True)
