"""Global Constants for the Project"""

import os
from environ.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"

for directory in [DATA_PATH, PROCESSED_DATA_PATH, FIGURE_PATH, TABLE_PATH]:
    os.makedirs(directory, exist_ok=True)
