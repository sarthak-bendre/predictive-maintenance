"""Central place for paths, column names and pipeline constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# Raw column name -> clean snake_case name
RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
ID_COLS = ["udi", "product_id"]
# Each failure-mode flag implies machine_failure == 1, so they leak the label.
LEAKY_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]
SENSOR_COLS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
PRODUCT_TYPES = ["L", "M", "H"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Business assumption used to pick the decision threshold:
# missing a failure costs this many times more than a false alarm.
COST_FN = 10.0
COST_FP = 1.0

MLFLOW_EXPERIMENT = "predictive-maintenance"
REGISTERED_MODEL_NAME = "pm-failure-classifier"
