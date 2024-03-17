import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

MODELS_DIR = os.path.join(INPUTS_DIR, "models")

TEST_KEYS_DIR = os.path.join(INPUTS_DIR, "datasets_test_keys")
