"""
Training Pipeline:
- Loads configuration
- Initializes production database
- Runs the full training pipeline (preprocessing, feature engineering, training, postprocessing)
- Saves the trained model to the models folder
"""

import os
import sys
from pathlib import Path


from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager
from common.utils import read_config





project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root) # Change directory to read the files from ./data folder


if __name__ == "__main__":

    # Load config file
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    # Initialize production database with historical raw data
    data_manager = DataManager(config)
    #data_manager.initialize_prod_database(project_root=project_root)



    # Initialize Pipeline Runner
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)



    # Run the training pipeline
    #report = pipeline_runner.run_inference(df_inference_batch,batch_number=10)[2]
    pipeline_runner.run_training()

