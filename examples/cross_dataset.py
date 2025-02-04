import logging
import yaml
from moabb.datasets import BNCI2014001, Zhou2016
from moabb.paradigms import MotorImagery
from moabb.evaluations.evaluations import CrossDatasetEvaluation
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_pipeline() -> Pipeline:
    """Create the CSP + SVM pipeline manually."""
    return Pipeline([
        ('covariances', Covariances(estimator='oas')),
        ('csp', CSP(nfilter=6)),
        ('svc', SVC(kernel='linear'))
    ])

def main():
    # Define train and test datasets
    train_dataset = BNCI2014001()
    test_dataset = Zhou2016()
    
    # Initialize the paradigm
    paradigm = MotorImagery(n_classes=2)
    
    # Initialize the CrossDatasetEvaluation
    evaluation = CrossDatasetEvaluation(
        paradigm=paradigm,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pretrained_model=None,  # Not using a pre-trained model
        fine_tune=False,        # Not fine-tuning
        target_channels=None,   # Use all channels from train_dataset
        sfreq=128,              # Target sampling frequency
        channel_strategy='zero',# Strategy for handling channels
        montage='standard_1020',# EEG montage for SSI
        min_channels=3,         # Minimum common channels for subset strategy
        hdf5_path=None,         # Path to save results and models
        save_model=False        # Do not save models
    )
    
    # Create the pipeline
    pipeline = create_pipeline()
    
    # Define parameter grid if needed (optional)
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear']
    }
    
    # Run the evaluation
    results = evaluation.evaluate(
        dataset=None,          # Not used in CrossDatasetEvaluation
        pipelines={'CSP_SVM': pipeline},
        param_grid=param_grid
    )
    
    # Collect and display results
    for res in results:
        # Create log message with available information
        log_msg = [
            f"Dataset: {res['dataset'].code}",
            f"Subject: {res['subject']}",
            f"Pipeline: {res['pipeline']}",
            f"Score: {res['score']:.4f}",
            f"Time: {res['time']:.2f}s"
        ]
        
        # Add session info if available
        if 'session' in res:
            log_msg.insert(2, f"Session: {res['session']}")
            
        logger.info(", ".join(log_msg))
                    
if __name__ == "__main__":
    main()
