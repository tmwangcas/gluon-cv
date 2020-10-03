from gluoncv.auto.tasks import ObjectDetection
from gluoncv.auto.estimators import SSDEstimator

# task = ObjectDetection({'dataset': 'voc', 'epoch': 0})
# det = SSDEstimator(config={'dataset': 'voc', 'gpus': [], 'train': {'epochs': 0}, 'num_workers': 0})
det = SSDEstimator(config={'dataset': 'voc_tiny', 'gpus': [0,1,2,3], 'train': {'epochs': 1}, 'num_workers': 4})
det.fit()

det.save('test.pkl')
det = ObjectDetection.load('test.pkl')
