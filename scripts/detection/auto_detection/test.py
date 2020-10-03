# from gluoncv.auto.tasks import ObjectDetection
# from gluoncv.auto import SSDEstimator

from gluoncv.auto.estimators import SSDEstimator
from gluoncv.auto.tasks.object_detection import ObjectDetection

#task = ObjectDetection({'dataset': 'voc', 'epoch': 0})
det = SSDEstimator(config={'dataset': 'voc_tiny', 'gpus': [0,1,2,3], 'train': {'epochs': 1}, 'num_workers': 4})
det.fit()

# ans = det.state_dict()
# print(ans)

det.save('test.pkl')
# det = ObjectDetection.load('test.pkl')