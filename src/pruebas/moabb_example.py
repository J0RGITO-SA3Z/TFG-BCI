import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

moabb.set_log_level("info")

pipelines = {"LogVar+LDA": make_pipeline(LogVariance(), LDA())}

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:2]

paradigm = LeftRightImagery(fmin=8, fmax=35)
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset])
results = evaluation.process(pipelines)

print(results.head())