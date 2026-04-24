
import abc

class Decision_filter(abc.ABC):
    
    def __init__(self):
        pass

    @abc.abstractmethod
    def decider(self, window_probs, windows_predictions):
        pass