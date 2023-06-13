from moabb.paradigms.p300 import SinglePass

class RestingStateToP300Adapter(SinglePass):
    """P300 for Target/NonTarget classification

    Metric is 'roc_auc'

    """

    def __init__(self, fmin=10, fmax=50, tmin=1, tmax=35, resample=128, **kwargs):
        super().__init__(fmin=fmin, fmax=fmas, tmin=tmin, tmax=tmax, resample=resample, **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "roc_auc"

phmdml = RestingStateToP300Adapter(events=["ON", "OFF"])