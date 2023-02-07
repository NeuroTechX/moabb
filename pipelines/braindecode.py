from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from sklearn.pipeline import Pipeline
from skorch.callbacks import LRScheduler
from torch.cuda import is_available
from torch.nn import NLLLoss
from torch.optim import AdamW

from moabb.pipelines.braindecode import (
    BraindecodeClassifierModel,
    CreateBraindecodeDataset,
)


# hard-coded for now
n_classes = 2
n_chans = 22
input_window_samples = 1001
# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 4

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)
device = "cuda" if is_available() else "cpu"

clf = EEGClassifier(
    model,
    criterion=NLLLoss,
    optimizer=AdamW,
    train_split=None,  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
)

create_dataset = CreateBraindecodeDataset()
fit_params = {"epochs": 10}

clf_braindecode = BraindecodeClassifierModel(clf, fit_params)

pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", clf_braindecode)])

pipes = {"ShallowFBCSPNet": pipe}

PIPELINE = {
    "name": "ShallowFBCSPNet",
    "paradigms": ["LeftRightImagery"],
    "pipeline": pipe,
}
