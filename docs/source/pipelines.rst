=========
Pipelines
=========

.. automodule:: moabb.pipelines

.. currentmodule:: moabb.pipelines

---------
Pipelines
---------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    features.LogVariance
    features.FM
    features.ExtendedSSVEPSignal
    features.AugmentedDataset
    features.StandardScaler_Epoch
    csp.TRCSP
	classification.SSVEP_CCA
    deep_learning.KerasDeepConvNet
    deep_learning.KerasEEGITNet
    deep_learning.KerasEEGNet_8_2
    deep_learning.KerasEEGNeX
    deep_learning.KerasEEGTCNet
    deep_learning.KerasShallowConvNet


------------
Base & Utils
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.create_pipeline_from_config
    utils.FilterBank
    utils_deep_model.EEGNet
    utils_deep_model.EEGNet_TC
    utils_deep_model.TCN_block
    utils_pytorch.BraindecodeDatasetLoader
    utils_pytorch.InputShapeSetterEEG
