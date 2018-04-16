from sklearn.pipeline import make_pipeline


def create_pipeline_from_config(config):
    """Create a pipeline from a config file.

    takes a config dict as input and return the coresponding pipeline.

    Parameters
    ----------
    config : Dict.
        Dict containing the config parameters.

    Return
    ------
    pipeline : Pipeline
        sklearn Pipeline

    """
    components = []

    for component in config:
        # load the package
        mod = __import__(component['from'], fromlist=[component['name']])
        # create the instance
        if 'parameters' in component.keys():
            params = component['parameters']
        else:
            params = {}
        instance = getattr(mod, component['name'])(**params)
        components.append(instance)

    pipeline = make_pipeline(*components)
    return pipeline
