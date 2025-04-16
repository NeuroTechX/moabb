"""
===================
Dataset bubble plot
===================

This tutorial shows how to use the :func:`moabb.analysis.plotting.dataset_bubble_plot`
function to visualize, at a glance, the number of subjects and sessions in each dataset
and the number of trials per session.

"""

# Authors: Pierre Guetschel
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import moabb
from moabb.analysis.plotting import dataset_bubble_plot
from moabb.datasets import (
    BNCI2014_001,
    Cho2017,
    Hinss2021,
    Lee2019_ERP,
    Sosulski2019,
    Thielen2015,
    Wang2016,
)


moabb.set_log_level("info")

print(__doc__)

###############################################################################
# Visualizing one dataset
# -----------------------
#
# The :func:`moabb.analysis.plotting.dataset_bubble_plot` is fairly simple to use.
# It takes a :class:`moabb.datasets.base.BaseDataset` as input and plots
# its characteristics.
# You can adjust plotting parameters, such as the scale of the bubbles, but
# we will leave the default values for this example.
# More details on the parameters can be found in the doc (:func:`moabb.analysis.plotting.dataset_bubble_plot`).


dataset = Lee2019_ERP()
dataset_bubble_plot(dataset)
plt.show()


##############################################################################
# In this example, we can see that the :class:`moabb.datasets.Lee2019_ERP` dataset
# has many subjects (54), 2 sessions, and a fairly large number of trials per session.
#
# Visualizing multiple datasets simultaneously
# --------------------------------------------
#
# Multiple datasets can be visualized at once by using the ``ax`` and ``center`` parameters.
# The ``ax`` parameter allows you to re-plot on the same axis, while the ``center`` parameter
# allows you to specify the center of each dataset.
# The following example shows how to plot multiple datasets on the same axis.

ax = plt.gca()
dataset_bubble_plot(Lee2019_ERP(), ax=ax, center=(10, 10), legend=False)
dataset_bubble_plot(BNCI2014_001(), ax=ax, center=(-2, 33), legend=False)
dataset_bubble_plot(Wang2016(), ax=ax, center=(37, 0), legend=True)
dataset_bubble_plot(Thielen2015(), ax=ax, center=(36, 16), legend=False)
dataset_bubble_plot(Hinss2021(), ax=ax, center=(31, 22), legend=False)
dataset_bubble_plot(Cho2017(), ax=ax, center=(33, 35), legend=False)
dataset_bubble_plot(Sosulski2019(), ax=ax, center=(13, 42), legend=False)
plt.show()
