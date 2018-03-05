from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
import matplotlib.pyplot as plt


def score_plot(data):
    '''
    Input:
        data: dataframe

    Out:
        ax: pyplot Axes reference
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sea.violinplot(data=data, y="score", x="dataset",
                        hue="pipeline", inner="stick", cut=0, ax=ax)
    ax.set_ylim([0.5, 1])

    return fig


def time_line_plot(data):
    '''
    plot data entries per timepoint
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data['n_entries'] = data['samples']*data['channels']

    for p in data['pipeline'].unique():
        ax.scatter(data[data['pipeline'] == p]['n_entries'],
                   data[data['pipeline'] == p]['time'])
    ax.legend(data['pipeline'].unique())
    ax.set_xlabel('Entries in training matrix')
    ax.set_ylabel('Time to fit decoding model')
    return fig


def time_plot(data):
    '''
    Input:
    data: dataframe

    Out:
    ax: pyplot Axes reference
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for p in data['pipeline'].unique():
        ax.scatter(data[data['pipeline'] == p]['channels'],
                   data[data['pipeline'] == p]['samples'],
                   data[data['pipeline'] == p]['time'])
    ax.legend(data['pipeline'].unique())

    ax.set_xlabel('Number of channels')
    ax.set_ylabel('Training samples')
    ax.set_zlabel('Time to fit decoding model')

    return fig
