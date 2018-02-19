from . import utils
import unittest


class Test_Utils(unittest.TestCase):

    def test_channel_intersection_fun(self):
        print(utils.find_intersecting_channels(
            [d() for d in utils.dataset_list])[0])

    def test_dataset_search_fun(self):
        print([type(i).__name__ for i in utils.dataset_search(
            'imagery', multi_session=True)])
        print([type(i).__name__ for i in utils.dataset_search(
            'imagery', multi_session=False)])
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'], two_class=False)
        for out in l:
            print('multiclass: {}'.format(out.selected_events))
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'feet'], two_class=False, exact_events=True)
        for out in l:
            print('rh/f: {}, {}'.format(type(out).__name__, out.selected_events))
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'], two_class=True)
        for out in l:
            print('two class: {}'.format(out.selected_events))
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'],
            two_class=True, channels=['C3', 'C4'])
        for out in l:
            print('C3 C4: {}'.format(out.selected_events))


if __name__ == '__main__':
    unittest.main()
