from moabb import utils
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
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'])
        for out in l:
            print('multiclass: {}'.format(out.selected_events))
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'feet'], has_all_events=True)
        for out in l:
            print('rh/f: {}, {}'.format(type(out).__name__, out.selected_events))
        l = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'],total_classes=2)
        for out in l:
            print('two class: {}'.format(out.selected_events))


    def test_dataset_channel_search(self):
        all_datasets = utils.dataset_list
        chans = ['C3','Cz']
        All = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'])
        has_chans = utils.dataset_search('imagery', events=[
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest'], channels=chans)
        has_types = set([type(x) for x in has_chans])
        for d in has_chans:
            s1 = d.get_data([1], False)[0][0][0]
            self.assertTrue(set(chans) <= set(s1.info['ch_names']))
        for d in All:
            if type(d) not in has_types:
                s1 = d.get_data([1], False)[0][0][0]
                self.assertFalse(set(chans) <= set(s1.info['ch_names']))

if __name__ == '__main__':
    unittest.main()
