import unittest

from moabb.datasets import utils


class Test_Utils(unittest.TestCase):
    def test_channel_intersection_fun(self):
        print(utils.find_intersecting_channels([d() for d in utils.dataset_list])[0])

    def test_dataset_search_fun(self):
        found = utils.dataset_search("imagery", multi_session=True)
        print([type(dataset).__name__ for dataset in found])
        found = utils.dataset_search("imagery", multi_session=False)
        print([type(dataset).__name__ for dataset in found])
        res = utils.dataset_search(
            "imagery", events=["right_hand", "left_hand", "feet", "tongue", "rest"]
        )
        for out in res:
            print("multiclass: {}".format(out.event_id.keys()))

        res = utils.dataset_search(
            "imagery", events=["right_hand", "feet"], has_all_events=True
        )
        for out in res:
            self.assertTrue(set(["right_hand", "feet"]) <= set(out.event_id.keys()))

    def test_dataset_channel_search(self):
        chans = ["C3", "Cz"]
        All = utils.dataset_search(
            "imagery", events=["right_hand", "left_hand", "feet", "tongue", "rest"]
        )
        has_chans = utils.dataset_search(
            "imagery",
            events=["right_hand", "left_hand", "feet", "tongue", "rest"],
            channels=chans,
        )
        has_types = set([type(x) for x in has_chans])
        for d in has_chans:
            s1 = d.get_data([1])[1]
            sess1 = s1[list(s1.keys())[0]]
            raw = sess1[list(sess1.keys())[0]]
            self.assertTrue(set(chans) <= set(raw.info["ch_names"]))
        for d in All:
            if type(d) not in has_types:
                s1 = d.get_data([1])[1]
                sess1 = s1[list(s1.keys())[0]]
                raw = sess1[list(sess1.keys())[0]]
                self.assertFalse(set(chans) <= set(raw.info["ch_names"]))


if __name__ == "__main__":
    unittest.main()
