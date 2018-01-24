'''
Tests to ensure that datasets download correctly
'''
from moabb.datasets.gigadb import GigaDbMI
from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.bnci import BNCI2014001, BNCI2014002, BNCI2014004, BNCI2015001, BNCI2015004
from moabb.datasets.openvibe_mi import OpenvibeMI
import unittest

def test_dataset(data):
    obj = data()
    obj.subject_list = obj.subject_list[:2]
    obj.get_data(obj.subject_list)

class Test_Datasets(unittest.TestCase):

    def test_gigadb(self):
        test_dataset(GigaDbMI)

    def test_bnci(self):
        test_dataset(BNCI2014001)
        test_dataset(BNCI2014002)
        test_dataset(BNCI2014004)
        test_dataset(BNCI2015001)
        test_dataset(BNCI2015004)

    def test_alexmi(self):
        test_dataset(AlexMI)

    def test_ovmi(self):
        test_dataset(OpenvibeMI)

    def test_physionet(self):
        test_dataset(PhysionetMI)

if __name__ == '__main__':
    unittest.main()
