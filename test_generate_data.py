import unittest
import torch
import cf
from generate_data import generate_data
#import os

class TestGenerateData(unittest.TestCase):

    def setUp(self):
        self.input_file = '../data1.nc'
#         pass #use pre-existing
        # Create a mock input file using cf-python
#        f = cf.Field()
#        f.set_construct(cf.DimensionCoordinate(data=cf.Data(range(10)), properties={'standard_name': 'latitude'}))
#        f.set_construct(cf.DimensionCoordinate(data=cf.Data(range(10)), properties={'standard_name': 'pressure'}))
#        f.set_data(cf.Data([[i + j for j in range(10)] for i in range(10)]))
#        cf.write(f, self.input_file)

    def tearDown(self):
        pass
        # Remove the mock input file after tests
#        os.remove(self.input_file)

    def test_generate_data_no_test_split(self):
        mydata, x_train, y_train, x_test, y_test = generate_data(self.input_file)
        self.assertIsInstance(mydata, cf.Field)
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertEqual(x_train.shape, torch.Size([3680,2]))
        self.assertEqual(y_train.shape, torch.Size([3680,1]))
        self.assertEqual(x_test, None)
        self.assertEqual(y_test, None)
        torch.testing.assert_close(x_train[0], torch.Tensor([89.1415, 1000.0000]))
        torch.testing.assert_close(y_train[0], torch.Tensor([-1.0029]), rtol=1.e-4, atol=1.e-4)
        torch.testing.assert_close(x_train[3], torch.Tensor([89.1415, 775.0000]))
        torch.testing.assert_close(y_train[3], torch.Tensor([-1.2893]), rtol=1.e-4, atol=1.e-4)

    def test_generate_data_with_test_split(self):
        mydata, x_train, y_train, x_test, y_test = generate_data(self.input_file, n_test=10)
        self.assertIsInstance(mydata, cf.Field)
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertEqual(x_train.shape, torch.Size([3670,2]))
        self.assertEqual(y_train.shape, torch.Size([3670,1]))
        self.assertEqual(x_test.shape, torch.Size([10,2]))
        self.assertEqual(y_test.shape, torch.Size([10,1]))
        torch.testing.assert_close(x_test[0], torch.Tensor([89.1415, 1000.0000]))
        torch.testing.assert_close(y_test[0], torch.Tensor([-1.0029]), rtol=1.e-4, atol=1.e-4)
        torch.testing.assert_close(x_train[2], torch.Tensor([89.1415, 775.0000]))
        torch.testing.assert_close(y_train[2], torch.Tensor([-1.2893]), rtol=1.e-4, atol=1.e-4)

if __name__ == '__main__':
    unittest.main()