import unittest

import numpy as np
import torch

from src.python.training.losses import DiceSegmentationLoss


class LossesTestCase(unittest.TestCase):

    def test_unit_dice(self):
        m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        m2 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        l = DiceSegmentationLoss(reduction="none", apply_softmax=False)(m1, m2)
        assert l.shape == torch.Size([3])
        np.testing.assert_almost_equal(l.numpy(), np.zeros(3))

    def test_unit_dice_2(self):
        m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        m2 = torch.tensor(np.zeros((3, 5, 10, 10, 10)))
        l = DiceSegmentationLoss(reduction="none", apply_softmax=False)(m1, m2)
        assert l.shape == torch.Size([3])
        np.testing.assert_almost_equal(l.numpy(), np.ones(3))

    def test_unit_dice_one_class_zero(self):
        pred = np.ones((3, 3, 10, 10, 10))
        pred[:, 1, :, :, :] = 0

        gt = np.ones((3, 3, 10, 10, 10))
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        l = DiceSegmentationLoss(reduction="none", apply_softmax=False)(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), np.ones(3) * 0.33, decimal=2)

    def test_unit_dice_one_batch_zero(self):
        pred = np.ones((3, 3, 10, 10, 10))
        pred[0, :, :, :, :] = 0
        gt = np.ones((3, 3, 10, 10, 10))
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        l = DiceSegmentationLoss(reduction="none", apply_softmax=False)(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), [1, 0, 0])
        l = DiceSegmentationLoss(reduction="mean", apply_softmax=False)(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), 0.33, decimal=2)

    def test_unit_ce(self):
        import numpy as np
        m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        m2 = torch.tensor(np.ones((3, 5, 10, 10, 10)))

        ce = torch.nn.CrossEntropyLoss()
        l = ce(m1, m2)
        assert l.shape == torch.Size([])
