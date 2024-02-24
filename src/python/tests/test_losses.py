import unittest
import numpy as np
import torch
from src.python.training.losses import DiceSegmentationLoss

class LossesTestCase(unittest.TestCase):

    def test_dice_loss_with_equal_masks(self):
        pred = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        gt = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        loss_fn = DiceSegmentationLoss(reduction="none", apply_softmax=False, ignore_background=False)
        l = loss_fn(pred, gt)
        assert l.shape == torch.Size([3])
        np.testing.assert_almost_equal(l.numpy(), np.zeros(3))

    def test_dice_loss_with_different_masks(self):
        pred = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        gt = torch.tensor(np.zeros((3, 5, 10, 10, 10)))
        loss_fn = DiceSegmentationLoss(reduction="none", apply_softmax=False, ignore_background=False)
        l = loss_fn(pred, gt)
        assert l.shape == torch.Size([3])
        np.testing.assert_almost_equal(l.numpy(), np.ones(3))

    def test_dice_loss_with_ignore_background(self):
        pred_bg = np.ones((3, 5, 10, 10, 10))
        pred_bg[:, 0, :, :, :] = 0
        pred = torch.tensor(pred_bg)
        gt = torch.tensor(np.ones((3, 5, 10, 10, 10)))

        loss_fn_ignore_bg = DiceSegmentationLoss(reduction="mean", apply_softmax=False, ignore_background=False)
        l = loss_fn_ignore_bg(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), 0.2)

        loss_fn_ignore_bg_true = DiceSegmentationLoss(reduction="mean", apply_softmax=False, ignore_background=True)
        l = loss_fn_ignore_bg_true(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), 0)

    def test_dice_loss_with_one_class_zero(self):
        pred = np.ones((3, 3, 10, 10, 10))
        pred[:, 1, :, :, :] = 0
        gt = np.ones((3, 3, 10, 10, 10))
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        loss_fn = DiceSegmentationLoss(reduction="none", apply_softmax=False, ignore_background=False)
        l = loss_fn(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), np.ones(3) * 0.33, decimal=2)

    def test_dice_loss_with_one_batch_zero(self):
        pred = np.ones((3, 3, 10, 10, 10))
        pred[0, :, :, :, :] = 0
        gt = np.ones((3, 3, 10, 10, 10))
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        loss_fn = DiceSegmentationLoss(reduction="none", apply_softmax=False, ignore_background=False)
        l = loss_fn(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), [1, 0, 0])
        l = DiceSegmentationLoss(reduction="mean", apply_softmax=False, ignore_background=False)(pred, gt)
        np.testing.assert_almost_equal(l.numpy(), 0.33, decimal=2)

    def test_cross_entropy_loss(self):
        pred = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        gt = torch.tensor(np.ones((3, 5, 10, 10, 10)))

        ce = torch.nn.CrossEntropyLoss()
        l = ce(pred, gt)
        assert l.shape == torch.Size([])
