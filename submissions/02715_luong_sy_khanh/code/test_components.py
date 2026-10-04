"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC H). Chạy: python -m unittest test_components -v  (CPU, không cần dữ liệu)."""
import copy
import unittest

import numpy as np
import timm
import torch
import torch.nn.functional as F

import inference as I
import losses
import model as M
import train


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z, self.y = torch.randn(64, 9), torch.randint(0, 9, (64,))

    def test_focal_gamma0_equals_ce(self):
        self.assertLess(abs(losses.FocalLoss(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)).item(), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(losses.FocalLoss(2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_label_smoothing_eps0_equals_ce_and_matches_torch(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)).item(), 1e-6)
        ref = F.cross_entropy(self.z, self.y, label_smoothing=0.1)
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(self.z, self.y) - ref).item(), 1e-6)

    def test_class_weights(self):
        w = losses.class_weights([100, 10, 1], beta=0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertTrue(w[2] > w[1] > w[0])
        wb = losses.class_weights([100, 10, 1], beta=0.99)
        self.assertAlmostEqual(wb.sum().item(), 3.0, places=5)

    def test_cutmix_lambda_is_true_area_and_labels_mixed(self):
        np.random.seed(0)
        x = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(8)])
        y = torch.arange(8)
        for _ in range(20):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix")
            same = ya == yb                                           # perm có thể trùng chính nó -> ảnh không đổi, bỏ qua
            if (~same).any():                                         # lam phải bằng diện tích phần còn lại của ảnh gốc
                self.assertAlmostEqual(lam, (xm[~same, 0] == x[~same, 0]).float().mean().item(), places=5)
            self.assertTrue(0.0 <= lam <= 1.0)
            self.assertTrue(torch.equal(ya, y))

    def test_mixed_loss_is_convex_combo(self):
        ya, yb = self.y, self.y.roll(1)
        ce = torch.nn.CrossEntropyLoss()
        got = losses.mixed_loss(ce, self.z, (ya, yb, 0.3))
        self.assertAlmostEqual(got.item(), (0.3 * ce(self.z, ya) + 0.7 * ce(self.z, yb)).item(), places=5)


class TestModel(unittest.TestCase):
    def test_param_groups_no_wd_on_norm_bias_and_head_lr(self):
        net = timm.create_model("resnet18", pretrained=False, num_classes=9)
        g = M.param_groups(net, 1e-4, 1e-3, 0.05)
        self.assertEqual([x["weight_decay"] for x in g], [0.05, 0.0, 0.05])
        self.assertEqual([x["lr"] for x in g], [1e-4, 1e-4, 1e-3])
        self.assertTrue(all(p.ndim > 1 for p in g[0]["params"]) and all(p.ndim <= 1 for p in g[1]["params"]))
        self.assertEqual(sum(len(x["params"]) for x in g), len(list(net.parameters())))

    def test_freeze_backbone_trains_only_head(self):
        net = M.build_model("resnet18", False, 9, 0.0, "scratch")
        M.freeze_backbone(net)
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"fc.weight", "fc.bias"})


class TestInference(unittest.TestCase):
    def test_temperature_reduces_nll_and_keeps_argmax(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 9, 2000)
        z = rng.randn(2000, 9)
        z[np.arange(2000), np.where(rng.rand(2000) < 0.7, y, rng.randint(0, 9, 2000))] += 3.0
        z *= 4.0                                                    # quá tự tin
        T = I.fit_temperature(z, y)
        self.assertGreater(T, 1.5)
        nll = lambda p: -np.log(p[np.arange(len(y)), y]).mean()  # noqa: E731
        self.assertLess(nll(I.apply_temperature(z, T)), nll(I.apply_temperature(z, 1.0)))
        self.assertTrue((I.apply_temperature(z, T).argmax(1) == z.argmax(1)).all())

    def test_aggregate_views_probs_sum_to_one(self):
        L = [np.random.randn(5, 9) for _ in range(3)]
        for sp in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views(L, sp).sum(1), 1.0, atol=1e-9)

    def test_fuse_conv_bn_matches_original(self):
        net = timm.create_model("resnet18", pretrained=False, num_classes=9).eval()
        for m in net.modules():                                      # thống kê BN khác mặc định để kiểm tra có ý nghĩa
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
        fused = I.fuse_conv_bn(copy.deepcopy(net))
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertLess((fused(x) - net(x)).abs().max().item(), 1e-4)
        self.assertGreater(fused.n_fused, 10)


class TestTrain(unittest.TestCase):
    def test_ema_update_formula(self):
        net = torch.nn.Linear(2, 2)
        ema = train.EMA(net, 0.9)
        ema.n = 10**6                                                # d = decay (hết giai đoạn khởi động)
        before = ema.module.weight.detach().clone()
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        torch.testing.assert_close(ema.module.weight, 0.9 * before + 0.1 * net.weight.detach())

    def test_scheduler_warmup_then_cosine_to_zero(self):
        net = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=1.0)
        sch = train.build_scheduler(opt, train.Config(epochs=10, warmup_epochs=1.0), steps_per_epoch=10)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        self.assertLess(lrs[0], 0.05)
        self.assertAlmostEqual(max(lrs), 1.0, places=2)
        self.assertLess(lrs[-1], 0.01)
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(lrs[10:], lrs[11:])))

    def test_parse_overrides_types(self):
        d = train.parse_overrides(["seed=3", "lr_head=0.002", "ema_decay=none", "amp=false", "mix=cutmix", "backbone=x"])
        self.assertEqual(d, {"seed": 3, "lr_head": 0.002, "ema_decay": None, "amp": False, "mix": "cutmix", "backbone": "x"})
        with self.assertRaises(KeyError):
            train.parse_overrides(["nope=1"])


if __name__ == "__main__":
    unittest.main()
