import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image

from markprobe_core import (MarkProbe, bits_to_text, decode_image, encode_image,
                            load_checkpoint, save_checkpoint, text_to_bits,
                            train_model, run_finetune)
from markprobe_core.finetune import merge_finetune_settings
from markprobe_core.engine import (_StegaStampDiscriminator, _warp_projective, jpeg_compress_decompress, quantize_8bit_ste,
                                   _reference_geometry_embed, crop_resize_attack, stegastamp_transform,
                                   _stegastamp_yuv_loss, _edge_falloff, _rgb_yuv, _linear_ramp)


class CoreTests(unittest.TestCase):
    def test_text_payload(self):
        text = "hello"
        self.assertEqual(bits_to_text(text_to_bits(text, 96)), text)
        bits = text_to_bits("demo", 100)
        self.assertEqual(len(bits), 100)
        self.assertEqual(bits_to_text(bits), "demo")
        with self.assertRaises(ValueError):
            text_to_bits("x" * 12, 96)

    def test_forward_backward_and_roundtrip(self):
        model = MarkProbe(message_bits=12, image_size=32)
        image = torch.rand(2, 3, 32, 32)
        message = torch.randint(0, 2, (2, 12)).float()
        encoded, logits = model(image, message)
        self.assertEqual(tuple(encoded.shape), tuple(image.shape))
        self.assertEqual(tuple(logits.shape), (2, 12))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, message)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model)
            loaded, device = load_checkpoint(path, "cpu")
            cover = Image.new("RGB", (32, 32), color=(120, 150, 180))
            bits = [0, 1] * 6
            marked = encode_image(loaded, cover, bits, device)
            predicted, scores = decode_image(loaded, marked, device)
            self.assertEqual(marked.size, cover.size)
            self.assertEqual(len(predicted), 12)
            self.assertEqual(len(scores), 12)

    def test_stegastamp_reference_topology_at_400(self):
        # The released StegaStamp graph is defined at 400x400: the 50x50x3
        # message map is upsampled exactly 8x, and the STN/dense decoder
        # flatten dimensions are tied to this reference resolution.
        model = MarkProbe(message_bits=8, image_size=400)
        self.assertEqual(model.encoder.secret_dense.out_features, 7500)
        self.assertEqual(model.encoder.conv1.conv.conv.in_channels, 6)
        self.assertEqual(model.encoder.conv5.conv.conv.out_channels, 256)
        self.assertEqual(model.encoder.conv9.conv.conv.in_channels, 70)
        self.assertEqual(model.encoder.residual.conv.out_channels, 3)
        self.assertEqual(model.decoder.fc1.in_features, 128 * 13 * 13)
        self.assertEqual(model.decoder.fc1.out_features, 512)
        self.assertEqual(model.decoder.fc2.out_features, 8)

        image = torch.rand(1, 3, 400, 400)
        message = torch.randint(0, 2, (1, 8)).float()
        encoded, logits = model(image, message)
        self.assertEqual(tuple(encoded.shape), (1, 3, 400, 400))
        self.assertEqual(tuple(logits.shape), (1, 8))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, message)
        loss.backward()
        self.assertIsNotNone(model.encoder.secret_dense.weight.grad)
        self.assertIsNotNone(model.decoder.stn.fc.weight.grad)
        # The released source computes conv10 but sends conv9 to the residual head.
        self.assertIsNone(model.encoder.conv10.conv.conv.weight.grad)

    def test_yuv_loss_without_edge_gain_matches_reference_weighting(self):
        cover = torch.rand(2, 3, 12, 12)
        encoded = (cover + torch.randn_like(cover) * 0.01).clamp(0.0, 1.0)
        edge = _edge_falloff(12, 12, 0.0, cover.device, cover.dtype)
        self.assertTrue(torch.equal(edge, torch.zeros_like(edge)))
        scales = (1.0, 100.0, 100.0)
        actual = _stegastamp_yuv_loss(encoded, cover, scales, edge_gain=0.0)
        delta = _rgb_yuv(encoded) - _rgb_yuv(cover)
        expected = (delta.square().mean(dim=(0, 2, 3)) * torch.tensor(scales)).sum()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
    def test_stn_identity_and_full_message_projection_gradient(self):
        torch.manual_seed(4)
        model = MarkProbe(message_bits=12, use_stn=True, image_size=32)
        image = torch.rand(2, 3, 32, 32)
        corrected = model.decoder.stn(image)
        self.assertTrue(torch.allclose(corrected, image, atol=1e-5, rtol=1e-5))

        message = torch.randint(0, 2, (2, 12)).float()
        encoded, logits = model(image, message)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, message)
        loss.backward()
        grad = model.encoder.secret_dense.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertEqual(tuple(encoded.shape), tuple(image.shape))

    def test_jpeg_attack_and_png_quantization_are_differentiable(self):
        image = torch.rand(2, 3, 31, 37, requires_grad=True)
        jpeg = jpeg_compress_decompress(image, quality=72)
        self.assertEqual(tuple(jpeg.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(jpeg).all())
        jpeg.mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0.0)

        source = torch.tensor([0.5000], requires_grad=True)
        quantized = quantize_8bit_ste(source)
        quantized.sum().backward()
        self.assertEqual(float(source.grad.item()), 1.0)

    def test_projective_warp_uses_forward_source_to_destination_matrix(self):
        image = torch.zeros(1, 1, 5, 5)
        image[0, 0, 2, 1] = 1.0
        forward = torch.eye(3).unsqueeze(0)
        forward[:, 0, 2] = 1.0  # source pixel moves one position to the right
        warped = _warp_projective(image, forward)
        self.assertEqual(float(warped[0, 0, 2, 2]), 1.0)
        self.assertEqual(float(warped.sum()), 1.0)

    def test_stegastamp_training_transforms_and_geometry_have_gradients(self):
        torch.manual_seed(3)
        model = MarkProbe(message_bits=8, image_size=32)
        covers = torch.rand(2, 3, 32, 32, requires_grad=True)
        messages = torch.randint(0, 2, (2, 8)).float()
        encoded, warped, encoded_warped = _reference_geometry_embed(
            model, covers, messages, step=0, geometry_ramp_steps=100)
        self.assertEqual(tuple(encoded.shape), tuple(covers.shape))
        self.assertTrue(torch.isfinite(encoded).all())
        logits = model.decode(stegastamp_transform(encoded, step=10, use_jpeg=False))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, messages)
        loss = loss + _stegastamp_yuv_loss(encoded, covers)
        loss.backward()
        self.assertTrue(torch.isfinite(covers.grad).all())
        self.assertGreater(float(model.encoder.secret_dense.weight.grad.abs().sum()), 0.0)
        self.assertEqual(_linear_ramp(5, 10), 0.5)

    def test_all_reference_border_modes_are_supported(self):
        class ConstantEncoder(torch.nn.Module):
            def forward(self, image, message):
                return torch.full_like(image, 0.01)

        class EncoderOnly:
            encoder = ConstantEncoder()

        covers = torch.rand(2, 3, 16, 16)
        messages = torch.randint(0, 2, (2, 8)).float()
        for border in ("no_edge", "black", "random", "randomrgb", "image", "white"):
            torch.manual_seed(23)
            with patch("markprobe_core.engine.random.random", return_value=0.9):
                encoded, warped_cover, encoded_warped = _reference_geometry_embed(
                    EncoderOnly(), covers, messages, step=10,
                    geometry_ramp_steps=10, border=border,
                )
            self.assertEqual(tuple(encoded.shape), tuple(covers.shape))
            self.assertEqual(tuple(warped_cover.shape), tuple(covers.shape))
            self.assertEqual(tuple(encoded_warped.shape), tuple(covers.shape))
            self.assertTrue(torch.isfinite(encoded).all())

    def test_legacy_checkpoint_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            torch.save({"format": "markprobe-core-v1"}, path)
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
                load_checkpoint(path, "cpu")

    def test_v4_checkpoint_remains_loadable_for_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v4.pt"
            model = MarkProbe(message_bits=8, image_size=16)
            save_checkpoint(path, model)
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["format"] = "markprobe-core-v4-stegastamp-training"
            torch.save(state, path)
            loaded, device = load_checkpoint(path, "cpu")
            self.assertEqual(device.type, "cpu")
            self.assertEqual(loaded.message_bits, 8)
    def test_discriminator_updates_after_secret_only_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(6):
                Image.new("RGB", (24, 24), color=(35 * index, 90, 150)).save(
                    data_dir / "{:02d}.png".format(index)
                )

            discriminator = _StegaStampDiscriminator()
            initial = {key: value.detach().clone() for key, value in discriminator.state_dict().items()}
            output = root / "gan.pt"
            with patch("markprobe_core.engine._StegaStampDiscriminator", return_value=discriminator):
                train_model(
                    data_dir, output, image_size=16, message_bits=8, width=32,
                    epochs=1, batch_size=2, val_fraction=0.2, device="cpu",
                    workers=0, progress=False, seed=11, lpips_weight=0.0,
                    use_gan=True, no_im_loss_steps=1,
                )

            final = torch.load(output, map_location="cpu", weights_only=False)[
                "training_state"]["discriminator"]
            self.assertTrue(any(not torch.equal(initial[key], final[key]) for key in initial))

    def test_no_edge_border_training_with_gan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(3):
                Image.new("RGB", (20, 20), color=(50 * index, 90, 140)).save(
                    data_dir / "{:02d}.png".format(index)
                )
            train_model(
                data_dir, root / "no-edge.pt", image_size=16, message_bits=8,
                width=32, epochs=1, max_steps=1, batch_size=2,
                val_fraction=0.0, device="cpu", workers=0, progress=False,
                seed=19, lpips_weight=0.0, use_gan=True, no_im_loss_steps=0,
                border="no_edge",
            )
    def test_max_steps_stops_at_requested_optimizer_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (24, 24), color=(30 * index, 80, 140)).save(
                    data_dir / "{:02d}.png".format(index)
                )
            output = root / "steps.pt"
            train_model(
                data_dir, output, image_size=16, message_bits=8, width=32,
                epochs=10, max_steps=2, batch_size=2, val_fraction=0.0,
                device="cpu", workers=0, progress=False, seed=17,
                lpips_weight=0.0, use_gan=False, no_im_loss_steps=0,
            )
            state = torch.load(root / "steps.last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["global_step"], 2)

    def test_resume_training_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (24, 24), color=(40 * index, 80, 160)).save(
                    data_dir / "{:02d}.png".format(index)
                )

            output = root / "resume.pt"
            train_model(
                data_dir, output, image_size=16, message_bits=8, width=32,
                epochs=1, batch_size=2, val_fraction=0.25, device="cpu",
                workers=0, progress=False, seed=7, lpips_weight=0.0, use_gan=False, no_im_loss_steps=0,
            )
            latest = root / "resume.last.pt"
            self.assertTrue(latest.is_file())
            first_state = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(first_state["epoch"], 1)
            # train.py performs floor(3 / 2) + 1 full batches for this split.
            self.assertEqual(first_state["global_step"], 2)

            train_model(
                data_dir, output, image_size=16, message_bits=8, width=32,
                epochs=2, batch_size=2, val_fraction=0.25, device="cpu",
                workers=0, progress=False, seed=7, resume=latest, lpips_weight=0.0, use_gan=False, no_im_loss_steps=0,
            )
            second_state = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(second_state["epoch"], 2)
            self.assertTrue(output.is_file())

    def test_finetune_from_weights_resets_training_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (24, 24), color=(40 * index, 80, 160)).save(
                    data_dir / "{:02d}.png".format(index)
                )

            base = root / "base.pt"
            train_model(
                data_dir, base, image_size=16, message_bits=8, width=32,
                epochs=1, batch_size=2, val_fraction=0.25, device="cpu",
                workers=0, progress=False, seed=7, lpips_weight=0.0, use_gan=False,
                no_im_loss_steps=0,
            )
            source = root / "base.last.pt"
            finetuned = root / "finetuned.pt"
            run_finetune(
                data_dir, source, finetuned, device="cpu",
                settings={
                    "epochs": 1, "batch_size": 2, "val_fraction": 0.25,
                    "seed": 11, "lpips_weight": 0.0, "use_gan": False,
                    "no_im_loss_steps": 0, "crop_scale_min": 0.80,
                    "crop_scale_max": 0.95, "progress": False,
                },
            )
            state = torch.load(root / "finetuned.last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["global_step"], 2)
            self.assertEqual(state["training_state"]["train_config"]["crop_scale_min"], 0.80)

    def test_finetune_settings_reject_architecture_overrides(self):
        settings = merge_finetune_settings({"lr": 2e-5, "crop_scale_min": 0.75})
        self.assertEqual(settings["lr"], 2e-5)
        self.assertEqual(settings["crop_scale_min"], 0.75)
        with self.assertRaisesRegex(ValueError, "architecture"):
            merge_finetune_settings({"image_size": 256})

    def test_crop_resize_attack_preserves_shape_and_gradients(self):
        image = torch.rand(2, 3, 24, 24, requires_grad=True)
        attacked = crop_resize_attack(image, 0.7, 0.9)
        self.assertEqual(tuple(attacked.shape), tuple(image.shape))
        attacked.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_dual_branch_adaptive_ramp_validation_matrix_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (20, 20), color=(30 + index * 30, 80, 140)).save(
                    data_dir / "{:02d}.png".format(index)
                )
            output = root / "extended.pt"
            train_model(
                data_dir, output, image_size=16, message_bits=8, width=32,
                epochs=1, max_steps=1, batch_size=2, val_fraction=0.25,
                validation_samples=1, device="cpu", workers=0, progress=False,
                seed=31, lpips_weight=0.0, use_gan=False, no_im_loss_steps=0,
                dual_branch=True, adaptive_image_loss=True,
                adaptive_msg_threshold=100.0, adaptive_ber_threshold=1.0,
                adaptive_patience=1, adaptive_ramp_steps=4, gradient_log_interval=1,
            )
            state = torch.load(root / "extended.last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["format"], "markprobe-core-v6-robust-training")
            train_state = state["training_state"]
            self.assertEqual(train_state["adaptive_start_step"], 1)
            self.assertTrue(train_state["train_config"]["dual_branch"])
            record = json.loads((root / "extended.metrics.jsonl").read_text(encoding="utf-8"))
            scenarios = record["validation"]["scenarios"]
            self.assertEqual(set(scenarios), {"clean_png", "photo", "crop", "crop_photo", "jpeg75"})
            quality = record["validation"]["image_quality"]
            self.assertIn("ssim", quality)
            self.assertIn("mae", quality)
            self.assertIn("p95_abs_delta", quality)
            self.assertIn("lpips", quality)
            self.assertIn("lpips", quality["clean_png"])
            self.assertIn("lpips", quality["geometry_png"])
            self.assertGreater(record["train"]["gradient_norms"]["total"], 0.0)
            self.assertTrue(record["train"]["gradient_norms"]["finite"])
            self.assertEqual(record["epoch_inference"]["sample_count"], 1)
            sample_dir = root / "extended.epochs" / "epoch_0001" / "sample_0000"
            self.assertTrue((sample_dir / "cover.png").is_file())
            self.assertTrue((sample_dir / "encoded.png").is_file())
            self.assertTrue((sample_dir / "residual.png").is_file())
            self.assertTrue((sample_dir / "decoded.json").is_file())
            decoded = json.loads((sample_dir / "decoded.json").read_text(encoding="utf-8"))
            self.assertIn("residual_stats", decoded)
            self.assertIn("p95_abs", decoded["residual_stats"])
            self.assertTrue((root / "extended.curves.json").is_file())
            self.assertTrue((root / "extended.curves.csv").is_file())

    def test_adaptive_image_loss_requires_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "images"
            data_dir.mkdir()
            Image.new("RGB", (20, 20), color=(80, 100, 120)).save(data_dir / "one.png")
            with self.assertRaisesRegex(ValueError, "requires a held-out validation split"):
                train_model(data_dir, root / "bad.pt", image_size=16, message_bits=8,
                            epochs=1, max_steps=1, batch_size=1, val_fraction=0.0,
                            device="cpu", progress=False, lpips_weight=0.0, use_gan=False,
                            adaptive_image_loss=True)


if __name__ == "__main__":
    unittest.main()


