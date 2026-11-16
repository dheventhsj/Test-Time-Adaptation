import io, os, time, copy, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageOps
from model import TinyCNN, parameter_summary

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
WEIGHTS = ARTIFACTS / "tinycnn.pt"
DEVICE = torch.device("cpu")

class ExperimentEngine:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.history = []
        self._load_or_train()

    def _seed(self, seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def _dataset(self, seed=42):
        ds = load_digits()
        x = ds.images.astype(np.float32) / 16.0
        y = ds.target.astype(np.int64)
        x = torch.tensor(x).unsqueeze(1)
        x = torch.nn.functional.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, torch.tensor(y), test_size=0.25, random_state=seed,
            stratify=y.numpy()
        )
        return x_train, x_test, y_train, y_test

    def _train(self, epochs=8, learning_rate=0.003, batch_size=128, seed=42):
        self._seed(seed)
        x_train, x_test, y_train, y_test = self._dataset(seed)
        model = TinyCNN().to(DEVICE)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        history = []
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total_loss, correct, count = 0.0, 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                count += len(yb)
            history.append({"epoch": epoch + 1,
                            "loss": round(total_loss / count, 4),
                            "accuracy": round(100 * correct / count, 2)})
        elapsed = time.perf_counter() - started
        return model, x_test, y_test, history, elapsed

    def _load_or_train(self):
        self._seed()
        if WEIGHTS.exists():
            payload = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
            self.model = TinyCNN().to(DEVICE)
            self.model.load_state_dict(payload["state_dict"])
            self.history = payload.get("history", [])
            self.metadata = payload.get("metadata", {})
        else:
            model, _, _, history, elapsed = self._train()
            self.model = model
            self.history = history
            self.metadata = {"training_seconds": round(elapsed, 2), "seed": 42,
                             "epochs": 8, "learning_rate": 0.003, "batch_size": 128}
            torch.save({"state_dict": model.state_dict(), "history": history,
                        "metadata": self.metadata}, WEIGHTS)

    def retrain(self, epochs=8, learning_rate=0.003, batch_size=128, seed=42):
        if not 1 <= epochs <= 50: raise ValueError("Epochs must be between 1 and 50.")
        if not 1e-5 <= learning_rate <= 0.1: raise ValueError("Learning rate must be between 0.00001 and 0.1.")
        if not 16 <= batch_size <= 512: raise ValueError("Batch size must be between 16 and 512.")
        model, _, _, history, elapsed = self._train(epochs, learning_rate, batch_size, seed)
        self.model, self.history = model, history
        self.metadata = {"training_seconds": round(elapsed, 3), "seed": seed,
                         "epochs": epochs, "learning_rate": learning_rate,
                         "batch_size": batch_size}
        torch.save({"state_dict": model.state_dict(), "history": history,
                    "metadata": self.metadata}, WEIGHTS)
        return {"history": history, "training_seconds": round(elapsed, 3),
                "parameters": parameter_summary(model)}

    @staticmethod
    def _noise(x, sigma=0.30, seed=43):
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator) * sigma
        return torch.clamp(x + noise, 0, 1)

    @staticmethod
    def _entropy(logits):
        p = torch.softmax(logits, dim=1)
        return -(p * torch.log(p.clamp_min(1e-8))).sum(1)

    def _predict(self, model, x, batch_size=128):
        model.eval()
        preds, entropies = [], []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = model(x[start:start+batch_size])
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                entropies.extend(self._entropy(logits).cpu().numpy().tolist())
        return np.array(preds), np.array(entropies)

    def _adapt_tent(self, base, x, batch_size=128, lr=0.0001, entropy_threshold=None):
        model = copy.deepcopy(base)
        for p in model.parameters(): p.requires_grad_(False)
        bn_params = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()  # use current target batch statistics
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
                bn_params.extend([module.weight, module.bias])
        optimizer = torch.optim.Adam(bn_params, lr=lr)
        ent_log, updated_batches = [], 0
        start_time = time.perf_counter()
        for start in range(0, len(x), batch_size):
            xb = x[start:start+batch_size]
            optimizer.zero_grad()
            logits = model(xb)
            per_sample = self._entropy(logits)
            # Optional confidence gate: only lower-entropy samples contribute.
            if entropy_threshold is not None:
                mask = per_sample.detach() <= entropy_threshold
                if mask.any():
                    loss = per_sample[mask].mean()
                else:
                    ent_log.append(float(per_sample.mean().item()))
                    continue
            else:
                loss = per_sample.mean()
            ent_log.append(float(per_sample.mean().item()))
            loss.backward()
            optimizer.step()
            updated_batches += 1
        elapsed = time.perf_counter() - start_time
        return model, ent_log, elapsed, updated_batches

    def run_experiment(self, sigma=0.30, batch_size=128, tent_lr=0.0001,
                       seed=43, confidence_gate=False):
        if not 0 <= sigma <= 1: raise ValueError("Noise sigma must be between 0 and 1.")
        if not 8 <= batch_size <= 512: raise ValueError("Batch size must be between 8 and 512.")
        _, x_test, y_test = self._dataset(42)
        x_noisy = self._noise(x_test, sigma, seed)
        base = copy.deepcopy(self.model)
        base.eval()

        clean_pred, clean_ent = self._predict(base, x_test, batch_size)
        noisy_pred, noisy_ent = self._predict(base, x_noisy, batch_size)

        # BatchNorm-only ablation: target batch statistics, no gradient updates.
        bn_only = copy.deepcopy(base)
        for p in bn_only.parameters(): p.requires_grad_(False)
        for m in bn_only.modules():
            if isinstance(m, nn.BatchNorm2d): m.train()
        bn_pred, bn_ent = self._predict_batchnorm(bn_only, x_noisy, batch_size)

        gate = 1.5 if confidence_gate else None
        adapted, ent_steps, adapt_time, updates = self._adapt_tent(
            base, x_noisy, batch_size, tent_lr, gate)
        tent_pred, tent_ent = self._predict(adapted, x_noisy, batch_size)

        def metric(name, pred, ent):
            return {"name": name, "accuracy": round(float((pred == y_test.numpy()).mean()) * 100, 2),
                    "error_rate": round(float((pred != y_test.numpy()).mean()) * 100, 2),
                    "mean_entropy": round(float(np.mean(ent)), 4),
                    "correct": int((pred == y_test.numpy()).sum()), "total": len(y_test)}
        results = [
            metric("Clean source-only", clean_pred, clean_ent),
            metric("Noisy source-only", noisy_pred, noisy_ent),
            metric("BatchNorm statistics only", bn_pred, bn_ent),
            metric("Noisy + TENT", tent_pred, tent_ent),
        ]
        return {
            "results": results,
            "improvement_pp": round(results[3]["accuracy"] - results[1]["accuracy"], 2),
            "recovered_lost_accuracy_percent": round(
                100 * (results[3]["accuracy"] - results[1]["accuracy"]) /
                max(0.01, results[0]["accuracy"] - results[1]["accuracy"]), 2),
            "adaptation_seconds": round(adapt_time, 4),
            "entropy_by_batch": [round(v, 4) for v in ent_steps],
            "batch_updates": updates, "batch_size": batch_size,
            "noise_sigma": sigma, "tent_learning_rate": tent_lr,
            "parameters": parameter_summary(base),
            "confusion_matrix": confusion_matrix(y_test.numpy(), tent_pred, labels=list(range(10))).tolist(),
            "classification_report": classification_report(y_test.numpy(), tent_pred, labels=list(range(10)),
                                                           output_dict=True, zero_division=0),
            "predictions": [{"actual": int(a), "predicted": int(p)}
                            for a, p in zip(y_test.numpy()[:60], tent_pred[:60])],
            "note": "Target labels are used only after adaptation to calculate evaluation metrics."
        }

    def _predict_batchnorm(self, model, x, batch_size):
        # Each target batch is forwarded independently using its own batch statistics.
        preds, ents = [], []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = model(x[start:start+batch_size])
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                ents.extend(self._entropy(logits).cpu().numpy().tolist())
        return np.array(preds), np.array(ents)

    def predict_upload(self, raw_bytes):
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
        image = ImageOps.invert(image) if np.asarray(image).mean() > 127 else image
        image = ImageOps.fit(image, (32, 32))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        return {"prediction": int(np.argmax(probs)),
                "confidence": round(float(np.max(probs)) * 100, 2),
                "entropy": round(float(self._entropy(logits)[0]), 4),
                "probabilities": [round(float(p) * 100, 2) for p in probs]}

    def export_csv(self, experiment):
        import csv
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["method", "accuracy", "error_rate", "mean_entropy", "correct", "total"])
        writer.writeheader()
        for row in experiment["results"]:
            writer.writerow({"method": row["name"], **{k: row[k] for k in
                           ["accuracy", "error_rate", "mean_entropy", "correct", "total"]}})
        return out.getvalue()
