import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

from datasets import load_mnist, load_adult
from models import build_mnist_cnn, build_adult_mlp
from tensorflow_privacy_utils import build_dp_sgd_optimizer, compute_epsilon
from composite_dp_utils import (
    auto_calibrate_composite_parameters,
    add_composite_dp_noise_to_gradient,
)
from membership_inference import loss_threshold_membership_inference_auc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model with DP-SGD or Composite DP noise.")
    parser.add_argument("--dataset", choices=["mnist", "adult"], required=True)
    parser.add_argument("--mechanism", choices=["dpsgd", "cdp", "none"], default="dpsgd")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Target epsilon (for reporting and C-DP calibration).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta for DP guarantees (used in DP-SGD accounting).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.15)
    parser.add_argument("--l2_norm_clip", type=float, default=1.0)
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.1,
        help="Gaussian noise multiplier for DP-SGD; also used to define the target variance for C-DP matching.",
    )
    parser.add_argument("--cdp_L", type=float, default=1.0, help="Composite DP parameter L (support bound).")
    parser.add_argument("--cdp_m", type=float, default=0.5, help="(Unused) initial C-DP parameter m.")
    parser.add_argument("--cdp_y", type=float, default=0.05, help="(Unused) initial C-DP parameter y.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and model weights will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_dataset_and_model(dataset_name: str):
    if dataset_name == "mnist":
        x_train, y_train, x_val, y_val, x_test, y_test, meta = load_mnist()
        model = build_mnist_cnn(meta["input_shape"], meta["num_classes"])
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, meta = load_adult()
        model = build_adult_mlp(meta["input_shape"][0], meta["num_classes"])
    return x_train, y_train, x_val, y_val, x_test, y_test, meta, model


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size: int):
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, val_ds, test_ds


def train_with_dp_sgd(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_epochs: int,
    learning_rate: float,
    l2_norm_clip: float,
    noise_multiplier: float,
    batch_size: int,
    num_examples: int,
    delta: float,
) -> Dict[str, Any]:
    """
    Train using TensorFlow Privacy DPKerasSGDOptimizer and return metrics including epsilon estimates.
    """
    optimizer = build_dp_sgd_optimizer(
        learning_rate=learning_rate,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=batch_size,
    )
    # TF Privacy handles per-example losses internally when num_microbatches is set.
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc = metrics.SparseCategoricalAccuracy()
    val_acc = metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_acc])

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epsilon_per_epoch": [],
    }

    for epoch in range(num_epochs):
        print(f"\n[DP-SGD] Starting epoch {epoch + 1}/{num_epochs}")
        epoch_hist = model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1)
        train_loss = float(epoch_hist.history["loss"][-1])
        val_loss = float(epoch_hist.history["val_loss"][-1])
        val_accuracy = float(epoch_hist.history["val_sparse_categorical_accuracy"][-1])

        epsilon_estimate = compute_epsilon(
            num_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epoch + 1,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epsilon_per_epoch"].append(epsilon_estimate)

        print(
            f"[DP-SGD] Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, eps≈{epsilon_estimate:.3f}"
        )

    return history


def train_with_composite_dp(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_epochs: int,
    learning_rate: float,
    l2_norm_clip: float,
    batch_size: int,
    epsilon: float,
    cdp_L: float,
    cdp_m: float,
    cdp_y: float,
    target_variance: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train with CompositeDP noise added to clipped gradients in a custom training loop.

    We first run the enumeration-based parameter optimization from the CompositeDP
    library (via auto_calibrate_composite_parameters) to obtain (k, m, y) for the
    given epsilon and mechanism index. Then we empirically estimate the resulting
    variance and log it as calibration_info.
    """
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = metrics.SparseCategoricalAccuracy()
    val_acc_metric = metrics.SparseCategoricalAccuracy()

    # After clipping, the L2 norm of the gradient is at most l2_norm_clip.
    sensitivity = l2_norm_clip
    lower_bound = -cdp_L

    # Which CompositeDP instantiation to use: A1B1 corresponds to index=1.
    cdp_index = 1

    print(f"\n[C-DP] Running enumeration-based parameter optimization for ε={epsilon}, index={cdp_index} ...")
    calibration_info = auto_calibrate_composite_parameters(
        epsilon=epsilon,
        target_variance=target_variance,
        sensitivity=sensitivity,
        lower_bound=lower_bound,
        L=cdp_L,
        m=cdp_m,
        y=cdp_y,
        index=cdp_index,
        calibration_samples=3000,
    )
    best_k = calibration_info["k"]
    best_m = calibration_info["m"]
    best_y = calibration_info["y"]
    print(
        f"[C-DP] Optimization result: k={best_k:.4f}, m={best_m:.4f}, y={best_y:.4f}, "
        f"target_var={calibration_info['target_variance']:.4f}, "
        f"empirical_var={calibration_info['empirical_variance']:.4f}"
    )

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):
        print(f"\n[C-DP] Starting epoch {epoch + 1}/{num_epochs}")
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)

            gradients = tape.gradient(loss_value, model.trainable_variables)

            # Global L2 clipping to l2_norm_clip
            global_norm = tf.linalg.global_norm(gradients)
            clip_ratio = l2_norm_clip / (global_norm + 1e-12)
            if clip_ratio < 1.0:
                gradients = [g * clip_ratio for g in gradients]

            noisy_gradients = []
            for g in gradients:
                g_np = g.numpy()
                noisy_g_np = add_composite_dp_noise_to_gradient(
                    gradient_array=g_np,
                    epsilon=epsilon,
                    sensitivity=sensitivity,
                    lower_bound=lower_bound,
                    L=cdp_L,
                    m=best_m,
                    y=best_y,
                    k=best_k,
                    index=cdp_index,
                )
                noisy_gradients.append(tf.convert_to_tensor(noisy_g_np, dtype=g.dtype))

            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))
            train_acc_metric.update_state(y_batch, model(x_batch, training=False))

        # End-of-epoch metrics
        train_acc = float(train_acc_metric.result().numpy())
        train_acc_metric.reset_states()

        val_losses = []
        for x_val_batch, y_val_batch in val_ds:
            val_logits = model(x_val_batch, training=False)
            val_loss_value = loss_fn(y_val_batch, val_logits)
            val_losses.append(float(val_loss_value.numpy()))
            val_acc_metric.update_state(y_val_batch, val_logits)

        val_loss = float(np.mean(val_losses))
        val_acc = float(val_acc_metric.result().numpy())
        val_acc_metric.reset_states()

        history["train_loss"].append(float(loss_value.numpy()))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"[C-DP] Epoch {epoch + 1}: train_loss={float(loss_value.numpy()):.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    return history, calibration_info


def train_without_dp(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_epochs: int,
    learning_rate: float,
) -> Dict[str, Any]:
    """
    Non-private training baseline.
    """
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=1)
    return {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"],
    }


def main():
    args = parse_arguments()
    set_random_seeds(args.seed)

    x_train, y_train, x_val, y_val, x_test, y_test, meta, model = build_dataset_and_model(args.dataset)
    train_ds, val_ds, test_ds = make_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=args.batch_size
    )

    results: Dict[str, Any] = {
        "args": vars(args),
        "dataset_metadata": meta,
        "mechanism": args.mechanism,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.mechanism == "dpsgd":
        history = train_with_dp_sgd(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            batch_size=args.batch_size,
            num_examples=meta["train_size"],
            delta=args.delta,
        )
        results["training_history"] = history
        results["epsilon_estimate"] = history["epsilon_per_epoch"][-1] if history["epsilon_per_epoch"] else None

    elif args.mechanism == "cdp":
        gaussian_variance = args.noise_multiplier ** 2
        history, calibration_info = train_with_composite_dp(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2_norm_clip=args.l2_norm_clip,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            cdp_L=args.cdp_L,
            cdp_m=args.cdp_m,
            cdp_y=args.cdp_y,
            target_variance=gaussian_variance,
        )
        results["training_history"] = history
        results["composite_dp_calibration"] = calibration_info

    else:
        history = train_without_dp(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        results["training_history"] = history

    # Final evaluation + membership inference attack
    loss_fn_ce = losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    train_logits = model.predict(x_train, verbose=0)
    train_loss_values = loss_fn_ce(y_train, train_logits).numpy()
    train_predictions = np.argmax(train_logits, axis=1)
    train_accuracy = float(np.mean(train_predictions == y_train))

    test_logits = model.predict(x_test, verbose=0)
    test_loss_values = loss_fn_ce(y_test, test_logits).numpy()
    test_predictions = np.argmax(test_logits, axis=1)
    test_accuracy = float(np.mean(test_predictions == y_test))

    mi_auc = loss_threshold_membership_inference_auc(train_loss_values, test_loss_values)

    results["final_train_accuracy"] = train_accuracy
    results["final_test_accuracy"] = test_accuracy
    results["final_train_loss_mean"] = float(np.mean(train_loss_values))
    results["final_test_loss_mean"] = float(np.mean(test_loss_values))
    results["membership_inference_auc"] = mi_auc

    save_dir = Path(args.results_dir) / args.dataset / args.mechanism / time.strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    model_path = save_dir / "model.keras"

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    model.save(model_path)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved model to:   {model_path}")
    print(f"Membership inference AUC: {mi_auc:.4f}")
    if args.mechanism == "dpsgd":
        print(f"Final epsilon estimate (DP-SGD): {results.get('epsilon_estimate')}")


if __name__ == "__main__":
    main()
