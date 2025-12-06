import argparse
import json
import os
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
    parser.add_argument("--epsilon", type=float, default=2.0, help="Target epsilon (for reporting and C-DP calibration).")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for DP guarantees.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.15)
    parser.add_argument("--l2_norm_clip", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=1.1,
                        help="Gaussian noise multiplier for DP-SGD; you can tune this to hit a desired epsilon.")
    parser.add_argument("--cdp_k", type=float, default=0.5, help="Composite DP parameter k (bump height).")
    parser.add_argument("--cdp_m", type=float, default=0.4, help="Composite DP parameter m (bump width).")
    parser.add_argument("--cdp_y", type=float, default=0.3, help="Composite DP base density floor y.")
    parser.add_argument("--cdp_index", type=int, default=1, choices=[1,2,3,4,5,6], 
                        help="Composite DP perturbation function index (1-6).")
    parser.add_argument("--results_dir", type=str, default="results")
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
    return (x_train, y_train, x_val, y_val, x_test, y_test, meta, model)


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size: int):
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000)
        .batch(batch_size, drop_remainder=True)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size, drop_remainder=True)
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

    loss_fn = losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )
    train_acc = metrics.SparseCategoricalAccuracy()
    val_acc = metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_acc])

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "epsilon_per_epoch": []}

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
    cdp_k: float,
    cdp_m: float,
    cdp_y: float,
    cdp_index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train with CompositeDP noise added to clipped gradients in a custom training loop.
    Uses the provided k, m, y parameters directly from CompositeDP paper.
    """
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = metrics.SparseCategoricalAccuracy()
    val_acc_metric = metrics.SparseCategoricalAccuracy()

    # Sensitivity is the L2 norm clip value
    sensitivity = l2_norm_clip
    # Lower bound for the query function
    lower_bound = 0.0
    
    # Store calibration info
    calibration_info = {
        "k": float(cdp_k),
        "m": float(cdp_m),
        "y": float(cdp_y),
        "index": int(cdp_index),
        "sensitivity": float(sensitivity),
        "lower_bound": float(lower_bound),
        "epsilon": float(epsilon),
    }
    
    print(f"[C-DP] Using parameters: k={cdp_k}, m={cdp_m}, y={cdp_y}, index={cdp_index}")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        print(f"\n[C-DP] Starting epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)
                mean_loss = tf.reduce_mean(loss_value)

            gradients = tape.gradient(mean_loss, model.trainable_variables)

            # Clip gradients to have L2 norm at most l2_norm_clip
            clipped_gradients = []
            global_norm = tf.linalg.global_norm(gradients)
            clip_ratio = l2_norm_clip / (global_norm + 1e-12)
            if clip_ratio < 1.0:
                gradients = [g * clip_ratio for g in gradients]

            # Add CompositeDP noise
            for g in gradients:
                g_np = g.numpy()
                noisy_g = add_composite_dp_noise_to_gradient(
                    gradient_array=g_np,
                    epsilon=epsilon,
                    sensitivity=sensitivity,
                    lower_bound=lower_bound,
                    k=cdp_k,
                    m=cdp_m,
                    y=cdp_y,
                    index=cdp_index,
                )
                clipped_gradients.append(tf.convert_to_tensor(noisy_g, dtype=g.dtype))

            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
            train_acc_metric.update_state(y_batch, model(x_batch, training=False))
            epoch_losses.append(float(mean_loss.numpy()))

        # End of epoch: compute training loss and validation metrics
        train_loss = float(np.mean(epoch_losses))
        train_acc = float(train_acc_metric.result().numpy())
        train_acc_metric.reset_states()

        # Compute validation metrics
        val_losses = []
        for x_val_batch, y_val_batch in val_ds:
            val_logits = model(x_val_batch, training=False)
            val_loss_value = loss_fn(y_val_batch, val_logits)
            val_losses.append(float(tf.reduce_mean(val_loss_value).numpy()))
            val_acc_metric.update_state(y_val_batch, val_logits)

        val_loss = float(np.mean(val_losses))
        val_acc = float(val_acc_metric.result().numpy())
        val_acc_metric.reset_states()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"[C-DP] Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
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
        "train_loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
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
        history, calibration_info = train_with_composite_dp(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2_norm_clip=args.l2_norm_clip,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            cdp_k=args.cdp_k,
            cdp_m=args.cdp_m,
            cdp_y=args.cdp_y,
            cdp_index=args.cdp_index,
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
        reduction=tf.keras.losses.Reduction.NONE
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

    # Save metrics and model
    save_dir = Path(args.results_dir) / args.dataset / args.mechanism / time.strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    model_path = save_dir / "model.keras"

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    model.save(model_path)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mechanism: {args.mechanism}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Membership inference AUC: {mi_auc:.4f}")
    if args.mechanism == "dpsgd":
        print(f"Final epsilon estimate: {results.get('epsilon_estimate'):.3f}")
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved model to:   {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()