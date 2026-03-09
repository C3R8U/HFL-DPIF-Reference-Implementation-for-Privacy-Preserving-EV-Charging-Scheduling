import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-8
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p1 = float(np.mean(y_true == 1))
    p0 = 1.0 - p1
    baseline = max(p0, p1)
    return {
        "accuracy": float(acc),
        "balanced_gain_over_majority": float(acc - baseline),
        "bce_loss": binary_cross_entropy(y_true, y_prob),
    }


def flatten_update(update: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([update["w"].ravel(), np.array([update["b"]], dtype=float)])


# ============================================================
# Data generation aligned with the uploaded station dataset
# ============================================================

def build_ev_sessions(
    station_df: pd.DataFrame,
    vehicles_per_station: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []

    location_multiplier = {
        "China": 1.10,
        "Norway": 1.18,
        "India": 0.92,
        "United States": 1.06,
        "Germany": 1.12,
        "France": 1.08,
        "UK": 1.09,
        "Japan": 1.07,
        "South Korea": 1.11,
        "Canada": 1.05,
    }
    charging_duration_bias = {
        "Level 1": 1.30,
        "Level 2": 1.00,
        "DC Fast": 0.55,
    }
    station_bias = {
        "Public": 1.00,
        "Private": 0.88,
        "Tesla Supercharger": 1.22,
    }

    df = station_df.copy()
    df["charging_type"] = df["charging_type"].replace({"Level 3": "DC Fast"})

    for _, row in df.iterrows():
        sid = int(row["station_id"])
        loc = str(row["location"])
        ctype = str(row["charging_type"])
        stype = str(row["station_type"])
        power_kw = float(row["power_output_kw"])
        adoption = float(row["ev_adoption_rate_per_1000"])
        gas_dist = float(row["nearby_gas_station_distance_km"])
        install_year = int(row["installation_year"])

        for vehicle_idx in range(vehicles_per_station):
            # Non-IID behavior: each station gets a shifted local distribution.
            local_shift = (sid % 7 - 3) * 0.09
            regional = location_multiplier.get(loc, 1.0)
            charging_factor = charging_duration_bias.get(ctype, 1.0)
            station_factor = station_bias.get(stype, 1.0)

            arrival_hour = int(np.clip(rng.normal(18 if station_factor >= 1 else 9, 4), 0, 23))
            weekday = int(rng.integers(0, 7) < 5)
            initial_soc = float(np.clip(rng.beta(2.5, 3.0), 0.05, 0.95))
            target_soc = float(np.clip(initial_soc + rng.uniform(0.10, 0.60), initial_soc + 0.05, 0.98))
            energy_need_kwh = float(np.clip((target_soc - initial_soc) * rng.uniform(45, 82), 2, 70))
            dwell_time_hr = float(np.clip((energy_need_kwh / max(power_kw, 7.0)) * charging_factor * rng.uniform(0.8, 1.3), 0.25, 10.0))
            queue_pressure = float(np.clip(
                0.3 * regional + 0.002 * adoption + 0.015 * max(arrival_hour - 16, 0) + local_shift + rng.normal(0, 0.08),
                0.0,
                2.5,
            ))
            tariff_level = float(np.clip(0.7 + 0.04 * arrival_hour + 0.15 * (1 - weekday) + rng.normal(0, 0.05), 0.5, 2.2))
            renewable_variability = float(np.clip(rng.beta(2, 5) + 0.15 * (arrival_hour in [17, 18, 19, 20]), 0.0, 1.2))
            flexibility_score = float(np.clip((1.0 - initial_soc) * 0.5 + dwell_time_hr / 10.0 + rng.normal(0, 0.05), 0.0, 1.5))

            peak_logit = (
                -0.8
                + 0.010 * power_kw
                + 0.003 * adoption
                + 0.42 * queue_pressure
                + 0.18 * tariff_level
                + 0.35 * renewable_variability
                + 0.20 * flexibility_score
                + 0.12 * weekday
                + 0.08 * gas_dist
                + 0.03 * max(install_year - 2018, 0)
                + local_shift
            )
            peak_prob = float(sigmoid(np.array([peak_logit]))[0])
            peak_demand = int(rng.random() < peak_prob)

            schedule_target = float(
                5.0
                + 0.28 * energy_need_kwh
                + 0.10 * power_kw
                + 0.85 * queue_pressure
                + 0.40 * tariff_level
                + 0.30 * renewable_variability
                - 0.65 * flexibility_score
                + rng.normal(0, 0.75)
            )

            records.append(
                {
                    "station_id": sid,
                    "vehicle_id": f"{sid}_{vehicle_idx}",
                    "location": loc,
                    "charging_type": ctype,
                    "station_type": stype,
                    "power_output_kw": power_kw,
                    "nearby_gas_station_distance_km": gas_dist,
                    "ev_adoption_rate_per_1000": adoption,
                    "popular_ev_models": str(row["popular_ev_models"]),
                    "installation_year": install_year,
                    "arrival_hour": arrival_hour,
                    "is_weekday": weekday,
                    "initial_soc": initial_soc,
                    "target_soc": target_soc,
                    "energy_need_kwh": energy_need_kwh,
                    "dwell_time_hr": dwell_time_hr,
                    "queue_pressure": queue_pressure,
                    "tariff_level": tariff_level,
                    "renewable_variability": renewable_variability,
                    "flexibility_score": flexibility_score,
                    "peak_demand": peak_demand,
                    "schedule_target": schedule_target,
                }
            )

    return pd.DataFrame(records)


def split_non_iid_by_station(session_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    station_groups = {}
    for sid, group in session_df.groupby("station_id"):
        station_groups[int(sid)] = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return station_groups


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(feature_df: pd.DataFrame) -> Pipeline:
    numeric_cols = [
        "power_output_kw",
        "nearby_gas_station_distance_km",
        "ev_adoption_rate_per_1000",
        "installation_year",
        "arrival_hour",
        "is_weekday",
        "initial_soc",
        "target_soc",
        "energy_need_kwh",
        "dwell_time_hr",
        "queue_pressure",
        "tariff_level",
        "renewable_variability",
        "flexibility_score",
    ]
    categorical_cols = [
        "location",
        "charging_type",
        "station_type",
        "popular_ev_models",
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    pipeline = Pipeline([("preprocessor", transformer)])
    pipeline.fit(feature_df)
    return pipeline


def transform_features(preprocessor: Pipeline, df: pd.DataFrame) -> np.ndarray:
    arr = preprocessor.transform(df)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float64)


# ============================================================
# Logistic model for peak-demand prediction
# ============================================================
@dataclass
class LogisticModel:
    w: np.ndarray
    b: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.w + self.b)

    def copy(self) -> "LogisticModel":
        return LogisticModel(self.w.copy(), float(self.b))



def init_model(input_dim: int, seed: int = 42) -> LogisticModel:
    rng = np.random.default_rng(seed)
    return LogisticModel(w=rng.normal(0, 0.01, size=input_dim), b=0.0)



def local_train_with_dp(
    global_model: LogisticModel,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    local_epochs: int,
    clip_norm: float,
    epsilon: float,
    delta: float,
    gamma: float,
    tau: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    model = global_model.copy()
    losses = []

    for _ in range(local_epochs):
        prob = model.predict_proba(x)
        error = prob - y
        grad_w = x.T @ error / len(x)
        grad_b = float(np.mean(error))

        grad_norm = float(np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2))
        clip_coef = min(1.0, clip_norm / (grad_norm + 1e-12))
        grad_w = grad_w * clip_coef
        grad_b = grad_b * clip_coef

        adaptive_sigma = (
            math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-6)
        ) * (1.0 + gamma / (np.linalg.norm(grad_w) + tau))

        noise_w = rng.normal(0.0, adaptive_sigma * clip_norm, size=grad_w.shape)
        noise_b = float(rng.normal(0.0, adaptive_sigma * clip_norm))

        grad_w_dp = grad_w + noise_w / len(x)
        grad_b_dp = grad_b + noise_b / len(x)

        model.w -= learning_rate * grad_w_dp
        model.b -= learning_rate * grad_b_dp
        losses.append(binary_cross_entropy(y, model.predict_proba(x)))

    update = {"w": model.w - global_model.w, "b": np.array(model.b - global_model.b)}
    metrics = {
        "local_loss": float(np.mean(losses)),
        "epsilon": float(epsilon),
        "grad_norm": float(grad_norm),
        "sigma": float(adaptive_sigma),
    }
    return update, metrics


# ============================================================
# DPBA: dynamic privacy budget allocation
# ============================================================

def compute_dynamic_epsilon(
    station_frame: pd.DataFrame,
    epsilon_min: float,
    epsilon_max: float,
    omega_urg: float,
    omega_pri: float,
    w_loc: float = 0.6,
    w_id: float = 0.5,
    w_pay: float = 0.4,
    w_sta: float = 0.3,
) -> float:
    # Proxy sensitive counts for a reproducible public implementation.
    n_loc = 1
    n_id = 1
    n_pay = 1 if station_frame["station_type"].iloc[0] == "Private" else 0
    n_sta = 1

    phi = w_loc * n_loc + w_id * n_id + w_pay * n_pay + w_sta * n_sta
    phi_max = w_loc + w_id + w_pay + w_sta
    phi_norm = phi / max(phi_max, 1e-9)

    delta_p = float(np.abs(station_frame["schedule_target"].mean() - station_frame["schedule_target"].median()))
    p_max = float(max(station_frame["schedule_target"].abs().max(), 1.0))
    response_window = float(np.clip(station_frame["dwell_time_hr"].mean(), 0.25, 10.0))
    psi = (delta_p / p_max) * (1.0 / response_window)
    psi_norm = min(psi / 2.0, 1.0)

    epsilon = epsilon_min + (epsilon_max - epsilon_min) * (
        omega_urg * psi_norm + omega_pri * (1.0 - phi_norm)
    )
    return float(np.clip(epsilon, epsilon_min, epsilon_max))


# ============================================================
# Mahalanobis clustering for Non-IID station grouping
# ============================================================

def compute_station_embeddings(station_groups: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, List[int]]:
    station_ids = []
    rows = []
    for sid, frame in station_groups.items():
        station_ids.append(sid)
        rows.append(
            [
                frame["power_output_kw"].mean(),
                frame["ev_adoption_rate_per_1000"].mean(),
                frame["queue_pressure"].mean(),
                frame["energy_need_kwh"].mean(),
                frame["dwell_time_hr"].mean(),
                frame["peak_demand"].mean(),
            ]
        )
    return np.asarray(rows, dtype=float), station_ids



def mahalanobis_whitening(x: np.ndarray) -> np.ndarray:
    cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    vals, vecs = np.linalg.eigh(inv_cov)
    whitening = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T
    return x @ whitening



def cluster_stations(station_groups: Dict[int, pd.DataFrame], k: int, seed: int = 42) -> Dict[int, List[int]]:
    station_features, station_ids = compute_station_embeddings(station_groups)
    whitened = mahalanobis_whitening(station_features)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(whitened)
    clusters: Dict[int, List[int]] = {}
    for sid, label in zip(station_ids, labels):
        clusters.setdefault(int(label), []).append(int(sid))
    return clusters


# ============================================================
# Hierarchical aggregation
# ============================================================

def aggregate_updates_weighted(updates: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
    total_n = sum(n for _, n in updates)
    if total_n == 0:
        raise ValueError("No updates were provided for aggregation.")

    agg_w = None
    agg_b = 0.0
    for upd, n in updates:
        weight = n / total_n
        if agg_w is None:
            agg_w = upd["w"] * weight
        else:
            agg_w += upd["w"] * weight
        agg_b += float(upd["b"]) * weight
    return {"w": agg_w, "b": np.array(agg_b)}



def hierarchical_aggregate(
    station_updates: Dict[int, Tuple[Dict[str, np.ndarray], int]],
    clusters: Dict[int, List[int]],
) -> Dict[str, np.ndarray]:
    cluster_updates = []
    for _, station_ids in clusters.items():
        local = [(station_updates[sid][0], station_updates[sid][1]) for sid in station_ids if sid in station_updates]
        if not local:
            continue
        cluster_update = aggregate_updates_weighted(local)
        cluster_n = sum(n for _, n in local)
        cluster_updates.append((cluster_update, cluster_n))
    return aggregate_updates_weighted(cluster_updates)


# ============================================================
# ADMM-style scheduling refinement
# ============================================================

def admm_refine_schedule(
    station_groups: Dict[int, pd.DataFrame],
    rho: float = 1.5,
    iterations: int = 20,
) -> Dict[str, float]:
    station_ids = list(station_groups.keys())
    m = len(station_ids)
    local_targets = np.array([station_groups[sid]["schedule_target"].mean() for sid in station_ids], dtype=float)
    x = local_targets.copy()
    z = np.mean(local_targets)
    u = np.zeros(m, dtype=float)

    for _ in range(iterations):
        x = (local_targets + rho * (z - u)) / (1.0 + rho)
        z = float(np.mean(x + u))
        u = u + x - z

    scheduling_error = float(np.mean(np.abs(x - local_targets) / np.maximum(np.abs(local_targets), 1.0)) * 100.0)
    return {
        "rho": float(rho),
        "consensus_schedule": float(z),
        "scheduling_error_percent": scheduling_error,
    }


# ============================================================
# SMSV blockchain-style verification simulation
# ============================================================

def simulate_smsv(
    n_nodes: int,
    malicious_ratio: float,
    num_shards: int = 64,
    bandwidth_mbps: float = 100.0,
    base_delay_ms: float = 50.0,
    delay_jitter_ms: float = 10.0,
    dropout_rate: float = 0.05,
    dynamic_hardening: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    malicious_nodes = max(1, int(round(n_nodes * malicious_ratio)))
    contaminated_shards = max(1, int(math.ceil(num_shards * min(malicious_ratio * 1.35, 1.0))))

    r = int(math.ceil(math.log2(num_shards)))
    abnormal_reject_rate = max(0.0, malicious_ratio - 1.0 / 3.0) * 0.30
    if dynamic_hardening and abnormal_reject_rate > 0.05:
        r = 2 * int(math.ceil(math.log2(num_shards)))

    miss_prob = (1.0 - contaminated_shards / num_shards) ** r
    detection_rate = (1.0 - miss_prob) * 100.0

    tree_depth = math.ceil(math.log2(max(n_nodes, 2)))
    latency = (
        base_delay_ms
        + delay_jitter_ms * tree_depth
        + 0.02 * n_nodes
        + 35.0 * malicious_ratio
        + 80.0 * dropout_rate
        + 0.18 * r
        + rng.normal(0, 1.2)
    )
    retransmission_overhead = max(0.0, abnormal_reject_rate) * 100.0 + malicious_nodes / max(n_nodes, 1) * 5.0

    return {
        "n_nodes": int(n_nodes),
        "malicious_ratio": float(malicious_ratio),
        "detected_malicious_rate_percent": float(np.clip(detection_rate, 0.0, 100.0)),
        "communication_latency_ms": float(max(latency, 1.0)),
        "retransmission_overhead_percent": float(retransmission_overhead),
        "num_verification_shards": int(r),
    }


# ============================================================
# Membership inference attack evaluation
# ============================================================

def membership_inference_attack(
    model: LogisticModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    train_loss = -(
        y_train * np.log(np.clip(model.predict_proba(x_train), 1e-8, 1 - 1e-8))
        + (1 - y_train) * np.log(np.clip(1 - model.predict_proba(x_train), 1e-8, 1 - 1e-8))
    )
    test_loss = -(
        y_test * np.log(np.clip(model.predict_proba(x_test), 1e-8, 1 - 1e-8))
        + (1 - y_test) * np.log(np.clip(1 - model.predict_proba(x_test), 1e-8, 1 - 1e-8))
    )

    threshold = float(np.median(train_loss))
    member_pred_train = (train_loss <= threshold).astype(int)
    member_pred_test = (test_loss <= threshold).astype(int)
    attack_acc = float((member_pred_train.mean() + (1.0 - member_pred_test).mean()) / 2.0)
    ppr = float((1.0 - max(attack_acc - 0.5, 0.0) / 0.5) * 100.0)
    return {
        "mia_attack_accuracy": attack_acc,
        "privacy_protection_rate_percent": ppr,
        "train_loss_mean": float(np.mean(train_loss)),
        "test_loss_mean": float(np.mean(test_loss)),
    }


# ============================================================
# Main training loop
# ============================================================

def run_hfl_dpif(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)

    station_df = pd.read_csv(args.data_path)
    if args.max_stations is not None:
        station_df = station_df.head(args.max_stations).copy()

    session_df = build_ev_sessions(
        station_df=station_df,
        vehicles_per_station=args.vehicles_per_station,
        seed=args.seed,
    )

    target_col = "peak_demand"
    feature_cols = [
        "location",
        "charging_type",
        "station_type",
        "power_output_kw",
        "nearby_gas_station_distance_km",
        "ev_adoption_rate_per_1000",
        "popular_ev_models",
        "installation_year",
        "arrival_hour",
        "is_weekday",
        "initial_soc",
        "target_soc",
        "energy_need_kwh",
        "dwell_time_hr",
        "queue_pressure",
        "tariff_level",
        "renewable_variability",
        "flexibility_score",
    ]

    train_df, test_df = train_test_split(
        session_df, test_size=0.2, stratify=session_df[target_col], random_state=args.seed
    )
    preprocessor = build_preprocessor(train_df[feature_cols])
    x_train_all = transform_features(preprocessor, train_df[feature_cols])
    y_train_all = train_df[target_col].to_numpy(dtype=np.float64)
    x_test = transform_features(preprocessor, test_df[feature_cols])
    y_test = test_df[target_col].to_numpy(dtype=np.float64)

    train_df = train_df.copy().reset_index(drop=True)
    train_df["row_index"] = np.arange(len(train_df))
    station_groups_df = split_non_iid_by_station(train_df)

    station_feature_groups: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, frame in station_groups_df.items():
        idx = frame["row_index"].to_numpy(dtype=int)
        station_feature_groups[sid] = (x_train_all[idx], y_train_all[idx])

    clusters = cluster_stations(station_groups_df, k=args.num_clusters, seed=args.seed)
    model = init_model(input_dim=x_train_all.shape[1], seed=args.seed)

    history = []
    epsilon_trace = {}
    for round_idx in range(args.global_rounds):
        station_updates: Dict[int, Tuple[Dict[str, np.ndarray], int]] = {}
        round_metrics = []

        for sid, frame in station_groups_df.items():
            x_local, y_local = station_feature_groups[sid]
            epsilon = compute_dynamic_epsilon(
                frame,
                epsilon_min=args.epsilon_min,
                epsilon_max=args.epsilon_max,
                omega_urg=args.omega_urg,
                omega_pri=args.omega_pri,
            )
            epsilon_trace[sid] = epsilon
            update, metrics = local_train_with_dp(
                global_model=model,
                x=x_local,
                y=y_local,
                learning_rate=args.learning_rate,
                local_epochs=args.local_epochs,
                clip_norm=args.clip_norm,
                epsilon=epsilon,
                delta=args.delta,
                gamma=args.dp_gamma,
                tau=args.dp_tau,
                seed=args.seed + round_idx * 1000 + sid,
            )
            station_updates[sid] = (update, len(x_local))
            round_metrics.append(metrics)

        global_update = hierarchical_aggregate(station_updates, clusters)
        model.w += global_update["w"]
        model.b += float(global_update["b"])

        test_prob = model.predict_proba(x_test)
        metrics = compute_classification_metrics(y_test, test_prob)
        metrics["round"] = round_idx + 1
        metrics["mean_local_epsilon"] = float(np.mean([m["epsilon"] for m in round_metrics]))
        metrics["mean_local_sigma"] = float(np.mean([m["sigma"] for m in round_metrics]))
        history.append(metrics)

    admm_summary = admm_refine_schedule(station_groups_df, rho=args.rho, iterations=args.admm_iterations)

    mia_summary = membership_inference_attack(model, x_train_all, y_train_all, x_test, y_test)

    smsv_summary = simulate_smsv(
        n_nodes=min(args.smsv_nodes, max(10, len(station_groups_df))),
        malicious_ratio=args.malicious_ratio,
        num_shards=args.smsv_num_shards,
        bandwidth_mbps=args.smsv_bandwidth_mbps,
        base_delay_ms=args.smsv_base_delay_ms,
        delay_jitter_ms=args.smsv_delay_jitter_ms,
        dropout_rate=args.smsv_dropout_rate,
        dynamic_hardening=True,
        seed=args.seed,
    )

    schedule_pred = np.array([station_groups_df[sid]["schedule_target"].mean() for sid in sorted(station_groups_df.keys())])
    schedule_consensus = np.full_like(schedule_pred, admm_summary["consensus_schedule"], dtype=float)
    mape = float(mean_absolute_percentage_error(schedule_pred, schedule_consensus) * 100.0)

    result = {
        "config": vars(args),
        "dataset": {
            "num_station_rows": int(len(station_df)),
            "num_simulated_ev_sessions": int(len(session_df)),
            "num_stations": int(session_df["station_id"].nunique()),
            "vehicles_per_station": int(args.vehicles_per_station),
            "num_clusters": int(args.num_clusters),
            "feature_dimension": int(x_train_all.shape[1]),
        },
        "training_history": history,
        "final_test_metrics": history[-1],
        "admm_summary": admm_summary,
        "membership_inference_summary": mia_summary,
        "smsv_summary": smsv_summary,
        "schedule_mape_percent": mape,
        "epsilon_statistics": {
            "min": float(np.min(list(epsilon_trace.values()))),
            "max": float(np.max(list(epsilon_trace.values()))),
            "mean": float(np.mean(list(epsilon_trace.values()))),
        },
        "cluster_sizes": {str(cid): len(sids) for cid, sids in clusters.items()},
    }
    return result


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runnable reference implementation of HFL-DPIF using the uploaded EV charging infrastructure dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the EV charging infrastructure CSV file.")
    parser.add_argument("--output-json", type=str, default="hfl_dpif_results.json", help="Path to save the result summary JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-stations", type=int, default=300, help="Maximum number of station rows to use from the dataset.")
    parser.add_argument("--vehicles-per-station", type=int, default=8, help="Number of simulated EV sessions generated per station.")
    parser.add_argument("--num-clusters", type=int, default=4, help="Number of Mahalanobis-aware station clusters for hierarchical aggregation.")
    parser.add_argument("--global-rounds", type=int, default=8, help="Number of global federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Number of local optimization epochs per station.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Local learning rate.")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping threshold for differential privacy.")
    parser.add_argument("--epsilon-min", type=float, default=0.5, help="Minimum dynamic privacy budget.")
    parser.add_argument("--epsilon-max", type=float, default=1.5, help="Maximum dynamic privacy budget.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Differential privacy delta parameter.")
    parser.add_argument("--dp-gamma", type=float, default=0.35, help="Adaptive noise scaling coefficient.")
    parser.add_argument("--dp-tau", type=float, default=1e-3, help="Stability constant for adaptive noise scaling.")
    parser.add_argument("--omega-urg", type=float, default=0.4, help="Urgency weight for dynamic privacy budget allocation.")
    parser.add_argument("--omega-pri", type=float, default=0.6, help="Privacy sensitivity weight for dynamic privacy budget allocation.")
    parser.add_argument("--rho", type=float, default=1.5, help="ADMM penalty parameter.")
    parser.add_argument("--admm-iterations", type=int, default=20, help="Number of ADMM consensus iterations.")
    parser.add_argument("--smsv-nodes", type=int, default=50, help="Number of nodes in the SMSV verification simulation.")
    parser.add_argument("--malicious-ratio", type=float, default=0.35, help="Malicious node ratio for the SMSV simulation.")
    parser.add_argument("--smsv-num-shards", type=int, default=64, help="Number of verification shards used by SMSV.")
    parser.add_argument("--smsv-bandwidth-mbps", type=float, default=100.0, help="Simulated bandwidth in Mbps.")
    parser.add_argument("--smsv-base-delay-ms", type=float, default=50.0, help="Base network delay in milliseconds.")
    parser.add_argument("--smsv-delay-jitter-ms", type=float, default=10.0, help="Delay jitter in milliseconds.")
    parser.add_argument("--smsv-dropout-rate", type=float, default=0.05, help="Node dropout ratio.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_hfl_dpif(args)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
