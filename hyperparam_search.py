#!/usr/bin/env python3

import os
import csv
import re
import subprocess
from itertools import product
from datetime import datetime
import random

N_EXPERIMENTS = 50
EPISODES = 20000

# ==========================
# HIPERPARÁMETROS A PROBAR
# ==========================

N_BINS = [20, 40]
LRS = [0.05, 0.1]
GAMMAS = [0.95, 0.99]
EPSILON_STARTS = [1.0]
EPSILON_ENDS = [0.01]
EPSILON_DECAYS = [0.9995, 0.9999]


# ==========================
# ARCHIVOS
# ==========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_FILE = f"mountaincar_gridsearch_{timestamp}.csv"

csv_headers = [
    "n_bins",
    "lr",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "mean_reward",
    "std_reward",
    "flag_success"
]

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)


best_reward = float("-inf")
best_config = None


current = 0

for current in range(1, N_EXPERIMENTS + 1):

    n_bins = random.choice([20, 30, 40, 50, 60])

    lr = random.choice([
        0.01,
        0.03,
        0.05,
        0.1,
        0.2
    ])

    gamma = random.uniform(
        0.95,
        0.999
    )

    eps_start = 1.0

    eps_end = 0.01

    eps_decay = random.uniform(
        0.995,
        0.99999
    )

    print("\n" + "=" * 80)
    print(f"Experimento {current}/{N_EXPERIMENTS}")
    print("=" * 80)

    print(
        f"bins={n_bins} "
        f"lr={lr:.5f} "
        f"gamma={gamma:.5f} "
        f"decay={eps_decay:.6f}"
    )


    current += 1

    env = os.environ.copy()

    env["MOUNTAIN_CAR_N_BINS"] = str(n_bins)
    env["MOUNTAIN_CAR_LR"] = str(lr)
    env["MOUNTAIN_CAR_GAMMA"] = str(gamma)
    env["MOUNTAIN_CAR_EPSILON_START"] = str(eps_start)
    env["MOUNTAIN_CAR_EPSILON_END"] = str(eps_end)
    env["MOUNTAIN_CAR_EPSILON_DECAY"] = str(eps_decay)

    #################################################
    # Limpiar entrenamiento previo
    #################################################

    subprocess.run(
        ["uv", "run", "mountaincar", "delete", "qlearning"],
        env=env,
        capture_output=True,
        text=True,
    )

    #################################################
    # Entrenar
    #################################################

    print("Entrenando...")

    train = subprocess.run(
        [
            "uv",
            "run",
            "mountaincar",
            "train",
            "qlearning",
            "--episodes",
            str(EPISODES),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    if train.returncode != 0:
        print("ERROR ENTRENANDO")
        print(train.stderr)
        continue

    #################################################
    # Evaluar
    #################################################

    print("Evaluando...")

    evaluate = subprocess.run(
        [
            "uv",
            "run",
            "mountaincar",
            "load",
            "qlearning",
            "--eval",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    output = evaluate.stdout

    print(output)

    #################################################
    # Parsear Mean reward
    #################################################

    reward_match = re.search(
        r"Mean reward:\s*(-?\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)",
        output,
    )

    flag_match = re.search(
        r"Reached the flag:\s*(\d+)\/(\d+)",
        output,
    )

    if not reward_match:
        print("No fue posible leer Mean reward")
        continue

    mean_reward = float(reward_match.group(1))
    std_reward = float(reward_match.group(2))

    if flag_match:
        flag_success = (
            f"{flag_match.group(1)}/{flag_match.group(2)}"
        )
    else:
        flag_success = "N/A"

    #################################################
    # Guardar CSV
    #################################################

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            n_bins,
            lr,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            mean_reward,
            std_reward,
            flag_success,
        ])

    #################################################
    # Mejor resultado
    #################################################

    if mean_reward > best_reward:
        best_reward = mean_reward

        best_config = {
            "n_bins": n_bins,
            "lr": lr,
            "gamma": gamma,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "reward": mean_reward,
            "std": std_reward,
            "flags": flag_success,
        }

#########################################################
# Resultado final
#########################################################

print("\n")
print("=" * 80)
print("MEJOR CONFIGURACIÓN")
print("=" * 80)

for k, v in best_config.items():
    print(f"{k}: {v}")

print(f"\nResultados almacenados en:\n{CSV_FILE}")