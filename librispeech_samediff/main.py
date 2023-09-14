import datetime
import json
from itertools import product
from pathlib import Path

import pkg_resources
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def read_data(subset):
    resource_package = __name__
    resource_path = f"data/{subset}.csv"
    data_path = pkg_resources.resource_filename(resource_package, resource_path)
    df = pd.read_csv(data_path, index_col=0)
    return df


@click.command()
@click.option(
    "--subset",
    type=click.Choice(["dev-clean", "dev-other", "test-clean", "test-other", "dev", "test", "all"]),
    help="Which LibriSpeech subset to use",
    required=True,
    prompt=True,
)
@click.option(
    "--feature-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Path to the directory containing the features",
    required=True,
    prompt=True,
)
@click.option(
    "--feature-rate",
    type=click.FLOAT,
    help="How many features per second",
    required=True,
    prompt=True,
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Directory to save logs results to",
    required=True,
    prompt=True,
)
@click.option(
    "--run-name",
    type=click.STRING,
    help="Name of subdirectory to save logs results to",
    default="None",
    prompt=True,
)
def main(subset, feature_dir, feature_rate, log_dir, run_name):

    # ============================ Create log dir ============================
    if run_name == "None":
        run_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = Path(log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "options.txt").write_text(
        f"subset: {subset}\nfeature_dir: {feature_dir}\nfeature_rate: {feature_rate}"
    )

    # ===================== Run for each subset chosen =====================
    if subset == "dev":
        subsets = ["dev-clean", "dev-other"]
    elif subset == "test":
        subsets = ["test-clean", "test-other"]
    elif subset == "all":
        subsets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    else:
        subsets = [subset]

    for subset in subsets:
        click.echo(click.style(f"\nRunning evaluation on {subset} subset of LibriSpeech...\n", fg="green", bold=True))

        # ========================== Load word data ==========================
        df = read_data(subset)

        # ========================= Extract features =========================
        click.echo("Extracting features...")
        words_features = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            _, _, filename, start, end = row
            matched_paths = Path(feature_dir).rglob(f"{filename}.npy")
            try:
                feats_path = next(matched_paths)
            except StopIteration:
                raise click.ClickException(f"Could not find file {filename}.npy in {feature_dir}")
            feats = np.load(feats_path)
            start = round(start * feature_rate)
            end = round(end * feature_rate)
            feats = feats[start:end]
            # normalize to unit vectors - this will turn euclidean DTW into cosine DTW (proportional)
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            words_features.append(feats.astype(np.float64))

        # ================== Compute pairwise DTW distances ==================
        click.echo("Computing pairwise DTW distances...")
        # split into blocks to reduce memory usage and show progress bar
        n = len(words_features)
        distance = np.full((n, n), np.inf)
        num_blocks = 14
        block_size = n // num_blocks
        remainder = n % num_blocks
        block_combinations = [(i, j) for i, j in product(range(num_blocks), repeat=2) if i <= j]
        for i, j in tqdm(block_combinations):
            # Calculate indices for block
            row_start = i * block_size + min(i, remainder)
            row_end = (i + 1) * block_size + min(i + 1, remainder)
            col_start = j * block_size + min(j, remainder)
            col_end = (j + 1) * block_size + min(j + 1, remainder)
            # Ensure row_end and col_end do not exceed matrix dimensions
            row_end = min(n, row_end)
            col_end = min(n, col_end)
            # Compute DTW distance for this block
            block_result = dtw_ndim.distance_matrix_fast(
                words_features, block=((row_start, row_end), (col_start, col_end))
            )
            # Update the result matrix with the block result
            distance[row_start:row_end, col_start:col_end] = block_result[row_start:row_end, col_start:col_end]
            # Since the matrix is symmetric, copy values to the lower triangle as well
            if i != j:
                distance[col_start:col_end, row_start:row_end] = block_result[row_start:row_end, col_start:col_end].T

        # ========================== Save distances ==========================
        np.save(save_dir / f"{subset}-distances.npy", distance)
        click.echo(f"Saved distance matrix to {save_dir / f'{subset}-distances.npy'}")

        click.echo("Computing precision and recall...")
        words = df.word.to_numpy()
        speaker_ids = df.speaker_id.to_numpy()

        # ================= Construct ground truth results ==================
        SW = words[:, np.newaxis] == words[np.newaxis, :]
        SP = speaker_ids[:, np.newaxis] == speaker_ids[np.newaxis, :]
        SWSP = np.logical_and(SW, SP)
        SWDP = np.logical_and(SW, ~SP)
        # DWSP = np.logical_and(~SW, SP)
        # DWDP = np.logical_and(~SW, ~SP)

        # ============== Calculate precision and recall curves ==============
        upper_tri = np.triu_indices(len(words), k=1)
        P_SW, R_SW, thresholds = precision_recall_curve(SW[upper_tri], -distance[upper_tri])
        _, R_SWSP, _ = precision_recall_curve(SWSP[upper_tri], -distance[upper_tri])
        _, R_SWDP, _ = precision_recall_curve(SWDP[upper_tri], -distance[upper_tri])

        # ==================== Calculate average precsion ====================
        AP_SW = np.abs(np.trapz(P_SW, R_SW))
        AP_SWSP = np.abs(np.trapz(P_SW, R_SWSP))
        AP_SWDP = np.abs(np.trapz(P_SW, R_SWDP))

        # =========== Calculate precision-recall break-even points ===========
        PRB_SW = P_SW[np.argmin(np.abs(R_SW - P_SW))]
        PRB_SWSP = P_SW[np.argmin(np.abs(R_SWSP - P_SW))]
        PRB_SWDP = P_SW[np.argmin(np.abs(R_SWDP - P_SW))]

        # ==================== Save results to JSON file ====================
        results = {
            "AP": {"SW": AP_SW, "SWSP": AP_SWSP, "SWDP": AP_SWDP},
            "PRB": {"SW": PRB_SW, "SWSP": PRB_SWSP, "SWDP": PRB_SWDP},
        }
        with open(save_dir / f"{subset}-results.json", "w") as f:
            json.dump(results, f, indent=4)
        click.echo(f"Saved results to {save_dir / f'{subset}-results.json'}")

        # =========================== Plot results ===========================
        # plot every 10th point to make the plot less dense
        plt.figure()
        plt.plot(R_SW[::10], P_SW[::10], label="SW")
        plt.plot(R_SWSP[::10], P_SW[::10], label="SWSP")
        plt.plot(R_SWDP[::10], P_SW[::10], label="SWDP")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(save_dir / f"{subset}-precision-recall.png", dpi=300)
        click.echo(f"Saved precision-recall plot to {save_dir / f'{subset}-precision-recall.png'}")

        # ========================== Print results ==========================
        click.echo("")
        click.echo(click.style(f"Average Precision:", fg="green"))
        click.echo(click.style(f"SW:\t{AP_SW:.4f}", fg="green"))
        click.echo(click.style(f"SWSP:\t{AP_SWSP:.4f}", fg="green"))
        click.echo(click.style(f"SWDP:\t{AP_SWDP:.4f}", fg="green"))
        click.echo("")
        click.echo(click.style(f"Precision Recall Break-Even Point:", fg="green"))
        click.echo(click.style(f"SW:\t{PRB_SW:.4f}", fg="green"))
        click.echo(click.style(f"SWSP:\t{PRB_SWSP:.4f}", fg="green"))
        click.echo(click.style(f"SWDP:\t{PRB_SWDP:.4f}", fg="green"))
        click.echo("")

    click.echo(click.style("\nFinished!", fg="green"))


if __name__ == "__main__":
    main()
