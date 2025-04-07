import datetime
import itertools
import json
import math
import multiprocessing as mp
from functools import partial
from pathlib import Path

import click
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from .common import get_subset_list, read_data


def compute_distances_chunk(pairs_chunk, words_unit_sequences):
    results = []
    for i, j in pairs_chunk:
        units_i = words_unit_sequences[i]
        units_j = words_unit_sequences[j]
        maxlen = max(len(units_i), len(units_j))
        dist = Levenshtein.distance(units_i, units_j) / maxlen if maxlen > 0 else 0
        results.append((i, j, dist))
    return results


@click.command()
@click.option(
    "--subset",
    type=click.Choice(get_subset_list()),
    help="The LibriSpeech subset to use",
    required=True,
    prompt="Pick a subset:",
)
@click.option(
    "--units-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Directory containing the 1D unit IDs",
    required=True,
    prompt="Specify directory containing unit IDs duplicated at some unit rate (e.g. 50 Hz)",
)
@click.option(
    "--unit-rate",
    type=click.FLOAT,
    help="How many units per second",
    required=True,
    prompt="Specify unit rate (Hz)",
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Directory to save logs results to",
    required=True,
    prompt="Specify the log directory (for results)",
)
@click.option(
    "--run-name",
    type=click.STRING,
    help="Name of subdirectory to save logs results to. If not specified, will use current timestamp",
    default="",
)
@click.option(
    "--num-processes",
    type=click.INT,
    help="Number of CPU cores to use for parallel computation of chunks Try setting to one less than your total number of cores.",
    default=mp.cpu_count(),
)
def main(
    subset,
    units_dir,
    unit_rate,
    log_dir,
    run_name,
    num_processes,
):
    click.echo(
        click.style(
            f"\nRunning evaluation using Levenshtein distance on {subset} subset of LibriSpeech...\n",
            fg="green",
            bold=True,
        )
    )

    # ============================ Create log dir ============================
    if run_name == "":
        run_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = Path(log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "options.txt").write_text(
        f"subset: {subset}\n"
        f"units_dir: {units_dir}\n"
        f"unit_rate: {unit_rate}\n"
        f"num_processes: {num_processes}"
    )

    # ========================== Load word data ==========================
    df = read_data(subset)

    # ========================= Extract units =========================
    click.echo("Loading units...")
    words_unit_sequences = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        _, _, filename, start, end = row
        matched_paths = Path(units_dir).rglob(f"{filename}.npy")
        try:
            unit_path = next(matched_paths)
        except StopIteration:
            raise click.ClickException(
                f"Could not find file {filename}.npy in {units_dir}"
            )
        units = np.load(unit_path)
        start = round(start * unit_rate)
        end = round(end * unit_rate)
        units = units[start:end]
        units = [k for k, _ in itertools.groupby(units)]
        units = np.array(units)
        words_unit_sequences.append(units)

    # ================== Compute pairwise Levenshtein distances ==================
    click.echo(
        f"Computing pairwise Levenshtein distances using {num_processes}..."
    )
    n = len(words_unit_sequences)
    distance = np.zeros((n, n))

    # Generate all unique pairs
    pairs = list(itertools.combinations(range(n), 2))

    number_of_pairs = len(pairs)
    chunk_size = math.ceil(number_of_pairs / num_processes)

    # Process chunks in parallel
    pair_chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    compute_chunk_partial = partial(
        compute_distances_chunk, words_unit_sequences=words_unit_sequences
    )
    click.echo(f"Processing {len(pair_chunks)} chunks in parallel...")
    all_results = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(pair_chunks)) as pbar:
            for chunk_result in pool.imap(compute_chunk_partial, pair_chunks):
                all_results.extend(chunk_result)
                pbar.update(1)

    # Fill the distance matrix
    for i, j, dist in all_results:
        distance[i, j] = dist
        distance[j, i] = dist

    # ========================== Save distances ==========================
    np.save(save_dir / f"{subset}-levenshtein-distances.npy", distance)
    click.echo(
        f"Saved distance matrix to {save_dir / f'{subset}-levenshtein-distances.npy'}"
    )

    click.echo("Computing precision and recall...")
    words = df.word.to_numpy()
    speaker_ids = df.speaker_id.to_numpy()

    # ================= Construct ground truth results ==================
    SW = words[:, np.newaxis] == words[np.newaxis, :]
    SP = speaker_ids[:, np.newaxis] == speaker_ids[np.newaxis, :]
    SWSP = np.logical_and(SW, SP)
    SWDP = np.logical_and(SW, ~SP)

    # ============== Calculate precision and recall curves ==============
    upper_tri = np.triu_indices(len(words), k=1)
    # Note: We negate the distance to get higher values for more similar items
    P_SW, R_SW, thresholds = precision_recall_curve(SW[upper_tri], -distance[upper_tri])
    _, R_SWSP, _ = precision_recall_curve(SWSP[upper_tri], -distance[upper_tri])
    _, R_SWDP, _ = precision_recall_curve(SWDP[upper_tri], -distance[upper_tri])

    # ==================== Calculate average precision ====================
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
    click.echo(
        f"Saved precision-recall plot to {save_dir / f'{subset}-precision-recall.png'}"
    )

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
