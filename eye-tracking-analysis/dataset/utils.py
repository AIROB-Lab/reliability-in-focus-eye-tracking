import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def get_subject_dirs(base_path: Path, pattern: str) -> Optional[Sequence[Path]]:
    """Filter for subject directories using a name pattern.

    (From empkins-io package)

    Parameters
    ----------
    base_path : path or str
        base path to filter for directories
    pattern : str
        name pattern as regex

    Returns
    -------
    list of path
        a list of path or an empty list if no subfolders matched the ``pattern``

    Raises
    ------
    FileNotFoundError
        if no subfolders in ``base_path`` match ``pattern``.

    Examples
    --------
    >>> base_path = Path(".")
    >>> get_subject_dirs(base_path, "Vp*")

    """
    # ensure pathlib
    base_path = Path(base_path)
    subject_dirs = [p for p in sorted(base_path.glob("*")) if p.is_dir()]
    subject_dirs = list(filter(lambda s: len(re.findall(pattern, s.name)) > 0, subject_dirs))
    if len(subject_dirs) == 0:
        raise FileNotFoundError(f"No subfolders matching the pattern '{pattern}' found in {base_path}.")
    return subject_dirs


def add_performance_exclusion(
    pbr: pd.DataFrame, performance_thr: float = 0.5
) -> Optional[pd.DataFrame]:
    """Add a column to the pasta box results indicating if participant and cell are excluded based on performance.

    Parameters
    ----------
    pbr : pd.DataFrame
        pasta box results dataframe
    performance_thr : float, optional
        performance threshold, by default 0.5

    Returns
    -------
    pd.DataFrame
        pasta box results dataframe with the column "excl_performance" added
    """
    pbr = pbr.copy()

    num_events = 20 * 3  # 20 runs, 3 movements

    performance_stats = get_performance_stats(pbr)
    performance_stats["max_num_successes"] = num_events - performance_stats["failure_planned"]

    # Exclude participants and cells with less than perf_thr% successes of the number of possible successes
    perf_excl = performance_stats.index[
        performance_stats.semi_success < performance_thr * performance_stats.max_num_successes
    ]
    # if len(perf_excl) > 0:
    #     print(f"Excluding the following cells:")
    #     print(perf_excl)

    # Add column to pasta box results indicating if participant and cell are excluded
    pbr["excl_performance"] = False
    pbr = pbr.reset_index(["run", "mov"])
    pbr.loc[perf_excl, "excl_performance"] = True

    pbr = pbr.reset_index().set_index(["participant", "cell", "run", "mov"])
    return pbr



def get_performance_stats(pbr):
    """Returns the fail(ure) counts per participant and cell."""
    pbr = pbr[~pbr["excl_bc_repetition"]].copy()

    # fill nans with False, so that sum works correctly
    pbr["semi_success"] = pbr["semi_success"].fillna(False)
    pbr["failure_started"] = pbr["failure_started"].fillna(False)

    # Count number of successes per participant and cell
    sums = pbr.reset_index().groupby(["participant", "cell"], observed=True).sum(numeric_only=True)
    sums.drop(["run", "planned_failure_perc"], axis=1, inplace=True, errors="ignore")
    return sums
