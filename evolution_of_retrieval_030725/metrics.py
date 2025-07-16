import numpy as np
import pandas as pd

def eval_retriever(run_fn, df, k=10, *, qcol="question_sc", gtcol="page_cf"):
    """Compute Recall@k, Precision@k, and MRR (unconditional) for a retrieval function.

    A hit contributes 1/rank to the MRR; misses are assigned a sentinel rank of k+1
    and thus contribute 1/(k+1) (a small non-zero value) to the reciprocal ranks.
    If you prefer misses to contribute exactly zero, you could mask them with
    `reciprocal_ranks = np.where(rank <= k, 1.0/rank, 0.0)` before averaging.

    Args:
        run_fn: Callable[[str, int], list[str]]
            Function taking (query, k) and returning top-k IDs.
        df: pd.DataFrame
            Queries with ground-truth IDs.
        k: int
            Cut-off for top-k.
        qcol: str
            Column name for query text.
        gtcol: str
            Column name for ground-truth ID.

    Returns:
        dict:
            {
                "recall@k": float,     # fraction of queries with a hit in top-k
                "precision@k": float,  # fraction of retrieved docs that are correct
                "mrr": float,          # mean reciprocal rank over all queries
                "name": str            # run_fn.__name__
            }
    """
    # predictions and ground truth arrays
    y_pred = np.array([run_fn(q, k) for q in df[qcol]])  # shape (n, k)
    y_true = df[gtcol].to_numpy()                        # shape (n,)

    # compute rank: position 1..k, or k+1 if missing
    rank = np.where(y_pred == y_true[:, None],
                    np.arange(1, k + 1),
                    k + 1).min(axis=1)

    # hits mask and recall@k
    hits = rank <= k
    recall = hits.mean()

    # precision@k = total hits / total retrieved (k * n)
    precision = hits.sum() / (k * len(y_true))

    # unconditional MRR: include misses as 1/(k+1)
    reciprocal_ranks = 1.0 / rank
    mrr = reciprocal_ranks.mean()

    return {
        "recall@k": round(recall, 4),
        "precision@k": round(precision, 4),
        "mrr": round(mrr, 4),
        "name": run_fn.__name__
    }