"""SciFact external claim-verification adapter.

This benchmark is not intended to compete with SciFact systems. Its role is to
test whether MouseBrainBench can consume a public scientific claim-verification
dataset and keep claim support separate from lexical similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize
from scipy.special import expit

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_ROOT = Path("data/external/scifact/data")
DEFAULT_OUTPUT = Path("results/scifact_claim_verification/summary.json")
DEFAULT_MARKDOWN = Path("results/scifact_claim_verification/summary.md")


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "with",
            "from",
            "that",
            "this",
            "into",
            "than",
            "were",
            "been",
            "have",
            "has",
        }
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_label(claim: dict[str, Any]) -> str:
    labels = []
    for entries in claim.get("evidence", {}).values():
        labels.extend(str(entry.get("label", "NOT_ENOUGH_INFO")) for entry in entries)
    if not labels:
        return "NOT_ENOUGH_INFO"
    counts = Counter(labels)
    return str(counts.most_common(1)[0][0])


def _gold_doc_ids(claim: dict[str, Any]) -> set[int]:
    """Return annotated evidence document ids for SUPPORT/CONTRADICT claims."""

    return {int(doc_id) for doc_id, entries in claim.get("evidence", {}).items() if entries}


def _doc_text(doc: dict[str, Any]) -> str:
    return f"{doc.get('title', '')} {' '.join(str(item) for item in doc.get('abstract', []))}"


def _lexical_score(claim_text: str, cited_docs: list[int], corpus: dict[int, dict[str, Any]]) -> float:
    claim_tokens = set(_tokens(claim_text))
    if not claim_tokens:
        return 0.0
    doc_tokens: set[str] = set()
    for doc_id in cited_docs:
        doc = corpus.get(int(doc_id), {})
        doc_tokens.update(_tokens(_doc_text(doc)))
    if not doc_tokens:
        return 0.0
    return len(claim_tokens & doc_tokens) / math.sqrt(len(claim_tokens) * len(doc_tokens))


class BM25Index:
    """Small deterministic BM25 index for SciFact abstracts.

    This is a retrieval baseline, not a neural SciFact system. It is deliberately
    local and dependency-free so that claim auditing remains reproducible on a
    clean workstation.
    """

    def __init__(self, corpus: dict[int, dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = corpus
        self.term_freqs: dict[int, Counter[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lengths: dict[int, int] = {}
        for doc_id, doc in corpus.items():
            counts = Counter(_tokens(_doc_text(doc)))
            self.term_freqs[doc_id] = counts
            self.doc_lengths[doc_id] = sum(counts.values())
            self.doc_freqs.update(counts.keys())
        self.num_docs = max(1, len(corpus))
        self.avgdl = sum(self.doc_lengths.values()) / self.num_docs if self.doc_lengths else 0.0

    def _idf(self, token: str) -> float:
        df = self.doc_freqs.get(token, 0)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: int) -> float:
        query_counts = Counter(_tokens(query))
        doc_counts = self.term_freqs.get(doc_id, Counter())
        doc_len = self.doc_lengths.get(doc_id, 0)
        if not query_counts or not doc_counts or self.avgdl <= 0:
            return 0.0
        total = 0.0
        for token, query_tf in query_counts.items():
            tf = doc_counts.get(token, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
            total += self._idf(token) * ((tf * (self.k1 + 1.0)) / denom) * min(query_tf, 2)
        return float(total)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.documents]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


def _best_rationale_score(claim_text: str, doc: dict[str, Any]) -> float:
    """Score the best abstract sentence against the claim."""

    claim_tokens = set(_tokens(claim_text))
    if not claim_tokens:
        return 0.0
    best = 0.0
    for sentence in doc.get("abstract", []):
        sent_tokens = set(_tokens(str(sentence)))
        if not sent_tokens:
            continue
        overlap = len(claim_tokens & sent_tokens) / len(claim_tokens | sent_tokens)
        best = max(best, overlap)
    return float(best)


def _label_from_retrieval(
    *,
    bm25_score: float,
    rationale_score: float,
    bm25_support_threshold: float,
    rationale_support_threshold: float,
) -> str:
    """Conservative support-only label from retrieved evidence.

    Contradiction detection usually needs natural-language inference. A lexical
    retriever is allowed to say SUPPORT or NOT_ENOUGH_INFO, but not CONTRADICT.
    This design is intentional because the benchmark audits overclaiming rather
    than pretending a local lexical model solves SciFact.
    """

    if bm25_score >= bm25_support_threshold and rationale_score >= rationale_support_threshold:
        return "SUPPORT"
    return "NOT_ENOUGH_INFO"


def _retrieval_features(
    claim: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
    bm25: BM25Index,
    *,
    top_k: int,
) -> tuple[np.ndarray, list[tuple[int, float]], float]:
    """Return transparent support features without using a gold label."""

    claim_text = str(claim["claim"])
    retrieved = bm25.search(claim_text, top_k=top_k)
    top_doc_id = retrieved[0][0] if retrieved else None
    top_score = float(retrieved[0][1]) if retrieved else 0.0
    second_score = float(retrieved[1][1]) if len(retrieved) > 1 else 0.0
    top_doc = corpus.get(int(top_doc_id), {}) if top_doc_id is not None else {}
    rationale_score = _best_rationale_score(claim_text, top_doc)
    cited_overlap = _lexical_score(claim_text, claim.get("cited_doc_ids", []), corpus)
    features = np.asarray(
        [
            math.log1p(max(0.0, top_score)),
            rationale_score,
            cited_overlap,
            math.log1p(len(_tokens(claim_text))),
            math.log1p(max(0.0, top_score - second_score)),
        ],
        dtype=float,
    )
    return features, retrieved, rationale_score


def _fit_support_logistic(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any] | None:
    """Fit an L2-regularized logistic baseline on the training split only."""

    if len(features) < 2 or len(np.unique(labels)) < 2:
        return None
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales[scales < 1e-8] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        probabilities = expit(logits)
        epsilon = 1e-12
        loss = -np.mean(
            labels * np.log(probabilities + epsilon)
            + (1.0 - labels) * np.log(1.0 - probabilities + epsilon)
        )
        regularization = 0.05 * float(np.sum(np.square(weights[1:])))
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient[1:] += 0.10 * weights[1:]
        return float(loss + regularization), gradient

    fitted = optimize.minimize(
        lambda weights: objective(weights),
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
    )
    if not fitted.success:
        return None
    train_probabilities = expit(design @ fitted.x)
    negative_count = int(np.sum(labels == 0))
    candidates = sorted(set(float(value) for value in train_probabilities), reverse=True)
    operating_points = []
    for threshold in candidates:
        predictions = train_probabilities >= threshold
        fp = int(np.sum(predictions & (labels == 0)))
        tp = int(np.sum(predictions & (labels == 1)))
        fpr = fp / negative_count if negative_count else 0.0
        tpr = tp / int(np.sum(labels == 1))
        if fpr <= 0.10:
            operating_points.append((tpr, -threshold, threshold, fpr))
    threshold = max(operating_points)[2] if operating_points else 1.0
    return {
        "means": means,
        "scales": scales,
        "weights": fitted.x,
        "threshold": float(threshold),
        "train_fpr_constraint": 0.10,
        "optimizer_success": True,
    }


def _support_probability(features: np.ndarray, model: dict[str, Any]) -> float:
    standardized = (features - model["means"]) / model["scales"]
    design = np.concatenate(([1.0], standardized))
    return float(expit(design @ model["weights"]))


def run(
    root: Path = DEFAULT_ROOT,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    max_claims: int | None = None,
) -> Path:
    """Run a lightweight SciFact claim-auditing benchmark."""

    started = time.perf_counter()
    claims_path = root / "claims_dev.jsonl"
    train_path = root / "claims_train.jsonl"
    corpus_path = root / "corpus.jsonl"
    if not claims_path.exists() or not corpus_path.exists():
        payload = {
            "version": __version__,
            "git_revision": code_revision(),
            "analysis": "scifact_claim_verification",
            "decision": "scifact_data_missing",
            "missing": [str(path) for path in (claims_path, corpus_path) if not path.exists()],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        write_markdown(payload, markdown)
        return output

    claims = _load_jsonl(claims_path)
    if max_claims is not None:
        claims = claims[:max_claims]
    corpus = {int(doc["doc_id"]): doc for doc in _load_jsonl(corpus_path)}
    bm25 = BM25Index(corpus)

    rows: list[dict[str, Any]] = []
    threshold = 0.18
    abstain_threshold = 0.12
    bm25_support_threshold = 4.0
    rationale_support_threshold = 0.18
    top_k = 5
    calibrated_model = None
    train_claims: list[dict[str, Any]] = []
    if train_path.exists():
        train_claims = _load_jsonl(train_path)
        train_features = []
        train_labels = []
        for train_claim in train_claims:
            features, _retrieved, _rationale = _retrieval_features(
                train_claim,
                corpus,
                bm25,
                top_k=top_k,
            )
            train_features.append(features)
            train_labels.append(_gold_label(train_claim) == "SUPPORT")
        calibrated_model = _fit_support_logistic(
            np.asarray(train_features, dtype=float),
            np.asarray(train_labels, dtype=float),
        )
    for claim in claims:
        label = _gold_label(claim)
        claim_text = str(claim["claim"])
        gold_docs = _gold_doc_ids(claim)
        features, retrieved, rationale_score = _retrieval_features(
            claim,
            corpus,
            bm25,
            top_k=top_k,
        )
        top_doc_id = retrieved[0][0] if retrieved else None
        top_bm25 = retrieved[0][1] if retrieved else 0.0
        retrieval_label = _label_from_retrieval(
            bm25_score=top_bm25,
            rationale_score=rationale_score,
            bm25_support_threshold=bm25_support_threshold,
            rationale_support_threshold=rationale_support_threshold,
        )
        retrieved_doc_ids = [doc_id for doc_id, _score in retrieved]
        evidence_retrieved = bool(gold_docs & set(retrieved_doc_ids)) if gold_docs else False
        score = _lexical_score(claim_text, claim.get("cited_doc_ids", []), corpus)
        shortcut_supported = score >= threshold
        abstaining_supported = score >= threshold
        abstained = score < abstain_threshold
        gold_supported = label == "SUPPORT"
        retrieval_supported = retrieval_label == "SUPPORT"
        calibrated_probability = (
            _support_probability(features, calibrated_model)
            if calibrated_model is not None
            else None
        )
        calibrated_supported = (
            calibrated_probability >= calibrated_model["threshold"]
            if calibrated_model is not None and calibrated_probability is not None
            else False
        )
        rows.append(
            {
                "claim_id": claim["id"],
                "gold_label": label,
                "gold_doc_ids": sorted(gold_docs),
                "lexical_score": score,
                "bm25_top_doc_id": top_doc_id,
                "bm25_top_score": top_bm25,
                "bm25_topk_doc_ids": retrieved_doc_ids,
                "evidence_retrieved_at_5": evidence_retrieved,
                "rationale_score": rationale_score,
                "retrieval_label": retrieval_label,
                "shortcut_supported": shortcut_supported,
                "abstained": abstained,
                "abstaining_supported": False if abstained else abstaining_supported,
                "retrieval_supported": retrieval_supported,
                "calibrated_support_probability": calibrated_probability,
                "calibrated_supported": calibrated_supported,
                "gold_supported": gold_supported,
                "shortcut_false_positive": shortcut_supported and not gold_supported,
                "shortcut_false_negative": (not shortcut_supported) and gold_supported,
                "abstaining_false_positive": (not abstained) and abstaining_supported and not gold_supported,
                "abstaining_false_negative": (not abstained) and (not abstaining_supported) and gold_supported,
                "retrieval_false_positive": retrieval_supported and not gold_supported,
                "retrieval_false_negative": (not retrieval_supported) and gold_supported,
                "calibrated_false_positive": calibrated_supported and not gold_supported,
                "calibrated_false_negative": (not calibrated_supported) and gold_supported,
                "retrieval_failed_despite_gold_evidence": bool(gold_docs) and not evidence_retrieved,
            }
        )

    fp = sum(row["shortcut_false_positive"] for row in rows)
    fn = sum(row["shortcut_false_negative"] for row in rows)
    gold_positive = sum(row["gold_supported"] for row in rows)
    gold_negative = len(rows) - gold_positive
    label_counts = Counter(row["gold_label"] for row in rows)
    per_label = {}
    for label in sorted(label_counts):
        subset = [row for row in rows if row["gold_label"] == label]
        per_label[label] = {
            "n": len(subset),
            "mean_lexical_score": sum(float(row["lexical_score"]) for row in subset) / len(subset),
            "shortcut_supported_rate": sum(row["shortcut_supported"] for row in subset) / len(subset),
        }
    non_abstained = [row for row in rows if not row["abstained"]]
    abstaining_fp = sum(row["abstaining_false_positive"] for row in non_abstained)
    abstaining_fn = sum(row["abstaining_false_negative"] for row in non_abstained)
    abstaining_gold_positive = sum(row["gold_supported"] for row in non_abstained)
    abstaining_gold_negative = len(non_abstained) - abstaining_gold_positive
    retrieval_fp = sum(row["retrieval_false_positive"] for row in rows)
    retrieval_fn = sum(row["retrieval_false_negative"] for row in rows)
    evidence_claims = [row for row in rows if row["gold_doc_ids"]]
    retrieval_recall_at_5 = (
        sum(row["evidence_retrieved_at_5"] for row in evidence_claims) / len(evidence_claims)
        if evidence_claims
        else 0.0
    )
    retrieval_label_counts = Counter(row["retrieval_label"] for row in rows)
    calibrated_fp = sum(row["calibrated_false_positive"] for row in rows)
    calibrated_fn = sum(row["calibrated_false_negative"] for row in rows)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "scifact_claim_verification",
        "dataset": "SciFact dev",
        "train_split": "SciFact train" if calibrated_model is not None else None,
        "num_training_claims": len(train_claims),
        "num_claims": len(rows),
        "label_counts": dict(label_counts),
        "bm25_top_k": top_k,
        "bm25_support_threshold": bm25_support_threshold,
        "rationale_support_threshold": rationale_support_threshold,
        "lexical_threshold": threshold,
        "abstain_threshold": abstain_threshold,
        "shortcut_false_positives": fp,
        "shortcut_false_negatives": fn,
        "shortcut_overclaiming_risk": fp / gold_negative if gold_negative else 0.0,
        "shortcut_conservativeness": fn / gold_positive if gold_positive else 0.0,
        "abstention_rate": 1.0 - (len(non_abstained) / len(rows) if rows else 0.0),
        "abstaining_overclaiming_risk": (
            abstaining_fp / abstaining_gold_negative if abstaining_gold_negative else 0.0
        ),
        "abstaining_conservativeness": (
            abstaining_fn / abstaining_gold_positive if abstaining_gold_positive else 0.0
        ),
        "retrieval_label_counts": dict(retrieval_label_counts),
        "retrieval_recall_at_5": retrieval_recall_at_5,
        "retrieval_false_positives": retrieval_fp,
        "retrieval_false_negatives": retrieval_fn,
        "retrieval_overclaiming_risk": retrieval_fp / gold_negative if gold_negative else 0.0,
        "retrieval_conservativeness": retrieval_fn / gold_positive if gold_positive else 0.0,
        "calibrated_baseline_available": calibrated_model is not None,
        "calibrated_feature_names": [
            "log_bm25_top_score",
            "top_rationale_overlap",
            "cited_document_overlap",
            "log_claim_token_count",
            "log_bm25_top_margin",
        ],
        "calibrated_threshold": (
            calibrated_model["threshold"] if calibrated_model is not None else None
        ),
        "calibrated_train_fpr_constraint": (
            calibrated_model["train_fpr_constraint"] if calibrated_model is not None else None
        ),
        "calibrated_false_positives": calibrated_fp,
        "calibrated_false_negatives": calibrated_fn,
        "calibrated_false_positive_rate": (
            calibrated_fp / gold_negative if gold_negative else 0.0
        ),
        "calibrated_false_negative_rate": (
            calibrated_fn / gold_positive if gold_positive else 0.0
        ),
        "claims_with_gold_evidence": len(evidence_claims),
        "retrieval_failed_gold_evidence": sum(
            row["retrieval_failed_despite_gold_evidence"] for row in rows
        ),
        "per_label": per_label,
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
        "decision": (
            "scifact_external_claim_audit_ready"
            if len(rows) >= 100 and fp > 0 and retrieval_recall_at_5 > 0.0
            else "scifact_external_claim_audit_insufficient"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write SciFact adapter report."""

    lines = [
        "# SciFact Claim Verification Adapter",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claims: `{payload.get('num_claims', 0)}`",
        f"- Label counts: `{payload.get('label_counts', {})}`",
        f"- Shortcut FPR: `{payload.get('shortcut_overclaiming_risk', 0.0):.3f}`",
        f"- Shortcut FNR: `{payload.get('shortcut_conservativeness', 0.0):.3f}`",
        f"- BM25 evidence recall@5: `{payload.get('retrieval_recall_at_5', 0.0):.3f}`",
        f"- BM25/rationale FPR: `{payload.get('retrieval_overclaiming_risk', 0.0):.3f}`",
        f"- BM25/rationale FNR: `{payload.get('retrieval_conservativeness', 0.0):.3f}`",
        f"- Train-calibrated baseline available: `{payload.get('calibrated_baseline_available')}`",
        f"- Train-calibrated baseline FPR: `{payload.get('calibrated_false_positive_rate', 0.0):.3f}`",
        f"- Train-calibrated baseline FNR: `{payload.get('calibrated_false_negative_rate', 0.0):.3f}`",
        f"- Abstention rate: `{payload.get('abstention_rate', 0.0):.3f}`",
        f"- Abstaining FPR: `{payload.get('abstaining_overclaiming_risk', 0.0):.3f}`",
        f"- Runtime seconds: `{payload.get('runtime_seconds', 0.0):.3f}`",
        "",
        "Interpretation: BM25/rationale is a transparent local evidence-retrieval "
        "baseline. It is used to separate retrieval, support classification, "
        "and overclaiming risk; it is not a SciFact SOTA system.",
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-claims", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.root, args.output, args.markdown, args.max_claims).resolve())}))


if __name__ == "__main__":
    main()
