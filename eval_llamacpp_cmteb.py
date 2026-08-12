#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from mteb.models.abs_encoder import AbsEncoder


def patch_local_cmteb_dataset(cmteb_root: str):
    """
    Map load_dataset("C-MTEB/xxx") or load_dataset("mteb/xxx")
    to local C-MTEB directory.
    """
    import datasets

    old_load_dataset = datasets.load_dataset
    cmteb_root = os.path.abspath(cmteb_root)

    alias = {
        "TNews": "TNews-classification",
        "TNews-classification": "TNews-classification",
        "IFlyTek": "IFlyTek-classification",
        "IFlyTek-classification": "IFlyTek-classification",
        "MultilingualSentiment": "MultilingualSentiment-classification",
        "MultilingualSentiment-classification": "MultilingualSentiment-classification",
        "JDReview": "JDReview-classification",
        "JDReview-classification": "JDReview-classification",
        "OnlineShopping": "OnlineShopping-classification",
        "OnlineShopping-classification": "OnlineShopping-classification",
        "Waimai": "waimai-classification",
        "waimai-classification": "waimai-classification",

        "CLSClusteringS2S.v2": "CLSClusteringS2S",
        "CLSClusteringP2P.v2": "CLSClusteringP2P",
        "CLSClusteringS2S": "CLSClusteringS2S",
        "CLSClusteringP2P": "CLSClusteringP2P",
        "ThuNewsClusteringS2S.v2": "ThuNewsClusteringS2S",
        "ThuNewsClusteringP2P.v2": "ThuNewsClusteringP2P",
        "ThuNewsClusteringS2S": "ThuNewsClusteringS2S",
        "ThuNewsClusteringP2P": "ThuNewsClusteringP2P",

        "Ocnli": "OCNLI",
        "OCNLI": "OCNLI",
        "Cmnli": "CMNLI",
        "CMNLI": "CMNLI",
        "ATEC": "ATEC",
        "BQ": "BQ",
        "LCQMC": "LCQMC",
        "PAWSX": "PAWSX",
        "AFQMC": "AFQMC",
        "QBQTC": "QBQTC",
        "STSB": "STSB",

        "T2Reranking": "T2Reranking",
        "T2Reranking_zh2en": "T2Reranking_zh2en",
        "T2Reranking_en2zh": "T2Reranking_en2zh",
        "MMarcoReranking": "Mmarco-reranking",
        "Mmarco-reranking": "Mmarco-reranking",
        "CMedQAv1-reranking": "CMedQAv1-reranking",
        "CMedQAv2-reranking": "CMedQAv2-reranking",

        "T2Retrieval": "T2Retrieval",
        "MMarcoRetrieval": "MMarcoRetrieval",
        "DuRetrieval": "DuRetrieval",
        "CovidRetrieval": "CovidRetrieval",
        "CmedqaRetrieval": "CmedqaRetrieval",
        "EcomRetrieval": "EcomRetrieval",
        "MedicalRetrieval": "MedicalRetrieval",
        "VideoRetrieval": "VideoRetrieval",
    }

    def new_load_dataset(path, *args, **kwargs):
        if isinstance(path, str) and (path.startswith("C-MTEB/") or path.startswith("mteb/")):
            _, name = path.split("/", 1)

            candidates = [
                os.path.join(cmteb_root, name),
            ]

            if name.endswith(".v2"):
                candidates.append(os.path.join(cmteb_root, name[:-3]))

            if name in alias:
                candidates.append(os.path.join(cmteb_root, alias[name]))

            try:
                local_names = os.listdir(cmteb_root)
                lower_map = {x.lower(): x for x in local_names}

                if name.lower() in lower_map:
                    candidates.append(os.path.join(cmteb_root, lower_map[name.lower()]))

                if name.endswith(".v2") and name[:-3].lower() in lower_map:
                    candidates.append(os.path.join(cmteb_root, lower_map[name[:-3].lower()]))

                if name in alias and alias[name].lower() in lower_map:
                    candidates.append(os.path.join(cmteb_root, lower_map[alias[name].lower()]))
            except Exception:
                pass

            uniq_candidates = []
            seen = set()
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    uniq_candidates.append(c)

            for local_path in uniq_candidates:
                if os.path.exists(local_path):
                    print(f"[LOCAL DATASET] {path} -> {local_path}", flush=True)
                    return old_load_dataset(local_path, *args, **kwargs)

            print(f"[LOCAL DATASET MISS] {path} candidates={uniq_candidates}", flush=True)

        return old_load_dataset(path, *args, **kwargs)

    datasets.load_dataset = new_load_dataset

    patched = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue

        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue

        if d.get("load_dataset") is old_load_dataset:
            d["load_dataset"] = new_load_dataset
            patched += 1

    print(f"[LOCAL DATASET] root={cmteb_root}, patched_modules={patched}", flush=True)


def parse_servers(items: List[str]) -> Dict[str, str]:
    servers = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--servers item must be name=url, got: {item}")

        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip().rstrip("/")

        if not name or not url:
            raise ValueError(f"bad server item: {item}")

        servers[name] = url

    return servers


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def flatten_embedding(emb: Any) -> List[float]:
    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
        arr = np.asarray(emb, dtype=np.float32)
        return arr.mean(axis=0).astype(np.float32).tolist()
    return emb


@dataclass
class LlamaCppEmbeddingModel(AbsEncoder):
    name: str
    base_url: str
    model: str = "dummy"
    batch_size: int = 1
    timeout: int = 600
    api_key: str = "no-key"
    normalize: bool = False
    retry: int = 3
    sleep_on_retry: float = 2.0

    max_chars: int = 0
    skip_bad_embedding: bool = False
    sanitize_nan: bool = False

    debug_bad_embedding_dir: str = "."
    bad_embedding_log_path: str = ""

    embedding_dim: int = 0

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.emb_url = self.base_url + "/embeddings"
        else:
            self.emb_url = self.base_url + "/v1/embeddings"

        self.model_name = self.name
        self.model_id = self.name
        self.model_name_or_path = self.name
        self.revision = "local"
        self.model_revision = "local"

        self.mteb_model_meta = None
        self.model_card_data = None
        self.max_seq_length = 512

    def _log_jsonl(self, item: Dict[str, Any]) -> None:
        if not self.bad_embedding_log_path:
            return

        log_path = Path(self.bad_embedding_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _dump_bad_embedding_batch(
        self,
        texts: List[str],
        arr: np.ndarray,
        bad_count: int,
        reason: str,
    ) -> str:
        bad_mask = ~np.isfinite(arr)
        bad_rows = np.where(bad_mask.any(axis=1))[0].tolist()

        dump_dir = Path(self.debug_bad_embedding_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

        dump_path = dump_dir / f"bad_embedding_batch_{self.name}_{int(time.time())}.json"

        payload = {
            "server": self.name,
            "url": self.emb_url,
            "reason": reason,
            "shape": list(arr.shape),
            "batch_size": len(texts),
            "bad_count": bad_count,
            "bad_rows": bad_rows,
            "bad_texts": [
                {
                    "row": i,
                    "text_type": type(texts[i]).__name__,
                    "text_len": len(texts[i]) if isinstance(texts[i], str) else None,
                    "text_preview": repr(texts[i])[:2000],
                }
                for i in bad_rows[:50]
            ],
        }

        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(dump_path)

    def _log_skipped_bad_embeddings(
        self,
        texts: List[str],
        arr: np.ndarray,
        reason: str,
    ) -> None:
        bad_mask = ~np.isfinite(arr)
        bad_rows = np.where(bad_mask.any(axis=1))[0].tolist()

        for i in bad_rows:
            self._log_jsonl(
                {
                    "time": int(time.time()),
                    "server": self.name,
                    "reason": reason,
                    "row": int(i),
                    "text_len": len(texts[i]) if isinstance(texts[i], str) else None,
                    "text": texts[i],
                    "replaced_with": "zero_vector",
                }
            )

    def _zero_embeddings_for_skipped_request(self, texts: List[str], reason: str, error: str) -> np.ndarray:
        dim = self.embedding_dim or 512
        arr = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            self._log_jsonl(
                {
                    "time": int(time.time()),
                    "server": self.name,
                    "reason": reason,
                    "row": int(i),
                    "text_len": len(text) if isinstance(text, str) else None,
                    "text_preview": repr(text)[:2000],
                    "error": error,
                    "replaced_with": "zero_vector",
                }
            )

        print(
            f"[SKIP REQUEST] server={self.name} reason={reason} "
            f"batch_size={len(texts)} dim={dim} replaced_with_zero err={error}",
            flush=True,
        )
        return arr

    def _request_embeddings_once(self, texts: List[str]) -> np.ndarray:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        resp = requests.post(
            self.emb_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")

        data = resp.json()
        if "data" not in data:
            raise RuntimeError(f"bad embedding response, missing data: {str(data)[:2000]}")

        rows = sorted(data["data"], key=lambda x: x.get("index", 0))
        embs = [flatten_embedding(row["embedding"]) for row in rows]

        arr = np.asarray(embs, dtype=np.float32)

        if arr.ndim != 2:
            raise RuntimeError(f"bad embedding shape: {arr.shape}")

        if arr.shape[0] != len(texts):
            raise RuntimeError(
                f"embedding row count mismatch: got={arr.shape[0]}, expected={len(texts)}"
            )

        self.embedding_dim = arr.shape[1]
        return arr

    def _handle_bad_embedding(self, texts: List[str], arr: np.ndarray, reason: str) -> np.ndarray:
        bad_count = int((~np.isfinite(arr)).sum())
        if bad_count == 0:
            return arr

        dump_path = self._dump_bad_embedding_batch(
            texts=texts,
            arr=arr,
            bad_count=bad_count,
            reason=reason,
        )

        print(
            f"[BAD EMBEDDING] server={self.name} "
            f"shape={arr.shape} bad_count={bad_count} dump={dump_path}",
            flush=True,
        )

        if self.skip_bad_embedding:
            self._log_skipped_bad_embeddings(texts=texts, arr=arr, reason=reason)

            bad_rows = np.where((~np.isfinite(arr)).any(axis=1))[0]
            for row in bad_rows:
                arr[row, :] = 0.0

            print(
                f"[SKIP BAD EMBEDDING] server={self.name} "
                f"bad_rows={bad_rows.tolist()} replaced_with_zero",
                flush=True,
            )
            return arr

        if self.sanitize_nan:
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        raise RuntimeError(f"embedding contains NaN/Inf: bad_count={bad_count}, dump={dump_path}")

    def _request_embeddings(self, texts: List[str]) -> np.ndarray:
        last_err = None

        for attempt in range(1, self.retry + 1):
            try:
                arr = self._request_embeddings_once(texts)

                bad_count = int((~np.isfinite(arr)).sum())
                if bad_count:
                    bad_rows = np.where((~np.isfinite(arr)).any(axis=1))[0].tolist()
                    fixed = 0

                    for row in bad_rows:
                        one_text = [texts[row]]
                        for retry_i in range(3):
                            time.sleep(0.2 * (retry_i + 1))
                            try:
                                one_arr = self._request_embeddings_once(one_text)
                            except Exception:
                                continue

                            if np.isfinite(one_arr).all():
                                arr[row, :] = one_arr[0]
                                fixed += 1
                                break

                    if int((~np.isfinite(arr)).sum()) == 0:
                        print(
                            f"[FIXED BAD EMBEDDING] server={self.name} fixed_rows={fixed}",
                            flush=True,
                        )
                    else:
                        arr = self._handle_bad_embedding(
                            texts=texts,
                            arr=arr,
                            reason="server returned NaN/Inf after single-row retry",
                        )

                if self.normalize:
                    arr = l2_normalize(arr)
                    arr = self._handle_bad_embedding(
                        texts=texts,
                        arr=arr,
                        reason="client-side normalize produced NaN/Inf",
                    )

                return arr

            except Exception as e:
                last_err = e
                msg = str(e)

                is_context_or_batch_error = (
                    "input is larger than the max context size" in msg
                    or "input is too large to process" in msg
                    or "increase the physical batch size" in msg
                )

                if is_context_or_batch_error:
                    if self.skip_bad_embedding:
                        return self._zero_embeddings_for_skipped_request(
                            texts=texts,
                            reason="context_or_physical_batch_too_small",
                            error=str(last_err),
                        )

                    raise RuntimeError(
                        f"embedding request failed, likely context/batch too small. "
                        f"url={self.emb_url}, batch_size={len(texts)}, err={last_err}"
                    )

                if attempt < self.retry:
                    print(
                        f"[WARN] embedding request failed, retry "
                        f"{attempt}/{self.retry}, err={repr(e)}",
                        flush=True,
                    )
                    time.sleep(self.sleep_on_retry * attempt)
                else:
                    raise RuntimeError(
                        f"embedding request failed after {self.retry} attempts, "
                        f"url={self.emb_url}, batch_size={len(texts)}, err={last_err}"
                    )

        raise RuntimeError(str(last_err))

    def _batch_encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.batch_size

        clean_texts = []
        for x in texts:
            if x is None:
                x = ""
            elif not isinstance(x, str):
                x = str(x)

            if self.max_chars and self.max_chars > 0 and len(x) > self.max_chars:
                x = x[: self.max_chars]

            clean_texts.append(x)

        if len(clean_texts) == 0:
            raise RuntimeError("encode got empty input list")

        all_embs = []
        for i in tqdm(
            range(0, len(clean_texts), batch_size),
            desc=f"encode[{self.name}]",
            leave=False,
        ):
            batch = clean_texts[i : i + batch_size]
            if not batch:
                continue
            all_embs.append(self._request_embeddings(batch))

        if not all_embs:
            raise RuntimeError("no embeddings generated")

        return np.vstack(all_embs)

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(inputs, str):
            return self._batch_encode([inputs], batch_size=batch_size)

        if isinstance(inputs, list):
            texts = []
            for item in inputs:
                if isinstance(item, dict):
                    title = item.get("title") or ""
                    text = item.get("text") or ""
                    body = item.get("body") or ""
                    merged = "\n".join(x for x in [title, text, body] if x).strip()
                    texts.append(merged)
                else:
                    texts.append("" if item is None else str(item))

            return self._batch_encode(texts, batch_size=batch_size)

        all_embs = []

        for batch in inputs:
            texts = []

            if isinstance(batch, dict):
                if "text" in batch:
                    batch_texts = batch["text"]
                elif "sentence" in batch:
                    batch_texts = batch["sentence"]
                elif "query" in batch:
                    batch_texts = batch["query"]
                elif "body" in batch:
                    batch_texts = batch["body"]
                elif "title" in batch:
                    batch_texts = batch["title"]
                else:
                    raise ValueError(f"Unsupported MTEB batch keys: {list(batch.keys())}")

                if isinstance(batch_texts, str):
                    texts = [batch_texts]
                else:
                    texts = ["" if x is None else str(x) for x in batch_texts]

            else:
                if isinstance(batch, str):
                    texts = [batch]
                else:
                    texts = ["" if x is None else str(x) for x in batch]

            if not texts:
                continue

            emb = self._batch_encode(texts, batch_size=batch_size or self.batch_size)
            all_embs.append(emb)

        if not all_embs:
            raise RuntimeError("encode got no input batches")

        return np.vstack(all_embs)

    def get_text_embeddings(
        self,
        sentences: List[str],
        *,
        task_name: Optional[str] = None,
        prompt_type=None,
        batch_size: int = 1,
        **kwargs,
    ) -> np.ndarray:
        texts = ["" if s is None else str(s) for s in sentences]
        return self._batch_encode(texts, batch_size=batch_size)


def get_tasks(args):
    import mteb

    chinese_task_list = [
        "TNews",
        "IFlyTek",
        "MultilingualSentiment",
        "JDReview",
        "OnlineShopping",
        "Waimai",
        "CLSClusteringS2S.v2",
        "CLSClusteringP2P.v2",
        "ThuNewsClusteringS2S.v2",
        "ThuNewsClusteringP2P.v2",
        "Ocnli",
        "Cmnli",
        "T2Reranking",
        "MMarcoReranking",
        "CMedQAv1-reranking",
        "CMedQAv2-reranking",
        "T2Retrieval",
        "MMarcoRetrieval",
        "DuRetrieval",
        "CovidRetrieval",
        "CmedqaRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "VideoRetrieval",
        "ATEC",
        "BQ",
        "LCQMC",
        "PAWSX",
        "STSB",
        "AFQMC",
        "QBQTC",
    ]

    task_names = args.task_names if args.task_names else chinese_task_list
    tasks = mteb.get_tasks(tasks=task_names)

    print(f"Loaded {len(tasks)} tasks:")
    for t in tasks:
        meta = getattr(t, "metadata", None)
        name = getattr(meta, "name", None) if meta else str(t)
        print("  -", name)

    return tasks


def result_to_dict(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): result_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [result_to_dict(x) for x in obj]

    if hasattr(obj, "model_dump"):
        try:
            return result_to_dict(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return result_to_dict(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return result_to_dict(
                {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            )
        except Exception:
            pass

    return str(obj)


def find_numeric_metrics(obj: Any, prefix: str = "") -> List[Tuple[str, float]]:
    rows = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(find_numeric_metrics(v, key))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            rows.extend(find_numeric_metrics(v, key))

    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(float(obj)):
            rows.append((prefix, float(obj)))

    return rows


def pick_main_score_from_dict(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    preferred_suffixes = [
        "main_score",
        "score",
        "accuracy",
        "f1",
        "f1_weighted",
        "precision",
        "recall",
        "recall_weighted",
        "ap",
        "map",
        "cos_sim_spearman",
        "spearman",
        "spearmanr",
        "pearson",
        "ndcg_at_10",
        "map_at_10",
        "recall_at_10",
        "v_measure",
    ]

    flat = find_numeric_metrics(d)

    for pref in preferred_suffixes:
        for key, val in flat:
            lk = key.lower()
            if lk == pref or lk.endswith("." + pref) or lk.endswith("_" + pref):
                return key, val

    for key, val in flat:
        lk = key.lower()
        if any(x in lk for x in ["accuracy", "f1", "spearman", "v_measure", "ndcg_at_10"]):
            return key, val

    return None, None


def infer_task_name_from_result(d: Dict[str, Any], fallback: str) -> str:
    candidates = [
        d.get("task_name"),
        d.get("dataset_name"),
        d.get("name"),
    ]

    task = d.get("task")
    if isinstance(task, dict):
        candidates.extend([task.get("name"), task.get("task_name"), task.get("dataset_name")])

    metadata = d.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend([metadata.get("name"), metadata.get("task_name"), metadata.get("dataset_name")])

    trs = d.get("task_results")
    if isinstance(trs, list) and len(trs) > 0 and isinstance(trs[0], dict):
        candidates.append(trs[0].get("task_name"))

    for x in candidates:
        if x:
            return str(x)

    return fallback


def summarize_mteb_results(
    results: Any,
    output_dir: Path,
    server_name: str,
    task_name: str,
) -> Dict[str, Any]:
    plain = result_to_dict(results)

    raw_dir = output_dir / server_name / "_raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)

    safe_task_name = task_name.replace("/", "_")
    raw_path = raw_dir / f"{safe_task_name}.json"
    raw_path.write_text(json.dumps(plain, ensure_ascii=False, indent=2), encoding="utf-8")

    metric, score = pick_main_score_from_dict(plain)

    inferred_task_name = task_name
    if isinstance(plain, dict):
        inferred_task_name = infer_task_name_from_result(plain, fallback=task_name)

    return {
        "server": server_name,
        "task": inferred_task_name,
        "main_metric": metric,
        "main_score": score,
        "status": "ok",
        "error": "",
        "source": "mteb.evaluate.single_task",
    }


def make_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty or "main_score" not in long_df.columns:
        return pd.DataFrame()

    valid = long_df.dropna(subset=["main_score"]).copy()
    if valid.empty:
        return pd.DataFrame()

    agg = (
        valid.groupby(["server", "task"], as_index=False)
        .agg(
            main_score=("main_score", "mean"),
            main_metric=("main_metric", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )

    return agg.pivot(index="task", columns="server", values="main_score").reset_index()


def run_one_server(args, server_name: str, server_url: str) -> pd.DataFrame:
    import mteb

    out_dir = Path(args.output_dir) / server_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = LlamaCppEmbeddingModel(
        name=server_name,
        base_url=server_url,
        model=args.model,
        batch_size=args.batch_size,
        timeout=args.timeout,
        api_key=args.api_key,
        normalize=args.normalize,
        retry=args.retry,
        max_chars=args.max_chars,
        skip_bad_embedding=args.skip_bad_embedding,
        sanitize_nan=args.sanitize_nan,
        debug_bad_embedding_dir=str(out_dir),
        bad_embedding_log_path=str(out_dir / "skipped_bad_embeddings.jsonl"),
    )

    if args.debug_protocol:
        try:
            from mteb.models import EncoderProtocol, SearchProtocol, CrossEncoderProtocol
            print("[DEBUG] isinstance EncoderProtocol:", isinstance(model, EncoderProtocol))
            print("[DEBUG] isinstance SearchProtocol:", isinstance(model, SearchProtocol))
            print("[DEBUG] isinstance CrossEncoderProtocol:", isinstance(model, CrossEncoderProtocol))
        except Exception as e:
            print("[DEBUG] protocol check failed:", repr(e))

    print(f"\n===== Running server: {server_name} => {server_url} =====")
    print(f"Embedding endpoint: {model.emb_url}")
    print(f"Output: {out_dir}")

    tasks = get_tasks(args)

    rows = []
    task_errors = []

    for idx, task in enumerate(tasks):
        meta = getattr(task, "metadata", None)
        task_name = getattr(meta, "name", None) if meta else str(task)

        print(f"\n===== [{server_name}] Evaluating single task: {task_name} ({idx + 1}/{len(tasks)}) =====")

        try:
            result = mteb.evaluate(
                model,
                tasks=[task],
                encode_kwargs={"batch_size": args.batch_size},
                cache=None,
                overwrite_strategy=args.overwrite_strategy,
                raise_error=True,
            )

            row = summarize_mteb_results(
                results=result,
                output_dir=Path(args.output_dir),
                server_name=server_name,
                task_name=task_name,
            )
            rows.append(row)

            print(f"[TASK OK] server={server_name} task={row['task']}: {row['main_metric']}={row['main_score']}")

        except Exception as e:
            err = traceback.format_exc()
            task_errors.append(
                {
                    "server": server_name,
                    "task": task_name,
                    "error": str(e),
                    "traceback": err,
                }
            )

            rows.append(
                {
                    "server": server_name,
                    "task": task_name,
                    "main_metric": None,
                    "main_score": math.nan,
                    "status": "error",
                    "error": str(e),
                    "source": "mteb.evaluate.single_task",
                }
            )

            print(f"[TASK ERROR] server={server_name} task={task_name}: {e}")

            if not args.continue_on_error:
                raise

    if task_errors:
        err_file = out_dir / "task_errors.json"
        err_file.write_text(
            json.dumps(task_errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[TASK ERRORS] server={server_name} saved to {err_file}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--servers",
        nargs="+",
        required=True,
        help="server list, format: name=http://host:port",
    )

    parser.add_argument("--server-workers", type=int, default=1)
    parser.add_argument("--model", default="dummy")
    parser.add_argument("--api-key", default="no-key")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retry", type=int, default=3)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sanitize-nan", action="store_true")
    parser.add_argument("--skip-bad-embedding", action="store_true")
    parser.add_argument("--max-chars", type=int, default=0)

    parser.add_argument("--output-dir", default="./cmteb_results")
    parser.add_argument("--cmteb-root", default=None)

    parser.add_argument(
        "--task-names",
        nargs="*",
        default=None,
        help="optional task names, e.g. TNews Waimai BQ LCQMC STSB",
    )

    parser.add_argument(
        "--overwrite-strategy",
        default="only-missing",
        choices=["always", "never", "only-missing", "only-cache"],
    )

    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--debug-protocol", action="store_true")

    args = parser.parse_args()

    if args.cmteb_root:
        patch_local_cmteb_dataset(args.cmteb_root)

    servers = parse_servers(args.servers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    errors = []

    server_workers = max(1, int(args.server_workers))
    server_workers = min(server_workers, len(servers))

    if server_workers == 1:
        for name, url in servers.items():
            try:
                df = run_one_server(args, name, url)
                if df is not None and not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                errors.append((name, str(e), traceback.format_exc()))
                print(f"[ERROR] server={name} failed: {e}", flush=True)
                if not args.continue_on_error:
                    raise
    else:
        print(f"[PARALLEL] evaluating {len(servers)} servers with {server_workers} workers", flush=True)

        with ThreadPoolExecutor(max_workers=server_workers) as executor:
            future_to_server = {
                executor.submit(run_one_server, args, name, url): (name, url)
                for name, url in servers.items()
            }

            for future in as_completed(future_to_server):
                name, url = future_to_server[future]

                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_dfs.append(df)

                    print(f"[SERVER OK] {name} => {url}", flush=True)

                except Exception as e:
                    errors.append((name, str(e), traceback.format_exc()))
                    print(f"[SERVER ERROR] {name} => {url}: {e}", flush=True)

                    if not args.continue_on_error:
                        raise

    if all_dfs:
        long_df = pd.concat(all_dfs, ignore_index=True)
    else:
        long_df = pd.DataFrame()

    pivot_df = make_pivot(long_df)

    long_csv = output_dir / "summary_long.csv"
    pivot_csv = output_dir / "summary_pivot.csv"
    long_json = output_dir / "summary_long.json"

    long_df.to_csv(long_csv, index=False)
    pivot_df.to_csv(pivot_csv, index=False)
    long_df.to_json(long_json, orient="records", force_ascii=False, indent=2)

    print("\n===== Summary files =====")
    print(f"Long CSV : {long_csv}")
    print(f"Pivot CSV: {pivot_csv}")
    print(f"Long JSON: {long_json}")

    if errors:
        err_file = output_dir / "errors.json"
        err_file.write_text(
            json.dumps(
                [{"server": n, "error": e, "traceback": tb} for n, e, tb in errors],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Errors   : {err_file}")

    print("\n===== Pivot preview =====")
    if not pivot_df.empty:
        print(pivot_df.head(80).to_string(index=False))
    else:
        print("No parsed result.")
        if not long_df.empty:
            print("\nLong result preview:")
            print(long_df.head(80).to_string(index=False))

    print("\n===== Skipped bad embeddings =====")
    for name in servers:
        p = output_dir / name / "skipped_bad_embeddings.jsonl"
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                cnt = sum(1 for _ in f)
            print(f"{name}: {cnt} skipped rows -> {p}")
        else:
            print(f"{name}: 0 skipped rows")


if __name__ == "__main__":
    main()
