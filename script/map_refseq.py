#!/usr/bin/env python3
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ENSEMBL_SEQUENCE_URL = "https://rest.ensembl.org/sequence/id/{ensembl_id}"
ENSEMBL_SEQUENCE_BATCH_URL = "https://rest.ensembl.org/sequence/id"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map source-specific protein IDs to reference protein sequences."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "merged_dataset.csv"),
        help="Input merged CSV without ref_seq.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "proceed" / "merged_dataset_with_refseq.csv"),
        help="Output CSV with ref_seq and variant_ref_match.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "data" / "seq_cache_api_mapping"),
        help="Directory for cached sequence lookup results.",
    )
    parser.add_argument(
        "--ncbi-api-key",
        default=os.getenv("NCBI_API_KEY"),
        help="NCBI API key. Defaults to environment variable NCBI_API_KEY.",
    )
    parser.add_argument(
        "--ncbi-email",
        default=os.getenv("NCBI_EMAIL", "your_email@example.com"),
        help="Email sent to NCBI E-utilities.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel API workers for batch requests.",
    )
    parser.add_argument(
        "--clinvar-batch-size",
        type=int,
        default=100,
        help="Number of ClinVar/NCBI NM IDs per batch request.",
    )
    parser.add_argument(
        "--humsavar-batch-size",
        type=int,
        default=200,
        help="Number of humsavar/UniProt IDs per batch request.",
    )
    parser.add_argument(
        "--cosmic-batch-size",
        type=int,
        default=50,
        help="Number of COSMIC/Ensembl IDs per batch request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        default=2,
        help="Maximum retry count for transient HTTP/network errors.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry IDs previously cached as failed.",
    )
    return parser.parse_args()


def safe_cache_name(source, mapping_id):
    raw = f"{source}_{mapping_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def parse_first_fasta_sequence(text):
    if text is None or not text.strip():
        return None

    seq_lines = []
    in_first_record = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_first_record and seq_lines:
                break
            in_first_record = True
            continue
        if in_first_record:
            seq_lines.append(line)

    seq = "".join(seq_lines).replace("*", "").upper()
    if re.fullmatch(r"[A-Z]+", seq):
        return seq
    return None


def parse_fasta_records(text, id_parser):
    records = {}
    current_id = None
    seq_lines = []

    def flush():
        if current_id is None:
            return
        seq = "".join(seq_lines).replace("*", "").upper()
        if re.fullmatch(r"[A-Z]+", seq):
            records[current_id] = seq

    if text is None or not text.strip():
        return records

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_id = id_parser(line)
            seq_lines = []
        else:
            seq_lines.append(line)

    flush()
    return records


def parse_ncbi_header(header):
    match = re.search(r"\|([^|\s]+?)_prot_", header)
    if match:
        return match.group(1)
    return None


def parse_uniprot_header(header):
    match = re.match(r">(?:sp|tr)\|([^|]+)\|", header)
    if match:
        return match.group(1)
    return None


def request_text(url, params=None, headers=None, timeout=10.0, max_retry=2):
    for attempt in range(max_retry):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 429:
                time.sleep(2 + attempt)
                continue

            if response.status_code >= 500:
                time.sleep(1 + attempt)
                continue

            if not response.ok:
                return None

            return response.text

        except requests.RequestException:
            time.sleep(1 + attempt)

    return None


def request_json(
    url,
    params=None,
    json_body=None,
    headers=None,
    timeout=10.0,
    max_retry=2,
):
    for attempt in range(max_retry):
        try:
            response = requests.post(
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 429:
                time.sleep(2 + attempt)
                continue

            if response.status_code >= 500:
                time.sleep(1 + attempt)
                continue

            if not response.ok:
                return None

            return response.json()

        except (requests.RequestException, ValueError):
            time.sleep(1 + attempt)

    return None


def chunked(items, batch_size):
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_cache(cache_dir, source, mapping_id, retry_failed):
    cache_key = safe_cache_name(source, mapping_id)
    seq_path = cache_dir / f"{cache_key}.txt"
    miss_path = cache_dir / f"{cache_key}.miss"

    if seq_path.exists():
        seq = seq_path.read_text().strip()
        if seq:
            return seq, True
        if not retry_failed:
            return None, True

    if miss_path.exists() and not retry_failed:
        return None, True

    return None, False


def write_cache(cache_dir, source, mapping_id, seq):
    cache_key = safe_cache_name(source, mapping_id)
    seq_path = cache_dir / f"{cache_key}.txt"
    miss_path = cache_dir / f"{cache_key}.miss"

    if seq:
        seq_path.write_text(seq)
        if miss_path.exists():
            miss_path.unlink()
    else:
        miss_path.write_text("")


def cached_or_pending(tasks, args, cache_dir):
    cached_results = []
    pending = []

    for source, mapping_id in tasks:
        cached_seq, from_cache = read_cache(
            cache_dir=cache_dir,
            source=source,
            mapping_id=mapping_id,
            retry_failed=args.retry_failed,
        )
        if from_cache:
            cached_results.append(
                {
                    "source": source,
                    "mapping_id": mapping_id,
                    "ref_seq": cached_seq,
                    "status": "cache_hit" if cached_seq else "cache_miss",
                }
            )
        else:
            pending.append((source, mapping_id))

    return cached_results, pending


def fetch_clinvar_refseq_protein(mapping_id, args):
    params = {
        "db": "nuccore",
        "id": mapping_id,
        "rettype": "fasta_cds_aa",
        "retmode": "text",
        "email": args.ncbi_email,
    }
    if args.ncbi_api_key:
        params["api_key"] = args.ncbi_api_key

    text = request_text(
        NCBI_EFETCH_URL,
        params=params,
        timeout=args.timeout,
        max_retry=args.max_retry,
    )
    return parse_first_fasta_sequence(text)


def fetch_clinvar_refseq_batch(mapping_ids, args):
    params = {
        "db": "nuccore",
        "id": ",".join(mapping_ids),
        "rettype": "fasta_cds_aa",
        "retmode": "text",
        "email": args.ncbi_email,
    }
    if args.ncbi_api_key:
        params["api_key"] = args.ncbi_api_key

    text = request_text(
        NCBI_EFETCH_URL,
        params=params,
        timeout=args.timeout,
        max_retry=args.max_retry,
    )
    seq_by_id = parse_fasta_records(text, parse_ncbi_header)
    return {mapping_id: seq_by_id.get(mapping_id) for mapping_id in mapping_ids}


def fetch_cosmic_ensembl_protein(mapping_id, args):
    headers = {"Content-Type": "text/plain"}
    ids_to_try = [mapping_id]

    if "." in mapping_id:
        ids_to_try.append(mapping_id.split(".", 1)[0])

    for ensembl_id in ids_to_try:
        text = request_text(
            ENSEMBL_SEQUENCE_URL.format(ensembl_id=ensembl_id),
            params={"type": "protein", "species": "homo_sapiens"},
            headers=headers,
            timeout=args.timeout,
            max_retry=args.max_retry,
        )
        seq = parse_first_fasta_sequence(text)
        if seq:
            return seq

    return None


def fetch_cosmic_ensembl_batch(mapping_ids, args):
    base_to_originals = {}
    for mapping_id in mapping_ids:
        base_id = mapping_id.split(".", 1)[0]
        base_to_originals.setdefault(base_id, []).append(mapping_id)

    data = request_json(
        ENSEMBL_SEQUENCE_BATCH_URL,
        json_body={
            "ids": list(base_to_originals),
            "type": "protein",
        },
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=args.timeout,
        max_retry=args.max_retry,
    )

    seq_by_original = {mapping_id: None for mapping_id in mapping_ids}
    if not isinstance(data, list):
        return seq_by_original

    for item in data:
        query = item.get("query")
        seq = item.get("seq")
        if not query or not seq:
            continue
        seq = str(seq).replace("*", "").upper()
        if not re.fullmatch(r"[A-Z]+", seq):
            continue
        for original_id in base_to_originals.get(query, []):
            seq_by_original[original_id] = seq

    return seq_by_original


def fetch_humsavar_uniprot_protein(mapping_id, args):
    text = request_text(
        UNIPROT_FASTA_URL.format(uniprot_id=mapping_id),
        timeout=args.timeout,
        max_retry=args.max_retry,
    )
    return parse_first_fasta_sequence(text)


def fetch_humsavar_uniprot_batch(mapping_ids, args):
    text = request_text(
        UNIPROT_ACCESSIONS_URL,
        params={
            "accessions": ",".join(mapping_ids),
            "format": "fasta",
        },
        timeout=args.timeout,
        max_retry=args.max_retry,
    )
    seq_by_id = parse_fasta_records(text, parse_uniprot_header)
    return {mapping_id: seq_by_id.get(mapping_id) for mapping_id in mapping_ids}


def fetch_ref_seq(source, mapping_id, args):
    if mapping_id is None:
        return None

    mapping_id = str(mapping_id).strip()
    if not mapping_id:
        return None

    if source == "ClinVar":
        return fetch_clinvar_refseq_protein(mapping_id, args)
    if source == "COSMIC":
        return fetch_cosmic_ensembl_protein(mapping_id, args)
    if source == "humsavar":
        return fetch_humsavar_uniprot_protein(mapping_id, args)

    return None


def fetch_batch(source, mapping_ids, args, cache_dir):
    if source == "ClinVar":
        seq_by_id = fetch_clinvar_refseq_batch(mapping_ids, args)
    elif source == "COSMIC":
        seq_by_id = fetch_cosmic_ensembl_batch(mapping_ids, args)
    elif source == "humsavar":
        seq_by_id = fetch_humsavar_uniprot_batch(mapping_ids, args)
    else:
        seq_by_id = {mapping_id: None for mapping_id in mapping_ids}

    results = []
    for mapping_id in mapping_ids:
        seq = seq_by_id.get(mapping_id)
        write_cache(cache_dir, source, mapping_id, seq)
        results.append(
            {
                "source": source,
                "mapping_id": mapping_id,
                "ref_seq": seq,
                "status": "api_ok" if seq else "api_miss",
            }
        )

    return results


def fetch_one(task, args, cache_dir):
    source, mapping_id = task
    cached_seq, from_cache = read_cache(
        cache_dir=cache_dir,
        source=source,
        mapping_id=mapping_id,
        retry_failed=args.retry_failed,
    )

    if from_cache:
        return {
            "source": source,
            "mapping_id": mapping_id,
            "ref_seq": cached_seq,
            "status": "cache_hit" if cached_seq else "cache_miss",
        }

    seq = fetch_ref_seq(source, mapping_id, args)
    write_cache(cache_dir, source, mapping_id, seq)

    return {
        "source": source,
        "mapping_id": mapping_id,
        "ref_seq": seq,
        "status": "api_ok" if seq else "api_miss",
    }


def check_variant_matches_refseq(protein_variant, ref_seq):
    if protein_variant is None or ref_seq is None:
        return None

    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", str(protein_variant).strip())
    if not match:
        return None

    ref_aa, pos, _alt_aa = match.groups()
    pos = int(pos)

    if pos < 1 or pos > len(ref_seq):
        return False

    return ref_seq[pos - 1] == ref_aa


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_dir}")
    print(f"Workers: {args.workers}")

    df = pl.read_csv(
        input_path,
        infer_schema_length=10000,
        schema_overrides={
            "gene_symbol": pl.Utf8,
            "protein_variant": pl.Utf8,
            "source": pl.Utf8,
            "database_id": pl.Utf8,
            "mapping_id": pl.Utf8,
            "ClinicalSig": pl.Int64,
        },
    )
    required_cols = {"source", "mapping_id", "protein_variant"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {sorted(missing_cols)}")

    mapping_table = (
        df.select(["source", "mapping_id"])
        .filter(pl.col("mapping_id").is_not_null())
        .unique(maintain_order=True)
    )
    tasks = list(mapping_table.iter_rows())

    print(f"Rows: {df.height}")
    print(f"Unique source + mapping_id pairs: {len(tasks)}")
    print("Source counts:")
    print(mapping_table["source"].value_counts())

    cached_results, pending_tasks = cached_or_pending(tasks, args, cache_dir)
    results = list(cached_results)
    status_counts = {
        "api_ok": 0,
        "api_miss": 0,
        "cache_hit": 0,
        "cache_miss": 0,
    }
    for result in cached_results:
        status_counts[result["status"]] += 1

    source_to_batch_size = {
        "ClinVar": args.clinvar_batch_size,
        "COSMIC": args.cosmic_batch_size,
        "humsavar": args.humsavar_batch_size,
    }

    batches = []
    for source in ["ClinVar", "COSMIC", "humsavar"]:
        source_ids = [
            mapping_id
            for task_source, mapping_id in pending_tasks
            if task_source == source
        ]
        for batch in chunked(source_ids, source_to_batch_size[source]):
            batches.append((source, batch))

    unknown_tasks = [
        (source, mapping_id)
        for source, mapping_id in pending_tasks
        if source not in source_to_batch_size
    ]
    for source, mapping_id in unknown_tasks:
        batches.append((source, [mapping_id]))

    print(f"Cache hits/misses already known: {len(cached_results)}")
    print(f"Pending API IDs: {len(pending_tasks)}")
    print(f"Pending API batches: {len(batches)}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_batch, source, batch, args, cache_dir): (source, batch)
            for source, batch in batches
        }

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Mapping ref_seq batches",
            unit="batch",
        )

        for future in progress:
            batch_results = future.result()
            results.extend(batch_results)
            for result in batch_results:
                status_counts[result["status"]] += 1
            progress.set_postfix(status_counts)

    seq_map = pl.DataFrame(results).select(["source", "mapping_id", "ref_seq"])

    mapped = df.join(seq_map, on=["source", "mapping_id"], how="left")
    mapped = mapped.with_columns(
        pl.struct(["protein_variant", "ref_seq"])
        .map_elements(
            lambda x: check_variant_matches_refseq(x["protein_variant"], x["ref_seq"]),
            return_dtype=pl.Boolean,
        )
        .alias("variant_ref_match")
    )

    mapped.write_csv(output_path)

    failed_path = output_path.with_suffix(".failed_mapping_ids.csv")
    seq_map.filter(pl.col("ref_seq").is_null()).write_csv(failed_path)

    print("\nDone.")
    print(f"Output written: {output_path}")
    print(f"Failed mapping IDs written: {failed_path}")
    print("Mapping status counts:")
    for key, value in status_counts.items():
        print(f"  {key}: {value}")
    print("ref_seq null rows:", mapped["ref_seq"].is_null().sum())
    print("variant_ref_match counts:")
    print(mapped["variant_ref_match"].value_counts())


if __name__ == "__main__":
    main()
