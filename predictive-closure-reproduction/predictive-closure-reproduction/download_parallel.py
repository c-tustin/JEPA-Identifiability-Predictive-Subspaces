"""Download the official CIFAR archive in ranges, then verify upstream hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
SIZE = 170498071
MD5 = "c58f30108f718f92721af3b95e74349a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    upstream = args.upstream.resolve()
    spec = importlib.util.spec_from_file_location("official_download", upstream / "download_data.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    data = upstream / "data"
    root = data / "cifar-10-batches-py"
    if root.exists():
        official.verify(root)
        print("Existing CIFAR-10 files verified", flush=True)
        return
    parts = data / "download-parts"
    parts.mkdir(parents=True, exist_ok=True)
    part_count = 16
    width = (SIZE + part_count - 1) // part_count

    def fetch(index):
        start, end = index * width, min((index + 1) * width, SIZE) - 1
        path = parts / f"part{index:02d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return index
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                raise RuntimeError("Server did not honor the requested byte range")
            with path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if path.stat().st_size != end - start + 1:
            raise RuntimeError("Truncated download part")
        return index

    def retry(index):
        for attempt in range(3):
            try:
                return fetch(index)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(retry, i) for i in range(part_count)]):
            print(f"Verified length of part {future.result() + 1}/{part_count}", flush=True)
    archive = data / "cifar-parallel.tar.gz"
    with archive.open("wb") as output:
        for index in range(part_count):
            with (parts / f"part{index:02d}").open("rb") as part:
                shutil.copyfileobj(part, output)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise RuntimeError("Official archive MD5 mismatch; not extracting")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(data, filter="data")
    official.verify(root)
    print("Official archive MD5 and all extracted SHA-256 hashes verified", flush=True)


if __name__ == "__main__":
    main()
