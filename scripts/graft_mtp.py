"""Graft unsloth's MTP layer (blk.32 + nextn.*) onto the HauhauCS weights.

  pip install gguf==0.19.0
  python scripts/graft_mtp.py <base.gguf> <mtp_source_url> <out.gguf> <work_dir>

How rula-4b-mtp-q4_k_m.gguf was made:
  python scripts/graft_mtp.py ..\\..\\_q4\\rula-q4_k_m.gguf ^
      https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf ^
      ..\\..\\_mtp\\rula-4b-mtp-q4_k_m.gguf ..\\..\\_mtp

Only the MTP tensors are downloaded (HTTP range requests against the header's
offsets): 77.4 MiB instead of the 2.7 GB file. Everything else — weights,
tokenizer, chat template — comes from the base file, which is read and never
modified. The two files have identical shapes for every shared tensor; the MTP
build differs only by blk.32.* and two metadata keys (block_count 32 -> 33,
nextn_predict_layers = 1), which is what this copies.
"""
import hashlib
import os
import re
import sys
import time
import urllib.request

import numpy as np

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import Source, read_header  # noqa: E402

base_path, mtp_url, out_path, work = sys.argv[1:5]
# Downloaded tensors are cached per source: the 4B and 9B MTP layers share tensor
# names, and some (the [256] norms) even share sizes, so a cache keyed by name
# alone would hand one model's weights to the other.
parts_dir = os.path.join(work, "parts-" + hashlib.sha1(mtp_url.encode()).hexdigest()[:12])
os.makedirs(parts_dir, exist_ok=True)
_repo = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/", mtp_url)
SOURCE = f"{_repo.group(1) if _repo else mtp_url} (blk.32 grafted)"


def fetch_range(start: int, size: int) -> bytes:
    for attempt in range(6):
        try:
            req = urllib.request.Request(mtp_url, headers={
                "Range": f"bytes={start}-{start + size - 1}", "User-Agent": "rula-graft"})
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status != 206:
                    raise RuntimeError(f"HTTP {r.status}")
                data = r.read()
            if len(data) != size:
                raise RuntimeError(f"short read {len(data)}/{size}")
            return data
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(3)
    raise RuntimeError("range fetch failed")


hdr = read_header(Source(mtp_url, work))
reader = gguf.GGUFReader(base_path)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
n_layers = reader.fields[f"{arch}.block_count"].contents()
have = {t.name: t for t in reader.tensors}
extra = [t for t in hdr["tensors"] if t["name"] not in have]
# The MTP block sits right after the last regular layer; anything else missing
# would mean the two files are not the same architecture.
assert all(t["name"].startswith(f"blk.{n_layers}.") for t in extra), [t["name"] for t in extra]
src_shapes = {t["name"]: t["dims"] for t in hdr["tensors"]}
mismatch = [n for n, t in have.items() if n in src_shapes and list(t.shape) != src_shapes[n]]
assert not mismatch, f"shared tensors differ in shape: {mismatch[:5]}"
print(f"base: {len(have)} tensors | MTP tensors to add: {len(extra)}, "
      f"{sum(t['nbytes'] for t in extra) / 2**20:.1f} MiB")

arrays = {}
for t in extra:
    part = os.path.join(parts_dir, t["name"] + ".bin")
    if os.path.exists(part) and os.path.getsize(part) == t["nbytes"]:
        raw = open(part, "rb").read()
    else:
        raw = fetch_range(hdr["data_start"] + t["offset"], t["nbytes"])
        open(part, "wb").write(raw)
    qt = GGMLQuantizationType(t["type_id"])
    dims = t["dims"]  # ggml order: ne0 (row length) first
    if qt == GGMLQuantizationType.F32:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(list(reversed(dims)))
    elif qt in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        arr = np.frombuffer(raw, dtype=np.uint16 if qt == GGMLQuantizationType.BF16
                            else np.float16).reshape(list(reversed(dims)))
    else:
        bs, ts = GGML_QUANT_SIZES[qt]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            list(reversed(dims[1:])) + [dims[0] // bs * ts])
    arrays[t["name"]] = (arr, qt)
    print(f"  {t['name']:42s} {qt.name:5s} {t['nbytes'] / 2**20:6.2f} MiB ok")

writer = gguf.GGUFWriter(out_path, arch=arch)
overrides = {f"{arch}.block_count": hdr["kv"][f"{arch}.block_count"]}
added = {f"{arch}.nextn_predict_layers": (hdr["kv"][f"{arch}.nextn_predict_layers"],
                                          GGUFValueType(hdr["kv_types"][f"{arch}.nextn_predict_layers"]))}
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vtype = field.types[0]
    sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
    val = overrides.get(field.name, field.contents())
    writer.add_key_value(field.name, val, vtype, sub_type=sub)
for k, (v, t) in added.items():
    writer.add_key_value(k, v, t)
writer.add_key_value("general.source.mtp", SOURCE, GGUFValueType.STRING)

for t in reader.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
for name, (arr, qt) in arrays.items():
    writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(t.data)
for name, (arr, _) in arrays.items():
    writer.write_tensor_data(arr)
writer.close()
print(f"written {out_path}: {os.path.getsize(out_path) / 2**20:.0f} MiB "
      f"(block_count {overrides}, added {list(added)})")
