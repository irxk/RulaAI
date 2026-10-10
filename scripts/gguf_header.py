"""Read a GGUF header from a local file or, with HTTP range requests, a URL.

Only the header is fetched for a URL: metadata, tensor names, types, shapes and
offsets — enough to find which tensors one file has that another lacks, and
where their bytes are, without downloading the whole model.
"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}


class Source:
    """Byte access with a growing prefix cache, local or remote."""

    def __init__(self, where: str, cache_dir: str | None = None):
        self.where = where
        self.remote = where.startswith("http")
        self.buf = b""
        self.cache = None
        if self.remote and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = os.path.join(cache_dir, hashlib.sha1(where.encode()).hexdigest()[:16]
                                      + ".head")
            if os.path.exists(self.cache):
                self.buf = open(self.cache, "rb").read()

    def need(self, end: int):
        if len(self.buf) >= end:
            return
        want = max(end, len(self.buf) * 2, 8 << 20)
        self.buf += self.fetch(len(self.buf), want - len(self.buf))
        if self.cache:
            open(self.cache, "wb").write(self.buf)

    def fetch(self, start: int, size: int) -> bytes:
        if not self.remote:
            with open(self.where, "rb") as f:
                f.seek(start)
                return f.read(size)
        req = urllib.request.Request(self.where, headers={
            "Range": f"bytes={start}-{start + size - 1}", "User-Agent": "rula-gguf-header"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:
                raise RuntimeError(f"range request not honoured (HTTP {r.status})")
            return r.read()


def read_header(src: Source) -> dict:
    pos = 0

    def take(n):
        nonlocal pos
        src.need(pos + n)
        b = src.buf[pos:pos + n]
        pos += n
        return b

    def scalar(t):
        fmt = _SCALAR[t]
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    def string():
        n = scalar(10)
        return take(n).decode("utf-8", "replace")

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = scalar(4), scalar(10)
            return [value(et) for _ in range(n)]
        return scalar(t)

    assert take(4) == b"GGUF", "not a GGUF file"
    version, n_tensors, n_kv = scalar(4), scalar(10), scalar(10)
    kv, kv_types = {}, {}
    for _ in range(n_kv):
        k = string()
        t = scalar(4)
        kv[k] = value(t)
        kv_types[k] = t
    tensors = []
    for _ in range(n_tensors):
        name = string()
        nd = scalar(4)
        dims = [scalar(10) for _ in range(nd)]
        ttype = scalar(4)
        off = scalar(10)
        n_el = 1
        for d in dims:
            n_el *= d
        bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(ttype)]
        tensors.append({"name": name, "dims": dims, "type": GGMLQuantizationType(ttype).name,
                        "type_id": ttype, "offset": off, "nbytes": n_el // bs * ts})
    align = kv.get("general.alignment", 32)
    data_start = (pos + align - 1) // align * align
    return {"version": version, "kv": kv, "kv_types": kv_types, "tensors": tensors,
            "data_start": data_start}


def summarise_kv(v):
    if isinstance(v, list):
        return f"<array len={len(v)} sha={hashlib.sha1(json.dumps(v).encode()).hexdigest()[:8]}>"
    return v


if __name__ == "__main__":
    a = read_header(Source(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None))
    b = read_header(Source(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    print(f"A: {len(a['tensors'])} tensors, data at {a['data_start']}")
    print(f"B: {len(b['tensors'])} tensors, data at {b['data_start']}")
    print("\n--- metadata that differs (A vs B) ---")
    for k in sorted(set(a["kv"]) | set(b["kv"])):
        va, vb = summarise_kv(a["kv"].get(k)), summarise_kv(b["kv"].get(k))
        if va != vb and not k.startswith("general.") or k in ("general.architecture",):
            if va != vb:
                print(f"  {k}: {va!r}  |  {vb!r}")
    an = {t["name"]: t for t in a["tensors"]}
    bn = {t["name"]: t for t in b["tensors"]}
    only_b = [t for t in b["tensors"] if t["name"] not in an]
    only_a = [t for t in a["tensors"] if t["name"] not in bn]
    print(f"\n--- tensors only in B: {len(only_b)}, "
          f"{sum(t['nbytes'] for t in only_b) / 2**20:.1f} MiB ---")
    for t in only_b:
        print(f"  {t['name']:45s} {t['type']:7s} {t['dims']}  {t['nbytes'] / 2**20:.2f} MiB")
    print(f"--- tensors only in A: {len(only_a)} ---")
    for t in only_a[:20]:
        print(f"  {t['name']}")
    shape_diff = [n for n in an if n in bn and an[n]["dims"] != bn[n]["dims"]]
    type_diff = [n for n in an if n in bn and an[n]["type"] != bn[n]["type"]]
    print(f"\nshared tensors with different shape: {len(shape_diff)} {shape_diff[:5]}")
    print(f"shared tensors with different quant type: {len(type_diff)} "
          f"e.g. {[(n, an[n]['type'], bn[n]['type']) for n in type_diff[:4]]}")
