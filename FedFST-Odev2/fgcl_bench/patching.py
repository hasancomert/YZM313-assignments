"""Idempotent metin yamaları.

apply_patch(), dosyada `old` bloğunu satır satır arar (satır sonu boşlukları yok
sayılır, çünkü orijinal dosyalarda boş satırlarda girinti boşlukları var) ve
`new` ile değiştirir. `new` bloğu dosyada zaten varsa hiçbir şey yapmaz; bu
yüzden hücre iki kez çalıştırılınca dosya bozulmaz.
"""
import difflib
from pathlib import Path


def _norm(block):
    return [line.rstrip() for line in block.strip("\n").split("\n")]


def _find(file_lines, block_lines):
    n = len(block_lines)
    return [i for i in range(len(file_lines) - n + 1) if file_lines[i:i + n] == block_lines]


def create_files(root, pid, files, content):
    created = []
    for rel in files:
        path = Path(root) / rel
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(rel)
    print(f"{pid}: " + (f"oluşturuldu: {', '.join(created)}" if created else "dosyalar zaten var, atlandı."))
    return bool(created)


def apply_patch(root, relpath, pid, old, new, show_diff=True, max_diff_lines=40):
    path = Path(root) / relpath
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    norm_lines = [line.rstrip() for line in lines]
    if _find(norm_lines, _norm(new)):
        print(f"{pid}: {relpath} zaten yamalı, atlandı.")
        return False
    hits = _find(norm_lines, _norm(old))
    if len(hits) != 1:
        raise RuntimeError(f"{pid}: {relpath} içinde beklenen orijinal metin {len(hits)} kez bulundu "
                           f"(1 bekleniyordu). Repo beklenen commit'te mi?")
    start = hits[0]
    old_len = len(_norm(old))
    new_lines = new.strip("\n").split("\n")
    patched = lines[:start] + new_lines + lines[start + old_len:]
    path.write_text("\n".join(patched), encoding="utf-8")
    removed, added = old_len, len(new_lines)
    print(f"{pid}: {relpath} yamalandı (satır {start + 1}: -{removed} / +{added}).")
    if show_diff:
        diff = list(difflib.unified_diff([l.rstrip() for l in lines[start:start + old_len]],
                                         new_lines, lineterm="", n=0))[2:]
        for line in diff[:max_diff_lines]:
            print("   " + line)
        if len(diff) > max_diff_lines:
            print(f"   ... ({len(diff) - max_diff_lines} satır daha)")
    return True
