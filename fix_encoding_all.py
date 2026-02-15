#!/usr/bin/env python3
"""
MRARFAI 全量编码修复脚本 — 修复 UTF-8→CP1252→UTF-8 双重编码
用法: python fix_encoding_all.py [--dry-run]
"""
import os, re, sys

CP1252_MAP = {
    0x80: 0x20AC, 0x82: 0x201A, 0x83: 0x0192, 0x84: 0x201E,
    0x85: 0x2026, 0x86: 0x2020, 0x87: 0x2021, 0x88: 0x02C6,
    0x89: 0x2030, 0x8A: 0x0160, 0x8B: 0x2039, 0x8C: 0x0152,
    0x8E: 0x017D, 0x91: 0x2018, 0x92: 0x2019, 0x93: 0x201C,
    0x94: 0x201D, 0x95: 0x2022, 0x96: 0x2013, 0x97: 0x2014,
    0x98: 0x02DC, 0x99: 0x2122, 0x9A: 0x0161, 0x9B: 0x203A,
    0x9C: 0x0153, 0x9E: 0x017E, 0x9F: 0x0178,
}
U2B = {b: b for b in range(256)}
for bv, ucp in CP1252_MAP.items():
    U2B[ucp] = bv

def fix_mojibake_text(text):
    result, i = [], 0
    while i < len(text):
        cp = ord(text[i])
        if cp in U2B and cp > 0x7F:
            j, seq = i, bytearray()
            while j < len(text) and ord(text[j]) in U2B and ord(text[j]) > 0x7F:
                seq.append(U2B[ord(text[j])]); j += 1
            if len(seq) >= 2:
                for try_len in range(len(seq), 1, -1):
                    try:
                        d = seq[:try_len].decode('utf-8')
                        if any(ord(c) > 0xFF for c in d):
                            result.append(d); i += try_len; break
                    except UnicodeDecodeError:
                        continue
                else:
                    result.append(text[i]); i += 1
            else:
                result.append(text[i]); i += 1
        else:
            result.append(text[i]); i += 1
    return ''.join(result)

dry_run = '--dry-run' in sys.argv
project = os.path.dirname(os.path.abspath(__file__))

for fname in sorted(os.listdir(project)):
    if not fname.endswith('.py'): continue
    fpath = os.path.join(project, fname)
    with open(fpath, 'rb') as f:
        raw = f.read()
    if b'\xc3\xa6' not in raw and b'\xc3\xa5' not in raw and b'\xc3\xa8' not in raw:
        continue
    text = raw.decode('utf-8')
    fixed = fix_mojibake_text(text)
    cn_before = len(re.findall(r'[\u4e00-\u9fff]', text))
    cn_after = len(re.findall(r'[\u4e00-\u9fff]', fixed))
    delta = cn_after - cn_before
    if delta > 0:
        print(f"{'[DRY] ' if dry_run else ''}✅ {fname}: +{delta} Chinese chars")
        if not dry_run:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(fixed)
    else:
        print(f"⚪ {fname}: no change")

print("\nDone!" + (" (dry run, no files changed)" if dry_run else ""))
