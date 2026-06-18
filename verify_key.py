#!/usr/bin/env python3
"""Verify a mkp224o-generated key directory without running Tor.

Checks:
  1. sk -> pk: scalar from secret key produces the stored public key
  2. pk -> hostname: public key hashes to the stored .onion hostname

Usage:
  pip install pynacl                              (optional; speeds up check 1)
  ./mkp224o -n 1 -d /tmp/keytest testt
  python3 verify_key.py /tmp/keytest/testtXXXXXX.onion

Should print PASS on both lines for both CPU and GPU generated keys.
Exit code is 1 if either check fails, so it's scriptable too.
"""

import sys
import base64
import hashlib

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <key_dir>")

key_dir = sys.argv[1].rstrip("/")

try:
    with open(f"{key_dir}/hs_ed25519_secret_key", "rb") as f:
        sk_expanded = f.read()[32:]   # skip 32-byte header -> 64 bytes
    with open(f"{key_dir}/hs_ed25519_public_key", "rb") as f:
        pk_stored = f.read()[32:]     # skip 32-byte header -> 32 bytes
    with open(f"{key_dir}/hostname") as f:
        hostname = f.read().strip()
except FileNotFoundError as e:
    sys.exit(f"error: {e}")

if len(sk_expanded) != 64:
    sys.exit(f"error: secret key file wrong size (got {len(sk_expanded)+32}, expected 96)")
if len(pk_stored) != 32:
    sys.exit(f"error: public key file wrong size (got {len(pk_stored)+32}, expected 64)")

# --- Pure Python ed25519 scalar multiplication (no deps, ~1-2s, reference correct) ---
_p = 2**255 - 19
_d = (-121665 * pow(121666, _p-2, _p)) % _p

def _recoverx(y):
    x2 = (y*y-1) * pow(_d*y*y+1, _p-2, _p) % _p
    x = pow(x2, (_p+3)//8, _p)
    if (x*x-x2) % _p != 0:
        x = x * pow(2, (_p-1)//4, _p) % _p
    return x if x % 2 == 0 else _p - x

_By = 4 * pow(5, _p-2, _p) % _p
_Bx = _recoverx(_By)
_B  = (_Bx, _By, 1, _Bx * _By % _p)   # extended twisted Edwards (X,Y,Z,T)

def _pt_add(P, Q):
    X1, Y1, Z1, T1 = P
    X2, Y2, Z2, T2 = Q
    A = (Y1-X1) * (Y2-X2) % _p
    B = (Y1+X1) * (Y2+X2) % _p
    C = T1 * 2 * _d * T2 % _p
    D = Z1 * 2 * Z2 % _p
    E, F, G, H = (B-A)%_p, (D-C)%_p, (D+C)%_p, (B+A)%_p
    return (E*F%_p, G*H%_p, F*G%_p, E*H%_p)

def _scalarmult(k, P):
    R = (0, 1, 1, 0)   # neutral element
    while k:
        if k & 1: R = _pt_add(R, P)
        P = _pt_add(P, P)
        k >>= 1
    return R

def _compress(P):
    zi = pow(P[2], _p-2, _p)
    x, y = P[0]*zi%_p, P[1]*zi%_p
    out = bytearray(y.to_bytes(32, 'little'))
    out[31] |= (x & 1) << 7
    return bytes(out)
# ---------------------------------------------------------------------------------

# 1. Derive public key from scalar (first 32 bytes of expanded sk).
#    Pure Python is authoritative; PyNaCl is a fast cross-check when available.
scalar = int.from_bytes(sk_expanded[:32], 'little')
pk_py  = _compress(_scalarmult(scalar, _B))
py_ok  = pk_py == pk_stored

note = ""
try:
    import nacl.bindings
    pk_nacl = nacl.bindings.crypto_scalarmult_ed25519_base_noclamp(sk_expanded[:32])
    nacl_ok = pk_nacl == pk_stored
    if nacl_ok != py_ok:
        note = f"  [pure-py={'PASS' if py_ok else 'FAIL'}, nacl={'PASS' if nacl_ok else 'FAIL'} — DISAGREE]"
except Exception:
    pass

print(f"sk->pk:       {'PASS' if py_ok else 'FAIL'}{note}")

# 2. Rederive .onion hostname from public key (v3 onion address spec).
checksum = hashlib.sha3_256(b".onion checksum" + pk_stored + b"\x03").digest()[:2]
expected = base64.b32encode(pk_stored + checksum + b"\x03").decode().lower() + ".onion"
addr_ok  = hostname == expected
print(f"pk->hostname: {'PASS' if addr_ok else 'FAIL'}  ({hostname})")

if not py_ok or not addr_ok:
    sys.exit(1)
