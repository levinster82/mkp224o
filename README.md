## mkp224o - vanity address generator for ed25519 onion services

This tool generates vanity ed25519 (hidden service version 3[^1][^2],
formely known as proposal 224) onion addresses.

### Requirements for building

* C99 compatible compiler (gcc and clang should work)
* libsodium (including headers)
* GNU make
* GNU autoconf (to generate configure script, needed only if not using release tarball)
* UNIX-like platform (currently tested in Linux and OpenBSD, but should
  also build under cygwin and msys2).

For debian-like linux distros, this should be enough to prepare for building:

```bash
apt install gcc libc6-dev libsodium-dev make autoconf
```

**For GPU acceleration (optional):** install the [NVIDIA CUDA Toolkit][CUDA].
The build system auto-detects `nvcc`; no extra configure flags are needed.
The binary links `libcudart` statically, so it runs on any machine — a CUDA
runtime is only required on the build machine, not the target.

On Debian/Ubuntu, follow the [CUDA download page][CUDA] for the network
installer, or use the distro packages as a quick start:

```bash
apt install nvidia-cuda-toolkit
```

On Fedora/RHEL, enable the NVIDIA CUDA repository and install:

```bash
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
dnf install cuda-toolkit
```

Adjust the repo URL for your Fedora/RHEL version — see the [CUDA download
page][CUDA] for the exact repo slug. After install, ensure `nvcc` is on
your PATH (typically `/usr/local/cuda/bin`) and re-run `./configure`.

### Building

Run `./autogen.sh` to generate a configure script, if there isn't one already.

Run `./configure` to generate a makefile.
On \*BSD platforms you may need to specify extra include/library paths:
`./configure CPPFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib"`.

On AMD64 platforms, you probably also want to pass something like
`--enable-amd64-51-30k` to the configure script invocation for faster key generation;
run `./configure --help` to see all available options.

Finally, `make` to start building (`gmake` in \*BSD platforms).

### GPU acceleration

If an NVIDIA GPU is present, mkp224o uses it automatically and will print a
line like:

```
GPU: NVIDIA GeForce RTX 3070 (sm_86, 46 SMs, 7833 MB)
using GPU acceleration
```

On machines without a GPU the binary falls back to CPU threads silently.
Use `-C` to force CPU-only mode even when a GPU is available.

GPU mode uses batch incremental point addition rather than per-key scalar
multiplication, which gives a large throughput advantage — an RTX 3070
delivers ~850 M keys/s versus ~25 M/s on a 16-core CPU (~34×).

#### Multiple GPUs

A single mkp224o process uses **one** GPU (the first CUDA device). To use
every card in a multi-GPU system, launch one process per card and pin each to
a different device with `CUDA_VISIBLE_DEVICES`:

```
CUDA_VISIBLE_DEVICES=0 ./mkp224o -S 3600 -d out0 myprefix &
CUDA_VISIBLE_DEVICES=1 ./mkp224o -S 3600 -d out1 myprefix &
CUDA_VISIBLE_DEVICES=2 ./mkp224o -S 3600 -d out2 myprefix &
wait
```

Each process seeds its own independent random starting points, so the
processes never duplicate each other's work, and combined throughput scales
linearly with the number of cards. Give each a separate output directory
(`-d`) so their stats and key output don't interleave.

This works with **mismatched card models** too: each process runs its own
auto-configuration against the card it is pinned to, sizing the batch to that
card's SM count and VRAM, and the processes run fully independently — a slower
card never holds back a faster one. Note that each process prints its own
`-S` statistics and its own ETA based on its own speed; the effective search
rate is the **sum** of the per-process speeds (so divide the displayed ETAs by
the number of identical cards, or sum the speeds for mixed cards).

### Usage

mkp224o needs one or more filters to work.
You may specify them as command line arguments,
eg `./mkp224o test`, or load them from file with `-f` switch.

It makes directories with secret/public keys and hostnames
for each discovered service. By default, the working directory is the current
directory, but that can be overridden with `-d` switch.

Use `-S <seconds>` to print periodic statistics. Each line shows elapsed
time, speed, expected match difficulty, ETA at 50% and 90% probability,
and total keys found:

```
> elapsed:      30s | speed:  855.2M/s | 1:34.4B | ETA 50%: 27s  90%: 1m32s | found: 0
```

Use `-C` to force CPU-only mode (disables GPU even if one is present).

Use `-h` switch to obtain all available options.

I highly recommend reading [OPTIMISATION.txt][OPTIMISATION] for
performance-related tips.

### FAQ and other useful info

* How do I generate address?

  Once compiled, run it like `./mkp224o neko`, and it will try creating
  keys for onions starting with "neko" in this example; use `./mkp224o
  -d nekokeys neko` to not litter current directory and put all
  discovered keys in directory named "nekokeys".

* How do I make tor use generated keys?

  Copy key folder (though technically only `hs_ed25519_secret_key` is required)
  to where you want your service keys to reside:

  ```bash
  sudo cp -r neko54as6d54....onion /var/lib/tor/nekosvc
  ```

  You may need to adjust ownership and permissions:

  ```bash
  sudo chown -R tor: /var/lib/tor/nekosvc
  sudo chmod -R u+rwX,og-rwx /var/lib/tor/nekosvc
  ```

  Then edit `torrc` and add new service with that folder.\
  After reload/restart tor should pick it up.

* How to generate addresses with `0-1` and `8-9` digits?

  Onion addresses use base32 encoding which does not include `0,1,8,9`
  numbers.\
  So no, that's not possible to generate these, and mkp224o tries to
  detect invalid filters containing them early on.

* How long is it going to take?

  It depends on your hardware and the length of the prefix. Use `-S 5`
  to print stats every 5 seconds — the output includes ETA at 50% and 90%
  probability based on your current speed, so you get a live estimate.\
  See [this issue][#27] for a detailed discussion.\
  As a rough guide at ~850 M keys/s (NVIDIA GeForce RTX 3070):

  | Prefix length | Difficulty  | ETA 50%   | ETA 90%   |
  |---------------|-------------|-----------|-----------|
  | 6 chars       | ~1.1B       | ~0.5s     | ~1.5s     |
  | 7 chars       | ~34.4B      | ~28s      | ~1m33s    |
  | 8 chars       | ~1.1T       | ~15m      | ~50m      |
  | 9 chars       | ~35.2T      | ~8h       | ~26h      |
  | 10 chars      | ~1.1P       | ~10d      | ~35d      |

  CPU-only (16 threads, ~25 M/s) is roughly 34× slower than the GPU
  column above. No promises — it is pure luck.

* Will this work with onionbalance?

  It appears that onionbalance supports loading usual
  `hs_ed25519_secret_key` key so it should work.

* Is there a docker image?

  Yes, if you do not wish to compile mkp224o yourself, you can use
  the `ghcr.io/cathugger/mkp224o` image like so:

  ```bash
  docker run --rm -it -v $PWD:/keys ghcr.io/cathugger/mkp224o:master -d /keys neko
  ```

### Acknowledgements & Legal

To the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the
public domain worldwide. This software is distributed without any
warranty.
You should have received a copy of the CC0 Public Domain Dedication
along with this software. If not, see [CC0][].

* `keccak.c` is based on [Keccak-more-compact.c][keccak.c]
* `ed25519/{ref10,amd64-51-30k,amd64-64-24k}` are adopted from
  [SUPERCOP][]
* `ed25519/ed25519-donna` adopted from [ed25519-donna][]
* Idea used in `worker_fast()` is stolen from [horse25519][]
* base64 routines and initial YAML processing work contributed by
  Alexander Khristoforov (heios at protonmail dot com)
* Passphrase-based generation code and idea used in `worker_batch()`
  contributed by [foobar2019][]

[OPTIMISATION]: ./OPTIMISATION.txt
[CUDA]: https://developer.nvidia.com/cuda-downloads
[#27]: https://github.com/cathugger/mkp224o/issues/27
[keccak.c]: https://github.com/XKCP/XKCP/blob/master/Standalone/CompactFIPS202/C/Keccak-more-compact.c
[CC0]: https://creativecommons.org/publicdomain/zero/1.0/
[SUPERCOP]: https://bench.cr.yp.to/supercop.html
[ed25519-donna]: https://github.com/floodyberry/ed25519-donna
[horse25519]: https://github.com/Yawning/horse25519
[foobar2019]: https://github.com/foobar2019
[^1]: https://spec.torproject.org/rend-spec/index.html
[^2]: https://gitlab.torproject.org/tpo/core/torspec/-/raw/main/attic/text_formats/rend-spec-v3.txt
