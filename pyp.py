# burn_gpu_compat.py
import os, time, math, subprocess, torch

# --- Config via env ---
TARGET_SECONDS = int(os.getenv("DURATION", "60"))
MEM_FRACTION   = float(os.getenv("MEM_FRAC", "0.75"))
DTYPE          = os.getenv("DTYPE", "float16")  # "float16" or "float32"
PRINT_EVERY    = int(os.getenv("PRINT_EVERY", "5"))
SAFETY_GIB     = float(os.getenv("SAFETY_GIB", "0.5"))
# ----------------------

assert torch.cuda.is_available(), "CUDA not available"
device = "cuda"

def get_free_total_gib():
    out = subprocess.run(
        ["nvidia-smi","--query-gpu=memory.free,memory.total","--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, encoding="utf-8", check=True
    ).stdout.strip()
    free_mb, total_mb = map(int, out.split(","))
    return free_mb/1024.0, total_mb/1024.0

free_gib, total_gib = get_free_total_gib()
print("GPU total: %.2f GiB; free: %.2f GiB" % (total_gib, free_gib))

dtype = torch.float16 if DTYPE.lower() in ("fp16","float16","half") else torch.float32
bytes_per_elem = 2 if dtype is torch.float16 else 4

usable_gib = max(0.0, free_gib - SAFETY_GIB) * MEM_FRACTION
usable_bytes = int(usable_gib * (1024**3))
if usable_bytes <= 0:
    raise SystemExit("Not enough free memory to allocate working tensors.")

# 3*N^2*bytes_per_elem <= usable_bytes  ->  N <= sqrt(usable_bytes/(3*bytes_per_elem))
N = int(math.sqrt(usable_bytes / (3.0 * bytes_per_elem)))
N = max(128, (N // 128) * 128)  # round for GEMM efficiency

needed_bytes = 3 * (N**2) * bytes_per_elem
needed_gib = needed_bytes / (1024**3)
print("Planning matrices: N=%d (dtype=%s) needing ~%.2f GiB for A,B,C" % (N, str(dtype), needed_gib))

# Allocate working tensors
try:
    A = torch.randn((N, N), device=device, dtype=dtype)
    B = torch.randn((N, N), device=device, dtype=dtype)
    C = torch.zeros((N, N), device=device, dtype=dtype)
except RuntimeError as e:
    raise SystemExit("Allocation failed (lower MEM_FRAC or raise SAFETY_GIB): %s" % e)

# Warmup (triggers kernel selection)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    # C = A @ B  (no extra allocs)
    C.addmm_(A, B, beta=0.0, alpha=1.0)
torch.cuda.synchronize()
print("Warmup done in %.2fs" % (time.time() - t0))

# Sustained burn for TARGET_SECONDS
start = time.time()
last_print = start
iters = 0
flops_per_gemm = 2.0 * (N**3)  # ~2*N^3

while True:
    # Unroll to reduce Python overhead
    C.addmm_(A, B, beta=0.0, alpha=1.0)
    C.addmm_(A, B, beta=0.0, alpha=1.0)
    C.addmm_(A, B, beta=0.0, alpha=1.0)
    C.addmm_(A, B, beta=0.0, alpha=1.0)
    iters += 4

    now = time.time()
    if now - last_print >= PRINT_EVERY:
        torch.cuda.synchronize()
        elapsed = now - start
        tflops = (flops_per_gemm * iters) / elapsed / 1e12
        print("Elapsed %.1fs, iterations %d, approx %.2f TFLOP/s" % (elapsed, iters, tflops))
        last_print = now

    if now - start >= TARGET_SECONDS:
        break

torch.cuda.synchronize()
print("Done. Holding tensors; check nvidia-smi.")
try:
    input("Press Enter to free memory and exit...")
except EOFError:
    pass

del A, B, C
torch.cuda.empty_cache()
print("Freed.")


# watch -n 0.5 nvidia-smi
# DURATION=60 MEM_FRAC=0.9 DTYPE=float16 python3 pyp.py
