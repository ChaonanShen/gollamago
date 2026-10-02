import time, torch
from solution import tl_softmax
# 覆盖任意 N/M，并覆盖 M=1、7、256、513、4096。
CASES=[(1,1),(3,7),(16,256),(17,513),(64,4096)]
def timed(fn, warmup=10, repeat=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); start=time.perf_counter()
    for _ in range(repeat): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-start)*1e6/repeat
print("N\tM\tcorrect\tTileLang(us)\tPyTorch(us)\tmax_abs_error")
# BLOCK_N设置成16效果很差，调小点，block多些
for n,m in CASES:
    x=torch.randn((n,m),device='cuda'); compile_start=time.perf_counter(); k=tl_softmax.compile(N=n,M=m,BLOCK_N=1,BLOCK_M=256); compile_ms=(time.perf_counter()-compile_start)*1e3
    out=k(x); ref=torch.softmax(x,dim=1); torch.testing.assert_close(out,ref,atol=2e-3,rtol=2e-3)
    print(f"{n}\t{m}\tPASS\t{timed(lambda:k(x)):.2f}\t\t{timed(lambda:torch.softmax(x,dim=1)):.2f}\t\t{(out-ref).abs().max().item():.3e}\tcompile={compile_ms:.1f}ms")
