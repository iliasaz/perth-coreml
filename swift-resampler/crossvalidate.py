import numpy as np, subprocess, os, math, librosa, soxr
from scipy.signal import resample_poly, firwin, upfirdn
from scipy.signal._upfirdn import _output_len
RS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"rs","rs")
def swift(x, direction, kind):
    x=np.ascontiguousarray(x,dtype=np.float32)
    x.tofile("/tmp/in.f32")
    subprocess.run([RS,"/tmp/in.f32","/tmp/out.f32",direction,kind],check=True,capture_output=True)
    return np.fromfile("/tmp/out.f32",dtype=np.float32)
def cos(a,b):
    n=min(len(a),len(b)); a=a[:n].astype(np.float64); b=b[:n].astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def snr(ref,est):
    n=min(len(ref),len(est)); ref=ref[:n].astype(np.float64); est=est[:n].astype(np.float64)
    return float(10*np.log10((ref**2).sum()/(((ref-est)**2).sum()+1e-30)))
def _poly(x,up,down,h,hl):
    n_in=len(x); n_out=-(-n_in*up//down); npp=(down-hl%down)%down; npr=(hl+npp)//down
    hh=np.concatenate([np.zeros(npp),h*up]); post=0
    while _output_len(len(hh)+post,n_in,up,down)<n_out+npr: post+=1
    return upfirdn(np.concatenate([hh,np.zeros(post)]),x.astype(np.float64),up,down)[npr:npr+n_out]
HQ=np.load("/tmp/hq801.npy")

print("=== T1) SWIFT scipy-taps  vs  scipy.signal.resample_poly (BIT-COMPAT CLAIM) ===")
rng=np.random.default_rng(7); worst=0
for n in [1,2,3,7,31,769,1000,4001,24000,24001,24002,144000]:
    for direction,(up,down) in [("up",(4,3)),("down",(3,4))]:
        x=(rng.standard_normal(n)*0.3).astype(np.float32)
        s=swift(x,direction,"scipy")
        p=resample_poly(x,up,down).astype(np.float32)   # exactly what librosa res_type='polyphase' calls
        assert len(s)==len(p)==math.ceil(n*up/down), (n,direction,len(s),len(p))
        d=float(np.abs(s.astype(np.float64)-p.astype(np.float64)).max()); worst=max(worst,d)
        if n in (1,31,4001,144000):
            print(f"  n={n:7d} {direction:4s} len={len(s):7d} (=ceil) max|Δ|={d:.3e}  ulp={d/max(np.abs(p).max(),1e-9)/1.19e-7:5.2f}")
print(f"  WORST max|Δ| across all cases = {worst:.3e}   (float32 eps = 1.19e-07)")

print("\n=== T2) SWIFT hq-taps vs PYTHON soxr_hq (the apply path) ===")
src,sr=librosa.load(os.environ.get("PERTH_VOICE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Chatterbox-TTS-Server", "voices", "Abigail.wav"),sr=None,mono=True)
y24=librosa.resample(src,orig_sr=sr,target_sr=24000,res_type='soxr_vhq').astype(np.float32)[:24000*6]
su=swift(y24,"up","hq")
pu=librosa.resample(y24,orig_sr=24000,target_sr=32000,res_type='soxr_hq')
print(f"  24->32: len swift={len(su)} librosa={len(pu)}  cos={cos(pu,su):.9f}  SNR={snr(pu,su):.2f} dB")
sd=swift(su,"down","hq")
pd=librosa.resample(pu,orig_sr=32000,target_sr=24000,res_type='soxr_hq')
print(f"  32->24: len swift={len(sd)} librosa={len(pd)}  cos={cos(pd,sd):.9f}  SNR={snr(pd,sd):.2f} dB")
print(f"  round trip vs original: cos={cos(y24,sd):.9f}  SNR={snr(y24,sd):.2f} dB   (python soxr rt: SNR={snr(y24,pd):.2f} dB)")
print(f"  SWIFT hq vs PYTHON-reference-impl of same taps: max|Δ|={np.abs(su.astype(np.float64)-_poly(y24.astype(np.float64),4,3,HQ,400)).max():.3e}")

print("\n=== T3) EDGE CASES ===")
for n in [0,1,2,3,4,5,769]:
    x=(rng.standard_normal(n)*0.3).astype(np.float32) if n else np.zeros(0,np.float32)
    s=swift(x,"up","hq"); exp=math.ceil(n*4/3)
    p=_poly(x.astype(np.float64),4,3,HQ,400) if n else np.zeros(0)
    d=float(np.abs(s.astype(np.float64)-p).max()) if n else 0.0
    print(f"  n={n}: swift_len={len(s)} expected={exp} {'OK' if len(s)==exp else 'FAIL'}  max|Δ|={d:.2e}")

print("\n=== T4) IMPULSE / DC / linearity sanity ===")
imp=np.zeros(2001,np.float32); imp[1000]=1.0
si=swift(imp,"up","hq"); print(f"  impulse: peak {si.max():.6f} at {int(np.argmax(si))}, sum={si.sum():.6f} (should be ~4/3={4/3:.6f})")
dc=np.ones(4000,np.float32); sdc=swift(dc,"up","hq")
mid=sdc[600:-600]; print(f"  DC gain (interior): mean={mid.mean():.9f} std={mid.std():.3e}  (should be 1.0, 0)")
