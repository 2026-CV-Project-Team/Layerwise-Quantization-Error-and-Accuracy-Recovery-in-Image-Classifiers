from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp_imagenet import pick_device
from src.exp_imagenet_propagation import Runner
from src.exp_imagenet_propagation_rounding import BIT, quant_fns
from src.imagenet_data import load_split, set_calib_seed, to_input
from src.quant import units as units_mod
CANDIDATES={
 ('resnet18','round'):('stem',69.00),
 ('resnet18','trunc_zero'):('layer3.0',47.40),
 ('resnet18','floor'):('layer4.0',0.25),
 ('deit_tiny','round'):('block9',64.65),
 ('deit_tiny','trunc_zero'):('block7',0.20),
 ('deit_tiny','floor'):('patch_embed',62.20),
}
class TestRunner(Runner):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.progress_label=''
 @torch.no_grad()
 def logits(self,w,a):
  self.q.apply(w,a)
  outs=[]
  n_batches=(len(self.x)+249)//250
  for bi,start in enumerate(range(0,len(self.x),250),1):
   outs.append(self.model(to_input(self.x[start:start+250],self.device)).float().cpu())
   if bi==1 or bi%10==0 or bi==n_batches:
    print(f'{self.progress_label} batch {bi}/{n_batches}',flush=True)
  return torch.cat(outs)
def run(name,mode,out_root):
 out=out_root/name/mode; out.mkdir(parents=True,exist_ok=True)
 marker=out/'complete.json'
 if marker.exists():
  print('skip complete',name,mode,flush=True); return
 set_calib_seed(0)
 old_w,old_a=units_mod.fake_quant_weight,units_mod.fake_quant_act
 units_mod.fake_quant_weight,units_mod.fake_quant_act=quant_fns(mode)
 runner=None; t0=time.time()
 try:
  runner=TestRunner(name,pick_device(),n_images=1)
  runner.x,runner.y=load_split(name,'test')
  runner.calib='seed0_minmax'
  runner.progress_label=f'{name}/{mode} FP32'
  runner.q.calibrate(runner.calib_batches())
  runner.ref=runner.logits(runner.off,runner.off)
  runner.record('fp32_test',runner.off,runner.off,runner.ref,bit=BIT)
  pd.DataFrame(runner.evals).to_csv(out/'evals.csv',index=False)
  runner.progress_label=f'{name}/{mode} uniform'
  runner.evaluate('uniform_test',[BIT]*runner.n,[BIT]*runner.n,bit=BIT)
  pd.DataFrame(runner.evals).to_csv(out/'evals.csv',index=False)
  candidate,rank_acc=CANDIDATES[(name,mode)]
  idx=next(i for i,u in enumerate(runner.units) if u.name==candidate)
  w,a=[BIT]*runner.n,[BIT]*runner.n
  w[idx]=a[idx]=None
  runner.progress_label=f'{name}/{mode} restore {candidate}'
  runner.evaluate('restore_fp32_test',w,a,bit=BIT,unit_idx=idx,unit=candidate,rank_restore_acc=rank_acc)
  pd.DataFrame(runner.evals).to_csv(out/'evals.csv',index=False)
  meta={'model':name,'mode':mode,'bit':BIT,'split':'test','images':len(runner.x),'calibration_seed':0,'calibration':'minmax','candidate':candidate,'candidate_selected_on':'rank','candidate_rank_accuracy':rank_acc,'device':str(pick_device()),'elapsed_seconds':round(time.time()-t0,2)}
  marker.write_text(json.dumps(meta,indent=2),encoding='utf-8')
  print('done',name,mode,'candidate',candidate,'elapsed_s',round(time.time()-t0,1),flush=True)
 finally:
  if runner is not None:
   try: runner.q.remove()
   except Exception: pass
  units_mod.fake_quant_weight,units_mod.fake_quant_act=old_w,old_a
def main():
 out_root=ROOT/'results'/'imagenet_propagation_rounding_test'
 for name in ('resnet18','deit_tiny'):
  for mode in ('round','trunc_zero','floor'):
   run(name,mode,out_root)
if __name__=='__main__': main()
