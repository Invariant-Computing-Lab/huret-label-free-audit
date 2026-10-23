from pathlib import Path
import json,numpy as np
R=Path(__file__).parent
if __name__=='__main__':
 p=R/'HURET_20261013_REPLAY1';count=0
 for task in ['identity','parity','magnitude']:
  for seed in range(5):
   with np.load(p/f'{task}_seed{seed}.npz',allow_pickle=False) as w:
    assert w['hidden.0.weight'].shape==(32,64) and w['hidden.2.weight'].shape==(16,32) and all(np.isfinite(w[k]).all() for k in w.files)
   count+=1
 Path('replay_result.json').write_text(json.dumps({'weights_checked':count,'training_rerun':False,'scope':'Archive structure and finite arrays, not predictor efficacy'}));print('PASS',count)
