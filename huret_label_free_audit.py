"""Source-method reimplementation for the 12 Oct 2026 publication revision.
Not a byte-identical rerun of the unavailable July HuReT training code.
True-label margin is an explicitly INELIGIBLE oracle control, not a deployable feature.
All meta-prediction folds group the five variants of the same original image.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,platform
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
from scipy.spatial.distance import cdist
from scipy.special import softmax

OUT=Path(sys.argv[1]).resolve()
if OUT.exists():raise SystemExit('Output must be new; refusing to overwrite previous evidence')
OUT.mkdir(parents=True)
PROTOCOL={'kind':'independent_source_method_reimplementation','date':'2026-10-12','master_seed':20260715,
 'network':'64-32-16-head; ReLU; AdamW lr0.002 wd0.0001','max_epochs':140,'batch_size':128,'patience':10,
 'validation_fraction_of_training':0.18,'test_fraction':0.30,'seeds':list(range(5)),
 'noise_levels':[0,0.18,0.35,0.55,0.75],'meta_folds':5,'meta_group_unit':'original held-out image',
 'models':['nuisance','confidence','label_free_geometry','confidence_plus_geometry','oracle_true_label_margin'],
 'primary_comparison':'confidence_plus_geometry versus confidence; report AUC AND log loss; no selection across seeds',
 'interpretation':'Algorithmic stability only. No biology. Original July source/weights not available; new results do not silently overwrite them.',
 'centroids':'Fit-partition activations only; no validation or test labels used to fit centroids',
 'protocol_lock':'local pre-run file, not externally preregistered or timestamp-certified'}
(OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2))
(OUT/'PROTOCOL.sha256').write_text(hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest()+'\n')
start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
digits=load_digits();X=digits.data.astype(np.float32)/16.;Y=digits.target
ix=np.arange(len(Y));tr,te=train_test_split(ix,test_size=.30,random_state=20260715,stratify=Y)
fit,val=train_test_split(tr,test_size=.18,random_state=20260715,stratify=Y[tr])
np.savez_compressed(OUT/'split_indices.npz',fit=fit,validation=val,test=te)
class Network(nn.Module):
 def __init__(self,n):
  super().__init__();self.hidden=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU());self.head=nn.Linear(16,n)
 def forward(self,x):return self.head(self.hidden(x))
metrics=[];pred_rows=[];geometry_rows=[];parity=[]
noise_rng=np.random.default_rng(20261012)
# Fixed test perturbations shared across network seeds. Meta-fits remain grouped by original image.
noisy=np.concatenate([np.clip(X[te]+noise_rng.normal(0,sd,X[te].shape).astype(np.float32),0,1) for sd in PROTOCOL['noise_levels']])
noise=np.repeat(PROTOCOL['noise_levels'],len(te));groups=np.tile(te,len(PROTOCOL['noise_levels']));truth=np.tile(Y[te],len(PROTOCOL['noise_levels']))
np.savez_compressed(OUT/'noise_inputs.npz',images=noisy,noise=noise,groups=groups,truth_for_scoring_only=truth)
for seed in range(5):
 states={}
 for task in ['identity','parity','magnitude']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed)
  target=Y if task=='identity' else (Y%2 if task=='parity' else (Y>=5).astype(int))
  net=Network(10 if task=='identity' else 2);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
  tx=torch.from_numpy(X[fit]);ty=torch.tensor(target[fit],dtype=torch.long);vx=torch.from_numpy(X[val]);vy=torch.tensor(target[val],dtype=torch.long)
  best=float('inf');best_state=None;bad=0;history=[]
  for epoch in range(140):
   net.train()
   for batch in np.array_split(rng.permutation(len(fit)),int(np.ceil(len(fit)/128))):
    opt.zero_grad();loss=nn.functional.cross_entropy(net(tx[batch]),ty[batch]);loss.backward();opt.step()
   net.eval()
   with torch.no_grad():vl=nn.functional.cross_entropy(net(vx),vy).item()
   history.append(vl)
   if vl<best:best=vl;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};bad=0
   else:bad+=1
   if bad>=10:break
  net.load_state_dict(best_state)
  with torch.no_grad():
   h=net.hidden(torch.from_numpy(X[te])).numpy();logits=net(torch.from_numpy(X[te])).numpy()
  states[task]=h
  weight=OUT/f'{task}_seed{seed}.npz';np.savez_compressed(weight,**{k:v.numpy() for k,v in best_state.items()})
  reloaded=Network(10 if task=='identity' else 2)
  with np.load(weight,allow_pickle=False) as w:reloaded.load_state_dict({k:torch.from_numpy(w[k]) for k in w.files})
  reloaded.eval()
  with torch.no_grad():reload_logits=reloaded(torch.from_numpy(X[te])).numpy()
  parity.append({'task':task,'seed':seed,'exact':bool(np.array_equal(logits,reload_logits)),'max_abs_error':float(np.max(np.abs(logits-reload_logits)))})
  metrics.append({'task':task,'seed':seed,'accuracy':float(accuracy_score(target[te],logits.argmax(1))),'epochs':len(history),'validation_loss':best})
  if task!='identity':continue
  with torch.no_grad():hf=net.hidden(torch.from_numpy(X[fit])).numpy();hn=net.hidden(torch.from_numpy(noisy)).numpy();ln=net(torch.from_numpy(noisy)).numpy()
  cent=np.stack([hf[Y[fit]==c].mean(0) for c in range(10)])
  distances=cdist(hn,cent);prediction=ln.argmax(1);correct=(prediction==truth).astype(int)
  probability=softmax(ln.astype(float),axis=1);sorted_prob=np.sort(probability,axis=1)
  entropy=-np.sum(probability*np.log(np.clip(probability,1e-15,1)),axis=1)
  nearest=np.sort(distances,axis=1)
  d_pred=distances[np.arange(len(prediction)),prediction];d_other=distances.copy();d_other[np.arange(len(prediction)),prediction]=np.inf
  predicted_margin=d_other.min(1)-d_pred
  oracle_other=distances.copy();oracle_other[np.arange(len(truth)),truth]=np.inf
  oracle=oracle_other.min(1)-distances[np.arange(len(truth)),truth]
  nuisance=np.column_stack([noise,np.linalg.norm(hn,axis=1),hn.mean(1),hn.std(1)])
  confidence=np.column_stack([nuisance,sorted_prob[:,-1],entropy,sorted_prob[:,-1]-sorted_prob[:,-2]])
  geo=np.column_stack([noise,nearest[:,1]-nearest[:,0],nearest[:,0],predicted_margin])
  features={'nuisance':nuisance,'confidence':confidence,'label_free_geometry':geo,'confidence_plus_geometry':np.column_stack([confidence,geo[:,1:]]),'oracle_true_label_margin':np.column_stack([noise,oracle,nearest[:,0]])}
  folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261012).split(noisy,correct,groups))
  predictions={}
  for name,feat in features.items():
   pp=np.empty(len(correct));fold_ids=np.empty(len(correct),int)
   for fi,(tri,tei) in enumerate(folds):
    assert not set(groups[tri])&set(groups[tei]);assert len(np.unique(correct[tri]))==2
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',random_state=20261012))
    model.fit(feat[tri],correct[tri]);pp[tei]=model.predict_proba(feat[tei])[:,1];fold_ids[tei]=fi
   predictions[name]=pp
   pred_rows.append({'seed':seed,'model':name,'auc':float(roc_auc_score(correct,pp)),'log_loss':float(log_loss(correct,pp)),'n_original_images':len(te),'n_variants':len(correct),'fold_disjoint':True,'inference_eligible':name!='oracle_true_label_margin'})
  np.savez_compressed(OUT/f'meta_predictions_seed{seed}.npz',correct=correct,groups=groups,fold=fold_ids,**predictions)
  # Positive values favor adding label-free geometry. Mean within image avoids counting variants as subjects.
  def loss(p):return -(correct*np.log(np.clip(p,1e-15,1))+(1-correct)*np.log(np.clip(1-p,1e-15,1)))
  delta=(loss(predictions['confidence'])-loss(predictions['confidence_plus_geometry'])).reshape(5,len(te)).mean(0)
  np.save(OUT/f'image_logloss_delta_seed{seed}.npy',delta)
 for a,c in [('identity','parity'),('identity','magnitude'),('parity','magnitude')]:
  A=states[a]-states[a].mean(0);B=states[c]-states[c].mean(0)
  cka=np.linalg.norm(A.T@B,'fro')**2/(np.linalg.norm(A.T@A,'fro')*np.linalg.norm(B.T@B,'fro'))
  geometry_rows.append({'seed':seed,'task_a':a,'task_b':c,'linear_cka':float(cka)})
 print('seed',seed,'complete',round(time.time()-start,2),flush=True)
summary={'protocol':PROTOCOL,'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform()},'training':metrics,'meta_prediction':pred_rows,'cross_task_cka':geometry_rows,'weights_reload_checks':parity,'split_sizes':{'fit':len(fit),'validation':len(val),'test':len(te)},'elapsed_seconds':time.time()-start}
summary['all_weights_reload_exact']=all(p['exact'] for p in parity)
summary['source_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
summary['model_means']={name:{metric:float(np.mean([r[metric] for r in pred_rows if r['model']==name])) for metric in ['auc','log_loss']} for name in PROTOCOL['models']}
image_delta=np.stack([np.load(OUT/f'image_logloss_delta_seed{s}.npy') for s in range(5)]).mean(0)
br=np.random.default_rng(2026101201);boots=np.array([image_delta[br.integers(0,len(te),len(te))].mean() for _ in range(2000)])
summary['primary_image_bootstrap']={'mean_logloss_improvement':float(image_delta.mean()),'ci95':list(map(float,np.quantile(boots,[.025,.975]))),'resamples':2000,'unit':'original test image; seeds averaged first','limitation':'conditional on these trained models and OOF predictions; not full refit or population inference'}
(OUT/'RESULT.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'means':summary['model_means'],'primary':summary['primary_image_bootstrap'],'reload':summary['all_weights_reload_exact']},indent=2))
