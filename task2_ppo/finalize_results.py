from __future__ import annotations
import argparse
import csv
import json
import statistics as st
from pathlib import Path
from common.data import load_yaml, read_jsonl, repo_path
from task2_ppo.run_forks import condition_name


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csv_file(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def stability(rows):
    return {'updates': len(rows), 'generated_tokens': sum(r['generated_tokens'] for r in rows),
            **{k+'_std': st.pstdev(r[k] for r in rows) for k in ['policy_loss','value_loss','policy_grad_norm']},
            'max_abs_policy_loss': max(abs(r['policy_loss']) for r in rows),
            'mean_clip_fraction': st.mean(r['clip_fraction'] for r in rows)}


def build(config='configs/ppo.yaml', output_dir=None, plots=True, suffix=''):
    cfg = load_yaml(config); results = repo_path(cfg['results_dir'])
    assets = repo_path(output_dir) if output_dir else results/('report_assets'+suffix); assets.mkdir(parents=True, exist_ok=True)
    names = ['standard'] + [condition_name(s,float(v)) for s,key in [('clipping','clip_values'),('kl','kl_values')] for v in cfg[key]]
    runs = {}; reference = None
    protocol = ['seed','num_examples','max_prompt_length','max_response_length','temperature','top_p','do_sample']
    for name in names:
        train = read_jsonl(results/f'{name}{suffix}_train.jsonl'); summary = load(results/f'{name}{suffix}_summary.json')
        ev = load(results/f'{name}{suffix}_eval_summary.json'); gen = read_jsonl(results/f'{name}{suffix}_generations.jsonl')
        expected = int(cfg['updates'] if name=='standard' else cfg['fork_updates'])
        if len(train)!=expected or [r['update'] for r in train]!=list(range(1,expected+1)):
            raise ValueError(f'{name}: incomplete or duplicate updates')
        if sum(r['generated_tokens'] for r in train)!=summary['total_generated_tokens']:
            raise ValueError(f'{name}: token totals disagree')
        ids = [r['prompt_id'] for r in gen]
        if len(gen)!=ev['num_examples'] or len(set(ids))!=len(ids):
            raise ValueError(f'{name}: evaluation size or IDs disagree')
        modes = {r.get('likelihood_mode','legacy') for r in train}
        if len(modes)!=1: raise ValueError(f'{name}: mixed forward modes')
        signature = ([ev[k] for k in protocol],
                     [(r['prompt_id'],r['source_index'],r['prompt']) for r in gen],next(iter(modes)))
        if reference is None: reference=signature
        elif signature!=reference: raise ValueError(f'{name}: unmatched evaluation protocol or forward mode')
        expected_clip = float(name.removeprefix('clip_eps_'))/100 if name.startswith('clip_eps_') else float(cfg['clip_epsilon'])
        expected_kl = float(name.removeprefix('kl_beta_'))/100 if name.startswith('kl_beta_') else float(cfg['kl_beta'])
        if any(r['seed']!=cfg['seed'] or r['clip_epsilon']!=expected_clip or r['kl_beta']!=expected_kl for r in train):
            raise ValueError(f'{name}: training condition or seed disagrees with config')
        runs[name]=(train,summary,ev,gen)
    train,summary,ev,_ = runs['standard']
    for name in names[1:]:
        other = runs[name][0]
        if [(r['prompt_ids'],r['seed']) for r in other]!=[(r['prompt_ids'],r['seed']) for r in train[:len(other)]]:
            raise ValueError(f'{name}: unmatched training prompts')
    csv_file(assets/'standard_trajectory.csv',train)
    eval_keys = ['reward_model_score_mean','reward_model_score_std','kl_from_reference','entropy','mean_response_length','std_response_length','median_response_length','iqr_response_length','terminated_with_eos_fraction','truncated_fraction']
    csv_file(assets/'standard_heldout_eval.csv',[{k:ev[k] for k in eval_keys}])
    csv_file(assets/'standard_training_prompt_ids.csv',[{'update':r['update'],'prompt_position_in_update':i,'prompt_id':pid,'source_index':r['source_indices'][i]} for r in train for i,pid in enumerate(r['prompt_ids'])])
    runtime = {'sum_update_seconds':sum(r['update_seconds'] for r in train),'mean_update_seconds':st.mean(r['update_seconds'] for r in train),'std_update_seconds':st.pstdev(r['update_seconds'] for r in train),'total_generated_tokens':sum(r['generated_tokens'] for r in train)}
    save(assets/'standard_runtime_derived.json',runtime); save(assets/'standard_runtime_and_config.json',{**summary,'config':cfg})
    cached = load(results/'clipping_cached_summary.json')['conditions']; csv_file(assets/'clipping_cached_table.csv',cached)
    clipping=[]; trajectories=[]; stats={}
    for value in cfg['clip_values']:
        rows,_,e,_=runs[condition_name('clipping',float(value))]; c=next(c for c in cached if c['epsilon']==value); s=stability(rows)
        stats[f'eps_{int(round(value*100)):03d}']=s
        clipping.append({'epsilon':value,'cached_clip_fraction':c['clip_fraction'],'cached_affected_token_fraction':c['affected_token_fraction'],'cached_clipped_surrogate':c['clipped_surrogate'],
          **{'heldout_'+k:e[v] for k,v in {'reward':'reward_model_score_mean','reward_std':'reward_model_score_std','kl':'kl_from_reference','entropy':'entropy','mean_response_length':'mean_response_length','std_response_length':'std_response_length','median_response_length':'median_response_length','iqr_response_length':'iqr_response_length'}.items()},
          **{k:v for k,v in s.items() if k!='mean_clip_fraction'},'mean_online_clip_fraction':s['mean_clip_fraction']})
        trajectories += [{**r,'comparison_epsilon':value,'update_index':r['update']} for r in rows]
    save(assets/'clipping_stability_summary.json',stats)
    # Only the default destination replaces the original derived stability file.
    if output_dir is None: save(results/('clipping_stability_summary'+suffix+'.json'),stats)
    csv_file(assets/'clipping_report_table.csv',clipping); csv_file(assets/'clipping_training_trajectories.csv',trajectories)
    kl=[]; trajectories=[]
    for value in cfg['kl_values']:
        rows,_,e,_=runs[condition_name('kl',float(value))]
        mapping={'heldout_reward':'reward_model_score_mean','heldout_reward_std':'reward_model_score_std','heldout_kl':'kl_from_reference','entropy':'entropy','mean_response_length':'mean_response_length','std_response_length':'std_response_length','median_response_length':'median_response_length','iqr_response_length':'iqr_response_length','eos_fraction':'terminated_with_eos_fraction','truncated_fraction':'truncated_fraction'}
        kl.append({'kl_beta':value,**{k:e[v] for k,v in mapping.items()}})
        trajectories += [{**r,'comparison_kl_beta':value,'update_index':r['update']} for r in rows]
    csv_file(assets/'kl_pressure_report_table.csv',kl); csv_file(assets/'kl_pressure_training_trajectories.csv',trajectories)
    candidates=[]
    for x,y in zip(runs['kl_beta_000'][3],runs['kl_beta_020'][3]):
        delta=x['reward_model_score']-y['reward_model_score']
        candidates.append({'prompt_id':x['prompt_id'],'source_index':x['source_index'],'prompt':x['prompt'],'beta0_reward':x['reward_model_score'],'beta020_reward':y['reward_model_score'],'reward_delta_beta0_minus_beta020':delta,'absolute_reward_delta':abs(delta),'beta0_length':x['response_length'],'beta020_length':y['response_length'],'beta0_response':x['response'],'beta020_response':y['response']})
    candidates.sort(key=lambda r:r['absolute_reward_delta'],reverse=True)
    csv_file(assets/'qualitative_all_candidates.csv',candidates); save(assets/'qualitative_top20_candidates.json',candidates[:20])
    manifest={'generator':'python -m task2_ppo.finalize_results','seed':cfg['seed'],'run_names':[name+suffix for name in names],'likelihood_mode':train[0].get('likelihood_mode','legacy'),'standard_config':cfg,'standard_eval':ev,'standard_summary':summary,'standard_derived_runtime':runtime,'cached_clipping':cached,'clipping_comparison':clipping,'kl_pressure_comparison':kl,'qualitative_selection':'Selected examples are human-reviewed separately; reward rank is not a correctness judgement.'}
    save(assets/'task2_report_manifest.json',manifest)
    if plots:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def plot(name,rows,x,y):
            fig,ax=plt.subplots(figsize=(6.2,3.8)); ax.plot([r[x] for r in rows],[r[y] for r in rows],marker='o',markersize=3)
            ax.set_xlabel(x.replace('_',' ')); ax.set_ylabel(y.replace('_',' ')); ax.grid(alpha=.25); fig.tight_layout()
            fig.savefig(assets/f'{name}.png',dpi=160); plt.close(fig)
        for field,stem in [('raw_reward_mean','reward'),('kl_from_reference','kl'),('entropy','entropy'),('response_length_mean','response_length'),('clip_fraction','clip_fraction'),('policy_loss','policy_loss'),('value_loss','value_loss'),('policy_grad_norm','policy_grad_norm')]: plot('standard_'+stem,train,'update',field)
        for field,stem in [('clip_fraction','clip_fraction'),('affected_token_fraction','affected_fraction'),('clipped_surrogate','clipped_surrogate')]: plot('cached_'+stem,cached,'epsilon',field)
        for field,stem in [('heldout_reward','reward'),('heldout_kl','kl'),('entropy','entropy'),('mean_response_length','response_length')]: plot('kl_pressure_'+stem,kl,'kl_beta',field)
    print(f'Validated {len(names)} runs and regenerated evidence in {assets}.')
    return manifest


def main():
    ap=argparse.ArgumentParser(description='Reaggregate saved PPO results without loading models.')
    ap.add_argument('--config',default='configs/ppo.yaml'); ap.add_argument('--output-dir'); ap.add_argument('--suffix',default=''); ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args(); build(args.config,args.output_dir,not args.no_plots,args.suffix)


if __name__=='__main__': main()
