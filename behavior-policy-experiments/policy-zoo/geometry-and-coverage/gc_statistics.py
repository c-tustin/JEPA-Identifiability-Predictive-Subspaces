"""All 54 contrasts specified before the mechanism/exposure campaign."""
import json
import numpy as np
from common_gc import prior,FACTOR_SEEDS,ZOO_SEEDS,ZOO_POLICIES
from replication_stats import sign_test,holm

METRICS=('recovery','folds','planning')


def value(row,metric):
    if metric=='recovery':return row['metrics']['affine_log_r2']
    if metric=='folds':return -int(row['metrics']['folded'])
    return -row['planning']['shared']['goal_error']


def bootstrap(delta):
    delta=np.asarray(delta)
    rng=np.random.default_rng(20261106)
    return np.quantile(delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1),[.025,.975]).tolist()


def comparisons(rows):
    lookup={(r['condition'],r['seed']):r for r in rows}
    factor_names=[f'factor_A{a}_R{r}' for a in ('early','late') for r in ('early','late')]
    factor_seeds=[s for s in FACTOR_SEEDS if all((n,s) in lookup for n in factor_names)]
    zoo_names=[f'zoo_{p}_{a}' for p in ZOO_POLICIES for a in ('native','reset')]
    zoo_seeds=[s for s in ZOO_SEEDS if all((n,s) in lookup for n in zoo_names)]
    tests=[]
    def add(name,group,seeds,effects,definition):
        wins,losses,p=sign_test(effects)
        tests.append({'test':name,'group':group,'definition':definition,'n':len(effects),
                      'mean_gain':float(np.mean(effects)) if effects else None,
                      'ci95':bootstrap(effects) if effects else None,
                      'wins':wins,'losses':losses,'p':p,
                      'seed_effects':dict(zip(map(str,seeds),map(float,effects)))})
    for source in ('R','A'):
        for fixed in ('early','late'):
            for metric in METRICS:
                if source=='R':
                    a,b=f'factor_A{fixed}_Rearly',f'factor_A{fixed}_Rlate'
                else:
                    a,b=f'factor_Aearly_R{fixed}',f'factor_Alate_R{fixed}'
                delta=[value(lookup[a,s],metric)-value(lookup[b,s],metric) for s in factor_seeds]
                add(f'factor_{source}_at_{fixed}_{metric}','factor',factor_seeds,delta,
                    f'{a} minus {b}; positive means earlier loss source is beneficial')
    for metric in METRICS:
        delta=[]
        for s in factor_seeds:
            v=lambda a,r:value(lookup[f'factor_A{a}_R{r}',s],metric)
            delta.append((v('early','early')-v('early','late'))-(v('late','early')-v('late','late')))
        add(f'factor_interaction_{metric}','factor',factor_seeds,delta,
            '(Y_EE-Y_EL)-(Y_LE-Y_LL); positive is not automatically beneficial')
    for phase in ('early','late'):
        for metric in METRICS:
            delta=[]
            for s in factor_seeds:
                coupled=json.loads((prior.OUT/f'icm_early_to_{phase}_seed{s}.json').read_text())
                delta.append(value(lookup[f'factor_A{phase}_R{phase}',s],metric)-value(coupled,metric))
            add(f'independent_vs_coupled_{phase}_{metric}','factor',factor_seeds,delta,
                'independent-endpoint source cell minus original coupled continuation')
    for policy in ZOO_POLICIES:
        if policy!='random':
            for metric in ('recovery','planning'):
                delta=[value(lookup[f'zoo_{policy}_reset',s],metric)-value(lookup['zoo_random_reset',s],metric) for s in zoo_seeds]
                add(f'zoo_{policy}_vs_random_{metric}','zoo',zoo_seeds,delta,
                    'policy minus random at identical reset-state and regularizer exposure')
        for metric in METRICS:
            delta=[value(lookup[f'zoo_{policy}_reset',s],metric)-value(lookup[f'zoo_{policy}_native',s],metric) for s in zoo_seeds]
            add(f'zoo_{policy}_reset_vs_native_{metric}','zoo',zoo_seeds,delta,
                'common-reset alignment minus original native alignment, same regularizer stream')
    assert len(tests)==54
    for row,adjusted in zip(tests,holm([r['p'] for r in tests])):row['holm_p']=float(adjusted)
    return {'family_size':54,'factor_complete_seeds':factor_seeds,'zoo_complete_seeds':zoo_seeds,
            'complete':len(factor_seeds)==20 and len(zoo_seeds)==20,'tests':tests}
