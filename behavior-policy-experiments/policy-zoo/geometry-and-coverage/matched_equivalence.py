"""Post hoc omnibus differences and global paired equivalence for matched studies."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kempner-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kempner-cache')

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
ENDPOINTS=('recovery','shared_goal_error','native_goal_error')


def holm(pvalues):
    result=np.zeros(len(pvalues));largest=0.
    for rank,index in enumerate(np.argsort(pvalues)):
        largest=max(largest,(len(pvalues)-rank)*pvalues[index]);result[index]=min(1.,largest)
    return result.tolist()


def friedman_permutation(matrix,permutations=100000):
    n,k=matrix.shape
    ranks=stats.rankdata(matrix,axis=1,method='average')
    tie_total=0
    for row in matrix:
        counts=np.unique(row,return_counts=True)[1]
        tie_total+=np.sum(counts**3-counts)
    correction=1-tie_total/(n*(k**3-k))
    if correction<=1e-14:
        return {'statistic':0.,'p':1.,'permutations':0,'all_tied':True}
    center=n*(k+1)/2
    statistic=12/(n*k*(k+1))*np.sum((ranks.sum(0)-center)**2)/correction
    reference=stats.friedmanchisquare(*matrix.T).statistic
    assert np.isclose(statistic,reference,atol=1e-10,rtol=1e-10)
    rng=np.random.default_rng(20261107)
    exceed=0
    for start in range(0,permutations,2000):
        size=min(2000,permutations-start)
        order=np.argsort(rng.random((size,n,k)),axis=2)
        permuted=np.take_along_axis(ranks[None,:,:],order,axis=2)
        totals=permuted.sum(1)
        values=12/(n*k*(k+1))*np.sum((totals-center)**2,axis=1)/correction
        exceed+=int(np.sum(values>=statistic-1e-12))
    return {'statistic':float(statistic),'p':(exceed+1)/(permutations+1),
            'permutations':permutations,'exceedances':exceed,'all_tied':False,
            'assumption':'Within-seed policy-label exchangeability under the null.'}


def paired_equivalence(difference,margin):
    # One difference per paired seed (policy A minus B), not per goal or grid
    # point. TOST tests H0: mean <= -margin OR mean >= +margin against the
    # alternative that the mean lies strictly between these practical bounds.
    d=np.asarray(difference,dtype=float);n=len(d);mean=float(d.mean())
    sem=float(d.std(ddof=1)/np.sqrt(n))
    if sem==0:
        ci=[mean,mean]
        lower=0. if mean>-margin else 1.
        upper=0. if mean<margin else 1.
    else:
        # For a single pair at alpha=.05, both one-sided tests pass exactly when
        # the 90% paired-t interval lies inside (-margin, +margin). This assumes
        # the t approximation is appropriate for independent seed differences.
        width=stats.t.ppf(.95,n-1)*sem
        ci=[mean-width,mean+width]
        lower=float(stats.t.sf((mean+margin)/sem,n-1))
        upper=float(stats.t.cdf((mean-margin)/sem,n-1))
    # Separate sign-based robustness check: typical differences rather than
    # population means. It is not a distribution-free mean-equivalence test.
    successes_lower=int(np.sum(d>-margin))
    successes_upper=int(np.sum(d<margin))
    median_p=max(float(stats.binom.sf(successes_lower-1,n,.5)),float(stats.binom.sf(successes_upper-1,n,.5)))
    return {'mean_difference':mean,'mean_ci90':list(map(float,ci)),
            'mean_tost_p':max(lower,upper),'mean_tost_lower_p':lower,'mean_tost_upper_p':upper,
            'median_sign_tost_p':median_p,'n':n,
            'min_required_mean_margin':float(max(abs(ci[0]),abs(ci[1]))),'constant_paired_differences':sem==0}


def global_equivalence(matrix,names,margin):
    # Rows are seed blocks; columns are policies. Seven policies give 21 pairs.
    # Every pair must pass: the maximum component p-value implements this
    # intersection-union claim. Failure to detect a difference is insufficient.
    pairs=[]
    for i,j in itertools.combinations(range(len(names)),2):
        pairs.append({'policy_a':names[i],'policy_b':names[j],
                      **paired_equivalence(matrix[:,i]-matrix[:,j],margin)})
    return {'margin':margin,'mean_global_p':max(p['mean_tost_p'] for p in pairs),
            'median_global_p':max(p['median_sign_tost_p'] for p in pairs),
            'min_required_mean_margin':max(p['min_required_mean_margin'] for p in pairs),
            'mean_equivalent':all(p['mean_tost_p']<.05 for p in pairs),
            'median_equivalent':all(p['median_sign_tost_p']<.05 for p in pairs),'pairs':pairs}


def load_studies():
    old_root=HERE.parent/'matched-coverage/output'
    old_names=['reference','near_bottleneck','far_bottleneck','icm_shaped']
    old_seeds=list(range(100,110));old=[]
    for seed in old_seeds:
        block={}
        for name in old_names:
            r=json.loads((old_root/f'{name}_seed{seed}.json').read_text())
            block[name]={'recovery':r['representation']['log_distance_r2'],
                         'shared_goal_error':r['planning']['shared']['goal_error'],
                         'native_goal_error':r['planning']['native']['goal_error']}
        old.append(block)
    result=[{'name':'previous_stationary_four','description':'Four stationary surrogate policies; exact endpoint marginals/action counts matched.',
             'names':old_names,'seeds':old_seeds,'blocks':old,'complete':True}]
    new_names=['icm','random','count','rnd','disagreement','ou_actions_lp_w0.1','ou_actions_hp_w0.3']
    new_seeds=list(range(8,28));new=[];complete=[]
    for seed in new_seeds:
        paths=[OUT/f'zoo_{name}_reset_seed{seed}.json' for name in new_names]
        if not all(p.exists() for p in paths):continue
        block={}
        for name,path in zip(new_names,paths):
            r=json.loads(path.read_text())
            block[name]={'recovery':r['metrics']['affine_log_r2'],
                         'shared_goal_error':r['planning']['shared']['goal_error'],
                         'native_goal_error':r['planning']['native']['goal_error']}
        new.append(block);complete.append(seed)
    result.append({'name':'original_policies_common_reset','description':'Seven original policies; starts and regularizer exposure matched, successors/stationarity not forced.',
                   'names':new_names,'seeds':complete,'blocks':new,'complete':complete==new_seeds})
    return result


def figure(name,endpoint_results):
    fig,axes=plt.subplots(1,3,figsize=(16, max(5,len(endpoint_results['recovery']['equivalence']['pairs'])*.28)),sharey=True)
    for ax,endpoint in zip(axes,ENDPOINTS):
        result=endpoint_results[endpoint]['equivalence'];pairs=result['pairs'];margin=result['margin']
        for index,pair in enumerate(pairs):
            mean=pair['mean_difference'];lo,hi=pair['mean_ci90']
            color='#1b9e77' if lo>-margin and hi<margin else '#d95f02'
            ax.plot([lo,hi],[index,index],color=color,lw=2)
            ax.plot(mean,index,'o',color=color,ms=4)
        ax.axvline(-margin,color='gray',ls='--');ax.axvline(margin,color='gray',ls='--')
        ax.axvline(0,color='black',lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if endpoint == 'recovery':
            ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
        labels={'reference':'Ref','near_bottleneck':'Slow near','far_bottleneck':'Slow far','icm_shaped':'ICM-shaped','icm':'ICM','random':'Random','count':'Count','rnd':'RND','disagreement':'Disagree','ou_actions_lp_w0.1':'OU-LP','ou_actions_hp_w0.3':'OU-HP'}
        ax.set_yticks(range(len(pairs)),[labels.get(p['policy_a'],p['policy_a'])+' − '+labels.get(p['policy_b'],p['policy_b']) for p in pairs],fontsize=7)
        if ax is not axes[0]:ax.tick_params(labelleft=False)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set(title=endpoint.replace('_',' '),xlabel='Paired mean difference and 90% t interval')
    fig.suptitle('Global equivalence requires every interval inside the dashed margins; green = inside\nPointwise intervals, not simultaneous intervals; paired-t assumptions apply.')
    fig.tight_layout()
    fig.savefig(OUT/f'equivalence_{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'equivalence_{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def write_summary(results, recovery_margin, goal_margin):
    lines = ['# Are the matched policies equivalent?', '',
             '**Yes for mean recovery and shared-data planning, within the stated tolerances; equivalence is not established for policy-specific-data planning at those tolerances.**', '',
             'Use paired **TOST** (two one-sided tests). For each pair, test whether the mean seed-wise difference lies strictly between the negative and positive tolerance. Require every policy pair to pass. Seeds are the independent units, not individual goals.', '',
             f'Illustrative tolerances: **±{recovery_margin:g} R²** and **±{goal_margin:g} physical goal error**. A physical action moves 0.25, so the goal-error tolerance is {100 * goal_margin / .25:g}% of one action step. These tolerances were chosen after the experiments: this is exploratory evidence, not a prespecified confirmation.', '',
             'A tolerance is the largest difference we are willing to treat as practically unimportant. It is not estimated by a hypothesis test. Choosing it after seeing outcomes risks making the test too easy: a wider tolerance makes equivalence easier to establish. For a confirmatory claim, fix the tolerance first and evaluate a fresh, fixed-size seed cohort. Exact equality cannot be established from these finite runs.', '',
             '| Study | Recovery equivalence p | Shared-data planning equivalence p | Policy-specific-data planning equivalence p |',
             '| --- | ---: | ---: | ---: |']
    for result in results:
        values = [result['endpoints'][endpoint]['equivalence']['mean_global_p'] for endpoint in ENDPOINTS]
        label = 'Seven original policies, 20 seeds' if len(result['names']) == 7 else 'Four stationary surrogates, 10 seeds'
        lines.append(f'| {label} | {values[0]:.3g} | {values[1]:.3g} | {values[2]:.3g} |')
    latest = results[-1]
    lines += ['', f"For seven policies, the joint claim that **both recovery and shared-data planning are equivalent** has p = {latest['primary_joint_mean_p']:.3g}. The sign-based check also passes for typical paired differences (joint p = {latest['primary_joint_median_p']:.3g}); that check concerns typical differences, not the population mean.", '',
              'The global null is that at least one pair lies outside the tolerance. Its p-value is the largest pairwise TOST p-value: all pairs must pass. This intersection-union test does not need an extra Bonferroni correction for its single all-pairs equivalence claim.', '',
              'Paired mean-TOST uses a t approximation for seed differences. With 10/20 seeds and discrete planning errors, interpret this with the sign check and tolerance sensitivity. A larger p-value for policy-specific-data planning means insufficient evidence of equivalence, not proof of a difference.', '',
              'Why not chi-squared? These outcomes are continuous paired scores. A chi-squared test on bins would answer a different question about category frequencies, and a nonsignificant ordinary chi-squared test would still not establish equivalence.', '',
              'Scope: the seven-policy experiment matches starting observations and SIGReg exposure, not successor distributions or stationarity. Policy-specific predictor data means the reset transitions from each matched cell. These results do not establish equivalence of the original native-policy zoo or exact mathematical identifiability.', '',
              '[Focused identifiability and fold-risk tests](IDENTIFIABILITY_EQUIVALENCE.md) · [All pairwise intervals and tolerance sensitivity](MATCHED_EQUIVALENCE.md) · [Verification](EQUIVALENCE_VERIFICATION.json) · [Complementary difference tests](MATCHED_OMNIBUS.md)', '']
    (OUT / 'EQUIVALENCE_SUMMARY.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-margin',type=float,default=.001)
    parser.add_argument('--goal-margin',type=float,default=.005)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--permutations',type=int,default=100000)
    args=parser.parse_args();studies=load_studies();results=[];difference_rows=[]
    if args.require_complete:assert all(s['complete'] for s in studies)
    for study in studies:
        output={k:study[k] for k in ['name','description','names','seeds','complete']}
        output['endpoints']={}
        if study['complete']:
            for endpoint in ENDPOINTS:
                matrix=np.array([[block[name][endpoint] for name in study['names']] for block in study['blocks']])
                margin=args.recovery_margin if endpoint=='recovery' else args.goal_margin
                omnibus=friedman_permutation(matrix,args.permutations)
                equivalence=global_equivalence(matrix,study['names'],margin)
                grid=[.000001,.00001,.0001,.001,.005,.01,.02] if endpoint=='recovery' else [.005,.01,.025,.05,.1,.25,.5,1.]
                sensitivity=[{'margin':m,'mean_global_p':global_equivalence(matrix,study['names'],m)['mean_global_p']} for m in grid]
                output['endpoints'][endpoint]={'omnibus':omnibus,'equivalence':equivalence,
                                               'mean_scores':dict(zip(study['names'],matrix.mean(0).tolist())),
                                               'margin_sensitivity':sensitivity}
                difference_rows.append((study['name'],endpoint,omnibus))
            output['primary_joint_mean_p']=max(output['endpoints'][e]['equivalence']['mean_global_p'] for e in ('recovery','shared_goal_error'))
            output['primary_joint_median_p']=max(output['endpoints'][e]['equivalence']['median_global_p'] for e in ('recovery','shared_goal_error'))
            output['all_three_joint_mean_p']=max(output['endpoints'][e]['equivalence']['mean_global_p'] for e in ENDPOINTS)
            figure(study['name'],output['endpoints'])
        results.append(output)
    # Keep the full six-test difference family even while the second study is pending.
    padded=[r[2]['p'] for r in difference_rows]+[1.]*(6-len(difference_rows))
    for row,p in zip(difference_rows,holm(padded)):row[2]['holm_p']=float(p)
    payload={'post_hoc':True,'user_endorsed_margins':False,'margins':{'recovery':args.recovery_margin,'physical_goal_error':args.goal_margin},
             'scope':'Equivalence of all policy pairs in each matched design; not equivalence of the original native zoo.',
             'studies':results,'difference_family_size':6,
             'assumptions':'Paired-mean TOST uses t/normal approximation for seed differences. Median sign-TOST is distribution-free for typical differences. Global equivalence is an intersection-union test.'}
    (OUT/'MATCHED_EQUIVALENCE.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Did matched policies perform around the same?','',
           '**Equivalence test:** paired two one-sided tests (TOST), requiring every policy pair to pass. Unlike a nonsignificant difference test, this tests whether differences are inside stated tolerances. This is a post hoc analysis, not one of the campaign’s 54 planned tests.','',
           f"Practical margins: **±{args.recovery_margin:g} R²**, **±{args.goal_margin:g} physical goal error**. The latter is {args.goal_margin/.25:g} of an action step. These are illustrative tolerances, not a prespecified scientific definition of equivalence. The default R² tolerance is 0.1 percentage point and the goal-error tolerance is 2% of one action step. A sensitivity table includes the earlier, broader tolerances of 0.005 R² and 0.05 goal error. [Test protocol](../EQUIVALENCE_PROTOCOL.md).",'']
    for result in results:
        lines += ['## '+result['name'].replace('_',' '),'',result['description'],'']
        if not result['complete']:
            lines += [f"Pending: {len(result['seeds'])}/20 complete seed blocks. No partial-cohort hypothesis test is substituted.",''];continue
        lines += [f"Policies: {len(result['names'])}; paired seeds: {len(result['seeds'])}.",'',
                  '| Endpoint | Difference-test Holm p | Mean-equivalence global p | Mean equivalent? | Median-equivalence p | Smallest margin exceeding all 90% mean-interval endpoints |',
                  '| --- | ---: | ---: | --- | ---: | ---: |']
        for endpoint,row in result['endpoints'].items():
            eq=row['equivalence']
            lines += [f"| {endpoint} | {row['omnibus']['holm_p']:.5g} | {eq['mean_global_p']:.5g} | {'yes, under stated assumptions/margin' if eq['mean_equivalent'] else 'not established'} | {eq['median_global_p']:.5g} | {eq['min_required_mean_margin']:.5g} |"]
        lines += ['',f"**Joint primary mean equivalence (recovery AND shared-data reaching): p = {result['primary_joint_mean_p']:.5g}.** Both endpoints must pass; native-data reaching asks an additional question. Including all three endpoints gives p = {result['all_three_joint_mean_p']:.5g}.",'',
                  f"![Pairwise intervals](equivalence_{result['name']}.png)",'',
                  '| Endpoint | Margin | Global mean-equivalence p |',
                  '| --- | ---: | ---: |']
        for endpoint,row in result['endpoints'].items():
            for sensitivity in row['margin_sensitivity']:
                lines += [f"| {endpoint} | {sensitivity['margin']:g} | {sensitivity['mean_global_p']:.5g} |"]
        lines += ['']
    lines += ['## Interpret the tests carefully','',
              '- Friedman tests detect policy rank differences; a nonsignificant result does not establish equivalence.',
              '- Global mean equivalence requires every paired TOST to pass. Its p is the maximum component p. This is an intersection-union test, so the global conjunction does not need an extra pair-count correction. The 90% intervals are pointwise, not simultaneous.',
              '- Mean-TOST assumes approximately normal paired seed differences. Exactly constant observed differences use its degenerate limit, not a distribution-free population-mean guarantee. With only 10/20 seeds and discrete goal errors, the normal approximation needs caution. The median sign-TOST is an assumption-light check of typical differences, not proof about their mean.',
              '- Detectable differences and practical equivalence can coexist: a tiny consistent effect can be statistically detectable yet lie comfortably inside a practical margin.',
              '- All conclusions are conditional on the chosen margins, these actors, representation/predictor budgets and goal distribution. No equivalence claim extends to the entire original native-policy zoo.',
              '- The minimum supported margin is a descriptive precision threshold, not a reason to choose a margin after inspecting the data.','',
              '[Full pairwise differences, intervals, tests and sensitivity](MATCHED_EQUIVALENCE.json).','']
    (OUT/'MATCHED_EQUIVALENCE.md').write_text('\n'.join(lines))
    if all(s['complete'] for s in results):
        write_summary(results,args.recovery_margin,args.goal_margin)
    print(json.dumps([{s['name']:s.get('primary_joint_mean_p','pending')} for s in results]))


if __name__=='__main__':main()
