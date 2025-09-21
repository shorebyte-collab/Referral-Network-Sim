# model.py
import pandas as pd
import numpy as np
from faker import Faker
import networkx as nx

fake = Faker()

# ---------- Core generator functions ----------

def gen_subscribers(n, t, pct_us=0.7, seed=None,
                    referral_ability_dist=None,
                    influencer_pct=0.0, influencer_high_range=(0.5,1.0),
                    influencer_low_range=(0.0,0.05),
                    seed_influencers_early=False):
    """
    Generate subscribers with referral_ability. A fraction influencer_pct are 'influencers'
    and get a much larger referral_ability sampled from influencer_high_range.
    If seed_influencers_early=True, influencers are assigned join_month=1 before others.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n+1)
    names = [fake.first_name() for _ in ids]

    # assign influencer flag
    if influencer_pct > 0:
        is_influencer = rng.random(n) < influencer_pct
    else:
        is_influencer = np.zeros(n, dtype=bool)

    # referral ability assignment
    referral_ability = np.zeros(n, dtype=float)
    for i in range(n):
        if is_influencer[i]:
            referral_ability[i] = rng.uniform(influencer_high_range[0], influencer_high_range[1])
        else:
            if referral_ability_dist is None:
                referral_ability[i] = rng.uniform(influencer_low_range[0], influencer_low_range[1])
            else:
                referral_ability[i] = referral_ability_dist(rng)

    # join months; optionally put influencers early
    if seed_influencers_early:
        join_month = np.empty(n, dtype=int)
        # influencers get join_month = 1
        join_month[is_influencer] = 1
        # non-influencers get uniform in 1..t
        join_month[~is_influencer] = rng.integers(1, t+1, size=(~is_influencer).sum())
    else:
        join_month = rng.integers(1, t+1, size=n)

    region = rng.choice(['US/Europe', 'ASEAN/Other'], size=n, p=[pct_us, 1-pct_us])
    df = pd.DataFrame({
        'subscriber_id': ids,
        'name': names,
        'join_month': join_month,
        'region': region,
        'referral_ability': referral_ability,
        'is_influencer': is_influencer
    })
    return df

def build_referral_links(sub_df, base_referral_prob=0.6, seed=None, bias_by_ability=False):
    """
    For each subscriber (ordered by join_month), decide whether they had a referrer and assign one
    chosen uniformly among earlier joiners or biased by referral_ability if requested.
    """
    rng = np.random.default_rng(seed)
    sub = sub_df.sort_values(['join_month','subscriber_id']).copy().reset_index(drop=True)
    referrer = []
    for i, row in sub.iterrows():
        month = row['join_month']
        earlier = sub[sub['join_month'] < month]
        if len(earlier)==0 or rng.random() > base_referral_prob:
            referrer.append(np.nan)
        else:
            if bias_by_ability:
                weights = earlier['referral_ability'].clip(lower=0.001).to_numpy()
                # normalize
                weights = weights / weights.sum()
                ref = int(rng.choice(earlier['subscriber_id'].values, p=weights))
            else:
                ref = int(rng.choice(earlier['subscriber_id'].values))
            referrer.append(ref)
    sub['referrer_id'] = referrer
    return sub

def compute_graph_maps(df):
    """Return direct_referrals_dict, total_downline_dict, networkx graph"""
    G = nx.DiGraph()
    for sid in df['subscriber_id']:
        G.add_node(int(sid))
    for _, r in df.iterrows():
        if pd.notna(r['referrer_id']):
            G.add_edge(int(r['referrer_id']), int(r['subscriber_id']))
    direct = {n: G.out_degree(n) for n in G.nodes()}
    total = {n: len(nx.descendants(G, n)) for n in G.nodes()}
    return direct, total, G

def assign_lifetimes(sub_df, churn_monthly_prob, t, seed=None):
    """
    Assign lifetime_months (>=1) sampled once per subscriber using geometric distribution.
    Cap lifetime to remaining months in t.
    """
    rng = np.random.default_rng(seed)
    lifetimes = []
    for _, r in sub_df.iterrows():
        max_len = int(t - int(r['join_month']) + 1)
        if churn_monthly_prob <= 0:
            L = max_len
        else:
            # geometric: number of months until churn, sample from geometric(p)
            L = int(rng.geometric(churn_monthly_prob))
            L = max(1, min(L, max_len))
        lifetimes.append(L)
    sub_df = sub_df.copy()
    sub_df['lifetime_months'] = lifetimes
    sub_df['end_month'] = sub_df['join_month'] + sub_df['lifetime_months'] - 1
    return sub_df

# ---------- Event factory ----------

def create_event_rows(sub_df, t,
                      price_us=14.99, price_other=2.99,
                      promo_months=(1,2), promo_rate=0.20,
                      tier_percents = [0.10,0.09,0.08,0.07,0.06]):
    """
    Returns DataFrame rows: one subscription payment per subscriber per month active.
    Includes uplines IDs (upline_l1..upline_l5) and commission_l1..l5.
    """
    rows = []
    ref_map = dict(zip(sub_df['subscriber_id'], sub_df['referrer_id']))
    for _, r in sub_df.iterrows():
        sid = int(r['subscriber_id'])
        jm = int(r['join_month'])
        em = int(r['end_month'])
        region = r['region']
        price = price_us if region == 'US/Europe' else price_other
        for month in range(jm, em+1):
            # find up to 5 uplines
            uplines = []
            u = ref_map.get(sid, np.nan)
            for _ in range(5):
                if pd.isna(u):
                    uplines.append(np.nan)
                    u = np.nan
                else:
                    uplines.append(int(u))
                    u = ref_map.get(int(u), np.nan)
            # direct promo
            l1_pct = promo_rate if month in promo_months else tier_percents[0]
            percent_list = [l1_pct] + list(tier_percents[1:])
            commissions = []
            for i in range(5):
                if pd.isna(uplines[i]):
                    commissions.append(0.0)
                else:
                    commissions.append(round(price * percent_list[i], 4))
            rows.append({
                'month': int(month),
                'subscriber_id': sid,
                'region': region,
                'price': float(price),
                'referrer_id': uplines[0],
                'upline_l1_id': uplines[0],
                'upline_l2_id': uplines[1],
                'upline_l3_id': uplines[2],
                'upline_l4_id': uplines[3],
                'upline_l5_id': uplines[4],
                'commission_l1': commissions[0],
                'commission_l2': commissions[1],
                'commission_l3': commissions[2],
                'commission_l4': commissions[3],
                'commission_l5': commissions[4],
            })
    events = pd.DataFrame(rows)
    events['commission_total'] = events[['commission_l1','commission_l2','commission_l3','commission_l4','commission_l5']].sum(axis=1)
    events['event_type'] = 'SUBSCRIPTION'
    events['bonus_amount'] = 0.0
    return events

# ---------- Bonuses ----------

def apply_bonuses(events, sub_df,
                  direct_bonus_map_us={5:30, 10:20}, total_bonus_map_us={500:500,1000:1000},
                  direct_bonus_map_other={5:5,10:3}, total_bonus_map_other={500:150,1000:300}):
    """
    Append bonus rows for direct and total downline thresholds. Each bonus paid once (first month hit).
    """
    # Prepare direct-referral events map
    df_direct = events[events['upline_l1_id'].notnull()][['month','subscriber_id','upline_l1_id']].copy()
    df_direct.columns = ['month','new_subscriber','upline']
    df_direct = df_direct.sort_values(['upline','month'])
    bonus_rows = []
    # direct referral bonuses
    grouped = df_direct.groupby('upline')
    for upline, group in grouped:
        group = group.reset_index(drop=True)
        u_reg = sub_df.loc[sub_df['subscriber_id']==upline,'region'].iloc[0] if len(sub_df.loc[sub_df['subscriber_id']==upline])>0 else 'US/Europe'
        direct_map = direct_bonus_map_us if u_reg=='US/Europe' else direct_bonus_map_other
        for th, amt in direct_map.items():
            if len(group) >= th:
                month_hit = int(group.loc[th-1,'month'])
                bonus_rows.append({
                    'month': month_hit,
                    'subscriber_id': int(upline),
                    'region': u_reg,
                    'price': 0.0,
                    'referrer_id': np.nan,
                    'upline_l1_id': np.nan,'upline_l2_id': np.nan,'upline_l3_id': np.nan,'upline_l4_id': np.nan,'upline_l5_id': np.nan,
                    'commission_l1': 0.0,'commission_l2':0.0,'commission_l3':0.0,'commission_l4':0.0,'commission_l5':0.0,
                    'commission_total': 0.0,
                    'event_type': 'BONUS_DIRECT',
                    'bonus_amount': float(amt)
                })
    # total downline bonuses: compute descendants
    _, total_map, G = compute_graph_maps(sub_df)
    join_map = dict(zip(sub_df['subscriber_id'], sub_df['join_month']))
    for node, total_count in total_map.items():
        if total_count == 0:
            continue
        descendants = list(nx.descendants(G, node))
        months = sorted([join_map[d] for d in descendants])
        u_reg = sub_df.loc[sub_df['subscriber_id']==node,'region'].iloc[0]
        total_map_conf = total_bonus_map_us if u_reg=='US/Europe' else total_bonus_map_other
        for th, amt in total_map_conf.items():
            if total_count >= th:
                month_hit = int(months[th-1])
                bonus_rows.append({
                    'month': month_hit,
                    'subscriber_id': int(node),
                    'region': u_reg,
                    'price': 0.0,
                    'referrer_id': np.nan,
                    'upline_l1_id': np.nan,'upline_l2_id': np.nan,'upline_l3_id': np.nan,'upline_l4_id': np.nan,'upline_l5_id': np.nan,
                    'commission_l1': 0.0,'commission_l2':0.0,'commission_l3':0.0,'commission_l4':0.0,'commission_l5':0.0,
                    'commission_total': 0.0,
                    'event_type': 'BONUS_DOWNLINE',
                    'bonus_amount': float(amt)
                })
    bonus_df = pd.DataFrame(bonus_rows)
    if not bonus_df.empty:
        events = pd.concat([events, bonus_df], ignore_index=True, sort=False).sort_values('month').reset_index(drop=True)
    return events

# ---------- Operating cost and summary ----------

def compute_operating_costs(month_index, base_cost=100.0, per_user_cost=0.01, scale_thresholds=None, active_users=None):
    """
    Compute operating cost for a month.
    - base_cost fixed per month
    - per_user_cost * active_users
    - optional scale_thresholds: list of (threshold, cost) to override when active_users >= threshold
    """
    cost = float(base_cost)
    if active_users is None:
        active_users = 0
    cost += float(per_user_cost) * float(active_users)
    if scale_thresholds:
        # scale_thresholds e.g. [(10000,10000)]
        for thr, c in sorted(scale_thresholds):
            if active_users >= thr:
                cost = float(c)  # override or set to c
    return cost

def summary_by_month(events, sub_df, t,
                     base_cost=100.0, per_user_cost=0.01, scale_thresholds=None):
    """
    Return DataFrame (month 1..t) with revenue, commissions, bonuses, operating_costs, net.
    active_users approximated by unique subscriber payments in that month.
    """
    df = events.copy()
    months = list(range(1, t+1))
    rows = []
    for m in months:
        subs = df[(df['month']==m) & (df['event_type']=='SUBSCRIPTION')]
        revenue = subs['price'].sum()
        commission = subs['commission_total'].sum()
        bonus = df[df['month']==m]['bonus_amount'].sum()
        active_users = subs['subscriber_id'].nunique()
        op_cost = compute_operating_costs(m, base_cost, per_user_cost, scale_thresholds, active_users)
        net = revenue - commission - bonus - op_cost
        rows.append({'month':m, 'revenue':revenue, 'commission':commission, 'bonus':bonus,
                     'operating_cost':op_cost, 'active_users':active_users, 'net':net})
    return pd.DataFrame(rows)

# ---------- Per-node earnings ----------

# model.py — replace node_earnings with this
def node_earnings(events, sub_df):
    """
    Return per-node summary including:
    - direct_referrals (count of upline_l1 occurrences)
    - indirect_referrals (total descendants)
    - direct_commissions, indirect_commissions (levels 2-5), total_bonuses_received
    - total_commissions_received, total_earnings
    """
    import networkx as nx

    # commissions at each level: sum by upline id
    comm_by_level = {}
    for level in range(1,6):
        ucol = f'upline_l{level}_id'
        ccol = f'commission_l{level}'
        grp = events[events[ucol].notnull()].groupby(ucol)[ccol].sum().reset_index().rename(columns={ucol:'subscriber_id', ccol:f'comm_l{level}'})
        comm_by_level[level] = grp

    # merge comm levels together
    if comm_by_level:
        comm_all = comm_by_level[1]
        for lvl in range(2,6):
            comm_all = comm_all.merge(comm_by_level[lvl], on='subscriber_id', how='outer')
        comm_all = comm_all.fillna(0)
        # totals
        comm_all['total_commissions_received'] = comm_all[[f'comm_l{i}' for i in range(1,6)]].sum(axis=1)
        comm_all['direct_commissions'] = comm_all.get('comm_l1', 0.0)
        comm_all['indirect_commissions'] = comm_all[[f'comm_l{i}' for i in range(2,6)]].sum(axis=1)
    else:
        comm_all = pd.DataFrame(columns=['subscriber_id','total_commissions_received','direct_commissions','indirect_commissions'])

    # bonuses received
    bonus = events[events['event_type'].str.contains('BONUS')].groupby('subscriber_id')['bonus_amount'].sum().reset_index().rename(columns={'bonus_amount':'total_bonuses_received'})

    # referral graph to compute counts (descendants)
    G = nx.DiGraph()
    for _, r in sub_df.iterrows():
        G.add_node(int(r['subscriber_id']))
    for _, r in sub_df.iterrows():
        if pd.notna(r['referrer_id']):
            G.add_edge(int(r['referrer_id']), int(r['subscriber_id']))

    # direct referrals count (out_degree)
    direct = {n: G.out_degree(n) for n in G.nodes()}
    # total indirect = descendants
    total_desc = {n: len(nx.descendants(G, n)) for n in G.nodes()}
    # indirect referrals = total_desc (since direct is included in out_degree separately)
    # build final DataFrame
    df_nodes = sub_df[['subscriber_id','name','region']].copy()
    df_nodes['direct_referrals'] = df_nodes['subscriber_id'].map(direct).fillna(0).astype(int)
    df_nodes['indirect_referrals'] = df_nodes['subscriber_id'].map(total_desc).fillna(0).astype(int)
    # merge commissions and bonuses
    df = df_nodes.merge(comm_all[['subscriber_id','total_commissions_received','direct_commissions','indirect_commissions']], on='subscriber_id', how='left').merge(bonus, on='subscriber_id', how='left')
    df = df.fillna(0)
    df['total_earnings'] = df['total_commissions_received'] + df['total_bonuses_received']
    # reorder columns
    cols = ['subscriber_id','name','region','direct_referrals','indirect_referrals',
            'direct_commissions','indirect_commissions','total_commissions_received','total_bonuses_received','total_earnings']
    # ensure columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]

# ---------- Monte Carlo ----------

def run_single_sim(params, seed=None):
    """
    Run one simulation and return events, summary_by_month, earnings df, graph
    params: dict containing keys: n,t,pct_us,base_referral_prob,churn_prob,price_us,price_other,...
    """
    sub = gen_subscribers(
                            params['n'],
                            params['t'],
                            pct_us=params.get('pct_us',0.7),
                            seed=seed,
                            referral_ability_dist=None,
                            influencer_pct=params.get('influencer_pct', 0.0),
                            influencer_high_range=params.get('influencer_high_range', (0.5,1.0)),
                            influencer_low_range=params.get('influencer_low_range', (0.0,0.05)),
                            seed_influencers_early=params.get('seed_influencers_early', False)
                        )

    sub = build_referral_links(sub, base_referral_prob=params.get('base_referral_prob',0.6),
                               seed=seed+1, bias_by_ability=params.get('bias_by_ability', False))
    sub = assign_lifetimes(sub, params.get('churn_prob', 0.05), params['t'], seed=seed+2)
    events = create_event_rows(sub, params['t'],
                               price_us=params.get('price_us',14.99),
                               price_other=params.get('price_other',2.99),
                               promo_months=params.get('promo_months',(1,2)),
                               promo_rate=params.get('promo_rate',0.20),
                               tier_percents=params.get('tier_percents',[0.10,0.09,0.08,0.07,0.06]))
    events = apply_bonuses(events, sub,
                           direct_bonus_map_us=params.get('direct_bonus_map_us',{5:30,10:20}),
                           total_bonus_map_us=params.get('total_bonus_map_us',{500:500,1000:1000}),
                           direct_bonus_map_other=params.get('direct_bonus_map_other',{5:5,10:3}),
                           total_bonus_map_other=params.get('total_bonus_map_other',{500:150,1000:300}))
    # add names
    events = events.merge(sub[['subscriber_id','name']], on='subscriber_id', how='left')
    summary = summary_by_month(events, sub, params['t'],
                               base_cost=params.get('base_cost',100.0),
                               per_user_cost=params.get('per_user_cost',0.01),
                               scale_thresholds=params.get('scale_thresholds', None))
    earnings = node_earnings(events, sub)
    _,_,G = compute_graph_maps(sub)
    return {'events':events, 'summary':summary, 'earnings':earnings, 'graph':G, 'subscribers':sub}

def run_monte_carlo(params, runs=100, seed=0, progress=None):
    """
    Run multiple simulations. Returns list of summaries and final net profits (and optionally time-series array).
    If progress is a callable, call progress(i, runs) to update UI.
    """
    rng = np.random.default_rng(seed)
    final_nets = []
    summaries = []
    ts_nets = []
    for i in range(runs):
        s = int(rng.integers(0, 2**31-1))
        out = run_single_sim(params, seed=s)
        summ = out['summary']
        total_net = summ['net'].sum()
        final_nets.append(total_net)
        summaries.append(summ)
        ts_nets.append(summ['net'].values)
        if callable(progress):
            progress(i+1, runs)
    df_nets = pd.DataFrame({'run':np.arange(1,len(final_nets)+1),'final_net':final_nets})
    ts_array = np.vstack(ts_nets)  # shape (runs, t)
    # compute percentiles per month
    perc_5 = np.percentile(ts_array, 5, axis=0)
    perc_50 = np.percentile(ts_array, 50, axis=0)
    perc_95 = np.percentile(ts_array, 95, axis=0)
    ts_percentiles = pd.DataFrame({'month': np.arange(1, params['t']+1),'p5':perc_5,'p50':perc_50,'p95':perc_95})
    return {'final_nets':df_nets, 'ts_percentiles':ts_percentiles, 'all_summaries':summaries}
