# app.py (patched: pass network_table into draw_pyvis_graph for richer tooltips)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import run_single_sim, run_monte_carlo, node_earnings
import networkx as nx
import streamlit.components.v1 as components
import tempfile
import os

monte_carlo_mode: bool = False   # overwritten by sidebar checkbox
run_btn: bool = False

st.set_page_config(layout="wide", page_title="Zen360Life Data Factory + Monte Carlo (Cached, PyVis, KPIs)")

st.title("Zen360Life — Data Factory + Monte Carlo Dashboard (Cached + PyVis + KPIs)")

# ---------------- Sidebar ----------------
# ---------- Sidebar (with Presets) ----------
# ---------- Sidebar (with Presets, fixed unique keys) ----------
with st.sidebar:
    st.header("Simulation Controls")

    # Presets dictionary
    PRESETS = {
        "Custom (no preset)": {},
        "Best case": {
            'n':5000,'t':24,'pct_us':0.8,'base_referral_prob':0.9,'churn_prob':0.03,'bias_by_ability':True,
            'influencer_pct':0.02,'influencer_high_range':(0.8,1.5),'influencer_low_range':(0.01,0.05),'seed_influencers_early':True,
            'base_cost':500,'per_user_cost':0.005,'scale_thresholds':"20000:30000"
        },
        "Worst case": {
            'n':2000,'t':12,'pct_us':0.2,'base_referral_prob':0.1,'churn_prob':0.5,'bias_by_ability':False,
            'influencer_pct':0.0,'influencer_high_range':(0.5,1.0),'influencer_low_range':(0.0,0.05),'seed_influencers_early':False,
            'base_cost':2000,'per_user_cost':0.05,'scale_thresholds':""
        },
        "Steady state": {
            'n':3000,'t':18,'pct_us':0.5,'base_referral_prob':0.4,'churn_prob':0.12,'bias_by_ability':False,
            'influencer_pct':0.005,'influencer_high_range':(0.3,0.6),'influencer_low_range':(0.0,0.05),'seed_influencers_early':False,
            'base_cost':500,'per_user_cost':0.02,'scale_thresholds':"10000:10000"
        },
        "Slow growth": {
            'n':2500,'t':24,'pct_us':0.6,'base_referral_prob':0.35,'churn_prob':0.08,'bias_by_ability':False,
            'influencer_pct':0.005,'influencer_high_range':(0.4,0.8),'influencer_low_range':(0.0,0.05),'seed_influencers_early':False,
            'base_cost':300,'per_user_cost':0.01,'scale_thresholds':""
        },
        "Steady growth": {
            'n':4000,'t':24,'pct_us':0.65,'base_referral_prob':0.55,'churn_prob':0.06,'bias_by_ability':True,
            'influencer_pct':0.01,'influencer_high_range':(0.5,1.0),'influencer_low_range':(0.0,0.05),'seed_influencers_early':True,
            'base_cost':400,'per_user_cost':0.01,'scale_thresholds':"15000:15000"
        },
        "Viral growth": {
            'n':3000,'t':12,'pct_us':0.7,'base_referral_prob':0.9,'churn_prob':0.04,'bias_by_ability':True,
            'influencer_pct':0.03,'influencer_high_range':(1.0,2.0),'influencer_low_range':(0.0,0.03),'seed_influencers_early':True,
            'base_cost':300,'per_user_cost':0.008,'scale_thresholds':"5000:8000,20000:25000"
        },
        "Pyramid growth": {
            'n':2000,'t':24,'pct_us':0.6,'base_referral_prob':0.8,'churn_prob':0.07,'bias_by_ability':True,
            'influencer_pct':0.01,'influencer_high_range':(1.0,3.0),'influencer_low_range':(0.0,0.02),'seed_influencers_early':True,
            'base_cost':200,'per_user_cost':0.01,'scale_thresholds':""
        },
        "Inverse pyramid": {
            'n':5000,'t':18,'pct_us':0.6,'base_referral_prob':0.7,'churn_prob':0.06,'bias_by_ability':False,
            'influencer_pct':0.0,'influencer_high_range':(0.2,0.5),'influencer_low_range':(0.05,0.2),'seed_influencers_early':False,
            'base_cost':300,'per_user_cost':0.01,'scale_thresholds':""
        },
        "Equally distributed": {
            'n':4000,'t':18,'pct_us':0.5,'base_referral_prob':0.6,'churn_prob':0.08,'bias_by_ability':False,
            'influencer_pct':0.0,'influencer_high_range':(0.0,0.05),'influencer_low_range':(0.02,0.08),'seed_influencers_early':False,
            'base_cost':400,'per_user_cost':0.01,'scale_thresholds':""
        },
        "Lopsided growth": {
            'n':2500,'t':18,'pct_us':0.7,'base_referral_prob':0.6,'churn_prob':0.07,'bias_by_ability':True,
            'influencer_pct':0.005,'influencer_high_range':(1.5,3.0),'influencer_low_range':(0.0,0.03),'seed_influencers_early':True,
            'base_cost':200,'per_user_cost':0.01,'scale_thresholds':""
        }
    }

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0, key="preset_select")
    preset = PRESETS.get(preset_name, {})

    # helper to read preset or fallback to default
    def p(key, default):
        return preset.get(key, default)

    # Use preset values as defaults for widgets below (keys ensure uniqueness)
    n = st.number_input("Number of subscribers (n)", min_value=10, max_value=50000, value=int(p('n', 2000)), step=10, key="n_input")
    t = st.number_input("Time horizon (months, t)", min_value=1, max_value=60, value=int(p('t', 12)), key="t_input")
    pct_us = st.slider("Proportion US/Europe (price $14.99)", 0.0, 1.0, value=float(p('pct_us', 0.7)), key="pct_us_slider")
    base_referral_prob = st.slider("Probability a joiner had a referrer", 0.0, 1.0, value=float(p('base_referral_prob', 0.6)), key="ref_prob_slider")
    churn_prob = st.slider("Monthly churn probability (used to sample lifetime)", 0.0, 0.5, value=float(p('churn_prob', 0.05)), key="churn_slider")
    bias_by_ability = st.checkbox("Bias referrer selection by referral ability", value=bool(p('bias_by_ability', False)), key="bias_checkbox")
    seed = int(st.number_input("Random seed (change to reproduce)", value=int(p('seed', 42)), step=1, key="seed_input"))

    st.markdown("---")
    st.header("Operating Costs")
    base_cost = st.number_input("Base monthly cost (USD)", min_value=0.0, value=float(p('base_cost', 100.0)), key="base_cost_input")
    per_user_cost = st.number_input("Per-active-user monthly cost (USD)", min_value=0.0, value=float(p('per_user_cost', 0.01)), key="per_user_cost_input")
    st.markdown("Scale thresholds (comma-separated as threshold:cost), e.g. 10000:10000")
    scale_input = st.text_input("Scale thresholds", value=str(p('scale_thresholds', "10000:10000")), key="scale_input")
    scale_thresholds = None
    try:
        if scale_input and isinstance(scale_input, str):
            pairs = [pstr.strip() for pstr in scale_input.split(',') if ':' in pstr]
            scale_thresholds = []
            for pstr in pairs:
                thr,cost = pstr.split(':')
                scale_thresholds.append((int(thr.strip()), float(cost.strip())))
    except Exception:
        scale_thresholds = None

    st.markdown("---")
    st.header("Influencers (optional)")
    influencer_pct = st.slider("Influencer % of universe", 0.0, 0.1, value=float(p('influencer_pct', 0.01)), step=0.005, key="infl_pct_slider",
                            help="Fraction of subscribers who are influencers (high referral ability).")
    seed_influencers_early = st.checkbox("Seed influencers in month 1 (early eligibility)", value=bool(p('seed_influencers_early', True)), key="seed_early_checkbox")
    influencer_high_min = st.number_input("Influencer ability min", min_value=0.01, max_value=10.0, value=float(p('influencer_high_range', (0.5,1.0))[0]), step=0.01, key="infl_high_min")
    influencer_high_max = st.number_input("Influencer ability max", min_value=0.01, max_value=10.0, value=float(p('influencer_high_range', (0.5,1.0))[1]), step=0.01, key="infl_high_max")
    influencer_low_min = st.number_input("Normal ability min", min_value=0.0, max_value=1.0, value=float(p('influencer_low_range', (0.0,0.05))[0]), step=0.01, key="infl_low_min")
    influencer_low_max = st.number_input("Normal ability max", min_value=0.0, max_value=1.0, value=float(p('influencer_low_range', (0.0,0.05))[1]), step=0.01, key="infl_low_max")

    st.markdown("---")
    st.header("Monte Carlo")
    monte_carlo_mode = st.checkbox("Enable Monte Carlo mode (run many simulations)", value=False, key="mc_checkbox")
    runs = int(st.number_input("Monte Carlo runs", min_value=10, max_value=2000, value=int(p('runs', 200)), step=10, key="mc_runs_input"))

    st.markdown("---")
    run_btn = st.button("Generate / Refresh", key="run_button")

# ---------- cached wrappers ----------
@st.cache_data(show_spinner=False)
def cached_run_single(params_key):
    params, seed_local = params_key
    return run_single_sim(params, seed=seed_local)

@st.cache_data(show_spinner=False)
def cached_run_monte(params_key):
    params, runs_local, seed_local = params_key
    return run_monte_carlo(params, runs_local, seed_local)

# ---------- helper: build params dict ----------
params = {
    'n': n, 't': t, 'pct_us': pct_us, 'base_referral_prob': base_referral_prob,
    'churn_prob': churn_prob, 'price_us':14.99, 'price_other':2.99,
    'promo_months': (1,2), 'promo_rate':0.20,
    'tier_percents':[0.10,0.09,0.08,0.07,0.06],
    'direct_bonus_map_us':{5:30,10:20}, 'total_bonus_map_us':{500:500,1000:1000},
    'direct_bonus_map_other':{5:5,10:3}, 'total_bonus_map_other':{500:150,1000:300},
    'base_cost': base_cost, 'per_user_cost': per_user_cost, 'scale_thresholds': scale_thresholds,
    'bias_by_ability': bias_by_ability,
    'influencer_pct': influencer_pct,
    'influencer_high_range': (influencer_high_min, influencer_high_max),
    'influencer_low_range': (influencer_low_min, influencer_low_max),
    'seed_influencers_early': seed_influencers_early,
    }

# ---------- PyVis rendering helper ----------
def draw_pyvis_graph(G, subscribers_df, earnings_df, height="600px", width="100%"):
    """
    Build interactive PyVis graph (safe filename use). If PyVis fails, fall back to Plotly HTML.
    earnings_df should be the node-level table (output of node_earnings) containing:
      subscriber_id, direct_referrals, indirect_referrals, direct_commissions, indirect_commissions, total_bonuses_received, total_earnings
    Returns path to generated HTML (safe for embedding).
    """
    import tempfile, os
    import plotly.graph_objects as go

    # lookups
    name_map = dict(zip(subscribers_df['subscriber_id'], subscribers_df['name']))
    net_map = dict(zip(earnings_df['subscriber_id'], zip(
            earnings_df['direct_referrals'],
            earnings_df['indirect_referrals'],
            earnings_df['direct_commissions'],
            earnings_df['indirect_commissions'],
            earnings_df['total_bonuses_received'],
            earnings_df['total_earnings']
        )))
    # size map uses total_earnings
    size_map = dict(zip(earnings_df['subscriber_id'], earnings_df['total_earnings']))

    # Try PyVis (use only filename when writing HTML)
    try:
        from pyvis.network import Network
        net = Network(height=height, width=width, notebook=False, directed=True)
        net.barnes_hut()
        for n in G.nodes():
            nm = name_map.get(n, str(n))
            total_earn = float(size_map.get(n, 0.0))
            dref, indref, dcomm, icomm, bonus_amt, total_earn2 = net_map.get(n, (0,0,0.0,0.0,0.0,0.0))
            size = 8 + min(40, total_earn/5) if total_earn>0 else 8
            title = (
                f"{n} - {nm}<br>"
                f"Direct referrals: {dref}<br>"
                f"Indirect referrals: {indref}<br>"
                f"Direct earnings: ${dcomm:,.2f}<br>"
                f"Indirect earnings: ${icomm:,.2f}<br>"
                f"Bonuses: ${bonus_amt:,.2f}<br>"
                f"Total earnings: ${total_earn2:,.2f}"
            )
            net.add_node(n, label=str(n), title=title, size=size)
        for u, v in G.edges():
            net.add_edge(u, v)

        tmp_dir = tempfile.gettempdir()
        filename = f"pyvis_{abs(hash(str(sorted(G.nodes())))) % (10**8)}.html"
        net.write_html(filename, open_browser=False, notebook=False)
        full_path = os.path.join(tmp_dir, filename)
        if not os.path.exists(full_path):
            full_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(full_path):
            return full_path
    except Exception as e_pyvis:
        print("PyVis failed:", repr(e_pyvis))

    # Plotly fallback (robust layout)
    try:
        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception as e_layout:
            print("spring_layout failed, falling back to random_layout:", repr(e_layout))
            pos = nx.random_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = []
        node_y = []
        texts = []
        sizes = []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            nm = name_map.get(n, str(n))
            dref, indref, dcomm, icomm, bonus_amt, total_earn2 = net_map.get(n, (0,0,0.0,0.0,0.0,0.0))
            texts.append(
                f"{n} - {nm}<br>"
                f"Direct referrals: {dref}<br>"
                f"Indirect referrals: {indref}<br>"
                f"Direct earnings: ${dcomm:,.2f}<br>"
                f"Indirect earnings: ${icomm:,.2f}<br>"
                f"Bonuses: ${bonus_amt:,.2f}<br>"
                f"Total earnings: ${total_earn2:,.2f}"
            )
            sizes.append(8 + min(40, float(size_map.get(n, 0.0))/5))
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=sizes), hovertext=texts, hoverinfo='text')
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=20), height=600)

        tmp_dir = tempfile.gettempdir()
        path = os.path.join(tmp_dir, f"plotly_net_{abs(hash(str(sorted(G.nodes()))) ) % (10**8)}.html")
        fig.write_html(path, include_plotlyjs='cdn')
        return path
    except Exception as e_plotly:
        tmp_dir = tempfile.gettempdir()
        path = os.path.join(tmp_dir, f"error_net_{abs(hash(str(sorted(G.nodes()))) ) % (10**8)}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("<html><body><h3>Network visualization failed to generate.</h3><pre>" + str(e_plotly) + "</pre></body></html>")
        return path

# ---------- main app logic ----------
if run_btn:
    st.info("Running simulation... (cached results used when params+seed match previous runs)")
    if not monte_carlo_mode:
        # use cached single-run
        out = cached_run_single((params, seed))
        events = out['events']
        summary = out['summary']
        earnings = out['earnings']
        G = out['graph']
        subscribers = out['subscribers']

        # --- Influencer diagnostics (show if any influencers exist) ---
        if 'is_influencer' in subscribers.columns:
            num_infl = int(subscribers['is_influencer'].sum())
            pct_infl = num_infl / max(1, len(subscribers))
            st.subheader("Influencer summary")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Influencers (count)", f"{num_infl:,}")
            col_b.metric("Influencer share", f"{pct_infl:.2%}")
            avg_infl = subscribers.loc[subscribers['is_influencer'],'referral_ability'].mean() if num_infl>0 else 0.0
            avg_non = subscribers.loc[~subscribers['is_influencer'],'referral_ability'].mean() if num_infl<len(subscribers) else 0.0
            col_c.metric("Avg ability (infl / non)", f"{avg_infl:.3f} / {avg_non:.3f}")

            st.markdown("**Top influencers (by referral ability)**")
            top_infl = subscribers[subscribers['is_influencer']].sort_values('referral_ability', ascending=False).head(20)
            st.dataframe(top_infl[['subscriber_id','name','join_month','referral_ability']].reset_index(drop=True))

            infl_earn = earnings.merge(subscribers[['subscriber_id','is_influencer']], on='subscriber_id', how='left')
            infl_earn = infl_earn[infl_earn['is_influencer']==True].sort_values('total_earnings', ascending=False).head(20)
            if not infl_earn.empty:
                st.markdown("**Top influencers (by realized earnings)**")
                st.dataframe(infl_earn[['subscriber_id','name','total_commissions_received','total_bonuses_received','total_earnings']].reset_index(drop=True))
            else:
                st.info("No influencer earned commissions in this run yet.")

        # KPI cards (top row)
        total_revenue = summary['revenue'].sum()
        total_comm = summary['commission'].sum()
        total_bonus = summary['bonus'].sum()
        total_op = summary['operating_cost'].sum()
        total_net = summary['net'].sum()
        last_month = summary['month'].max()
        mrr = float(summary.loc[summary['month']==last_month,'revenue'].iloc[0]) if last_month in summary['month'].values else 0.0
        prev_mrr = float(summary.loc[summary['month']==last_month-1,'revenue'].iloc[0]) if (last_month-1) in summary['month'].values else 0.0
        mrr_change = (mrr - prev_mrr) / prev_mrr * 100.0 if prev_mrr > 0 else 0.0
        avg_active = summary['active_users'].mean() if summary['active_users'].sum() > 0 else 0
        arpu = (summary['revenue'].sum() / (summary['active_users'].sum()/len(summary))) if avg_active>0 else 0.0
        current_active = int(summary.loc[summary['month']==last_month,'active_users'].iloc[0]) if last_month in summary['month'].values else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MRR (last month)", f"${mrr:,.2f}", f"{mrr_change:.2f}% vs prev")
        col2.metric("Total Revenue (all months)", f"${total_revenue:,.2f}")
        col3.metric("Total Commissions", f"${total_comm:,.2f}")
        col4.metric("Total Bonuses", f"${total_bonus:,.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Operating Cost", f"${total_op:,.2f}")
        col6.metric("Net Profit (total)", f"${total_net:,.2f}")
        col7.metric("Current Active Subscribers", f"{current_active:,}")
        col8.metric("Avg ARPU (per active user / month)", f"${arpu:,.2f}")

        st.markdown("---")

        # Events preview & download
        st.subheader("Events DataFrame (sample)")
        st.dataframe(events.head(300))
        st.download_button("Download events CSV", data=events.to_csv(index=False).encode('utf-8'), file_name='events.csv')
        # full events export (entire events dataframe)
        st.download_button(
            "Download FULL Events CSV",
            data=events.sort_values(['month','subscriber_id']).to_csv(index=False).encode('utf-8'),
            file_name='events_full.csv',
            mime='text/csv'
        )

        # referral network table (one row per subscriber) -> include direct/indirect and earnings
        network_table = node_earnings(events, subscribers)
        st.download_button(
            "Download Referral Network CSV",
            data=network_table.to_csv(index=False).encode('utf-8'),
            file_name='referral_network.csv',
            mime='text/csv'
        )

        # monthly chart
        st.subheader("Monthly Revenue / Commission / Bonus / Operating Cost")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Revenue', x=summary['month'], y=summary['revenue']))
        fig.add_trace(go.Bar(name='Commission', x=summary['month'], y=summary['commission']))
        fig.add_trace(go.Bar(name='Bonus', x=summary['month'], y=summary['bonus']))
        fig.add_trace(go.Bar(name='Operating Cost', x=summary['month'], y=summary['operating_cost']))
        fig.update_layout(barmode='group', xaxis_title='Month', yaxis_title='USD', height=420)
        st.plotly_chart(fig, use_container_width=True)

        # net chart
        st.subheader("Net P/L over time")
        fign = go.Figure()
        fign.add_trace(go.Scatter(x=summary['month'], y=summary['net'], mode='lines+markers', name='Net'))
        fign.add_trace(go.Scatter(x=summary['month'], y=summary['revenue'], mode='lines', name='Revenue', opacity=0.3))
        fign.update_layout(xaxis_title='Month', yaxis_title='USD', height=350)
        st.plotly_chart(fign, use_container_width=True)

        # Active subscribers
        st.subheader("Active subscribers (approx) by month")
        active_by_month = events[events['event_type']=='SUBSCRIPTION'].groupby('month')['subscriber_id'].nunique().reindex(range(1,t+1), fill_value=0)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(active_by_month.index), y=active_by_month.values, mode='lines+markers'))
        fig2.update_layout(xaxis_title='Month', yaxis_title='Active subscribers', height=300)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Top earners (commissions + bonuses)")
        st.dataframe(earnings.head(100))

        st.markdown("---")

        # PyVis network
        st.subheader("Interactive Referral Network (PyVis) — pan / zoom / drag")
        path = draw_pyvis_graph(G, subscribers, network_table, height="700px", width="100%")
        # read HTML & display
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        components.html(html, height=700, scrolling=True)

        st.success("Simulation complete (cached).")

    else:
        # Monte Carlo mode
        progress_bar = st.progress(0)
        def progress(i, total):
            progress_bar.progress(int(100*i/total))
        mc_out = cached_run_monte((params, runs, seed))
        progress_bar.empty()
        df_nets = mc_out['final_nets']
        ts_pct = mc_out['ts_percentiles']

        st.subheader("Monte Carlo: final net profit per run (distribution)")
        st.dataframe(df_nets.describe().T)

        # histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df_nets['final_net'], nbinsx=40))
        fig_hist.update_layout(xaxis_title='Final Net (total over t months)', yaxis_title='Count', height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        # percentiles over time
        st.subheader("Net P/L percentile bands over months (p5,p50,p95)")
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(x=ts_pct['month'], y=ts_pct['p95'], fill=None, name='95th', line=dict(color='lightgreen')))
        fig_pct.add_trace(go.Scatter(x=ts_pct['month'], y=ts_pct['p50'], fill='tonexty', name='Median', line=dict(color='green')))
        fig_pct.add_trace(go.Scatter(x=ts_pct['month'], y=ts_pct['p5'], fill='tonexty', name='5th', line=dict(color='red')))
        fig_pct.update_layout(xaxis_title='Month', yaxis_title='Net (USD)', height=400)
        st.plotly_chart(fig_pct, use_container_width=True)

        prob_loss = (df_nets['final_net'] < 0).mean()
        st.metric("Probability final net < 0 (loss)", f"{prob_loss*100:.2f}%")

        st.download_button("Download Monte Carlo results (final nets)", data=df_nets.to_csv(index=False).encode('utf-8'), file_name='mc_final_nets.csv')
        st.success("Monte Carlo complete (cached).")
else:
    st.info("Adjust parameters and press 'Generate / Refresh' to run the simulation.")
