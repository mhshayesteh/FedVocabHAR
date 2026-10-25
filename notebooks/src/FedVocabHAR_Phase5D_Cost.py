# %% [markdown]
# # FedVocabHAR Phase 5D: Systems cost of agreement-triggered vocabulary growth
#
# Measures what a deployment would actually pay, on the frozen encoders and the locked policy, for all three
# datasets (RealWorld selection clients, PAMAP2 integration clients, UCI-HAR integration clients):
#
# * **Uplink per client per round** for the proposed method (cluster signatures + support + medoid id) versus
#   (a) uploading the candidate raw windows and (b) one FedAvg round of encoder weights/gradients.
# * **Encoder cost**: parameters, multiply-accumulates per 128-sample window, CPU latency per window (batch 1
#   and batch 256), GPU throughput if available.
# * **Client compute**: novelty scoring + agglomerative clustering wall time per client.
# * **Server compute**: mutual-nearest-neighbour matching and ranking wall time per round.
# * **Oracle cost**: label queries per round (always <= 4).
#
# No accuracy metric is recomputed and no labels beyond the already-published Phase 5A/5C payloads are read.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import sys; sys.path.insert(0, '/content/drive/MyDrive/FedVocabHAR/colab/src')
import phase5_common as P5
from phase5_common import CFG, PROJECT, DEVICE, Frozen, Encoder, adapt_checkpoint_keys, load, calibration, ranked_windows, rows_from_clusters, cluster, mnn_groups, group_candidates, unit
import json, time, platform, os, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
OUT = PROJECT / 'artifacts' / 'phase5d'; OUT.mkdir(parents=True, exist_ok=True)
FLOAT_BYTES = 4; WINDOW_BYTES = 4 * 128 * FLOAT_BYTES  # 4 channels x 128 samples x float32
SIGNATURE_BYTES = 64 * FLOAT_BYTES + 4 + 4               # 64-D float32 signature + int32 support + int32 medoid id
print('device', DEVICE, '| cpu', platform.processor() or os.cpu_count(), 'cores', os.cpu_count())

# %% [markdown]
# ## Encoder size, MACs, and latency

# %%
def count_macs(model, shape=(1, 4, 128)):
    macs = []
    def hook(m, i, o):
        if isinstance(m, nn.Conv1d): macs.append(o.numel() // o.shape[0] * m.in_channels // m.groups * m.kernel_size[0])
        elif isinstance(m, nn.Linear): macs.append(m.in_features * m.out_features)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad(): model(torch.zeros(shape, device=next(model.parameters()).device))
    for h in hs: h.remove()
    return int(sum(macs))

def latency(model, batch, device, reps=30):
    model = model.to(device).eval(); x = torch.randn(batch, 4, 128, device=device)
    with torch.no_grad():
        for _ in range(5): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / batch * 1e3  # ms per window

enc = Encoder(6); n_params = sum(p.numel() for p in enc.parameters())
inference_params = n_params - sum(p.numel() for n, p in enc.named_parameters() if n.startswith(('domain', 'decoder')))  # heads unused at inference
macs = count_macs(enc); torch.set_num_threads(max(1, os.cpu_count() // 2))
enc_cost = {'parameters_total': n_params, 'parameters_inference_path': inference_params, 'weights_bytes_inference_fp32': inference_params * FLOAT_BYTES,
            'macs_per_window': macs, 'flops_per_window': 2 * macs,
            'cpu_ms_per_window_batch1': latency(enc, 1, torch.device('cpu')), 'cpu_ms_per_window_batch256': latency(enc, 256, torch.device('cpu')),
            'cpu_threads': torch.get_num_threads()}
if DEVICE.type == 'cuda': enc_cost['gpu_ms_per_window_batch256'] = latency(enc, 256, DEVICE); enc_cost['gpu_name'] = torch.cuda.get_device_name(0)
print(json.dumps(enc_cost, indent=1))

# %% [markdown]
# ## Per-client and per-round communication and compute (frozen encoders, locked policy)

# %%
rows_out = []; round_out = []
for dname in ('realworld', 'pamap2', 'ucihar'):
    D = Frozen(dname); ckey = D.ds['client_key']
    for fold in D.folds:
        fid = int(fold['fold']); clients = tuple(int(s) for s in fold[ckey])
        bundle = load(D.ds['models'] / f'fold_{fid}_models.pt'); m = Encoder(D.ds['domains'](fold)).to(DEVICE)
        m.load_state_dict(adapt_checkpoint_keys(bundle['anchor_state_dict']), strict=True)
        inp = D.normalized_input(bundle['stats'], bundle.get('input_scale', 1.0))
        t0 = time.perf_counter(); emb = P5.encode(m, inp); t_encode_all = time.perf_counter() - t0
        p, c = calibration(D, emb, fold); all_rows = []
        for s in clients:
            n_windows = int((D.subjects == s).sum())
            t0 = time.perf_counter(); q, nov = ranked_windows(D, emb, p, s); t_rank = time.perf_counter() - t0
            t0 = time.perf_counter(); labels = cluster(emb[q], c['local_distance']); t_cluster = time.perf_counter() - t0
            rows = rows_from_clusters(D, emb, q, labels, s); all_rows += rows
            rows_out.append({'dataset': dname, 'fold': fid, 'client': s, 'client_windows': n_windows, 'candidate_windows': len(q),
                             'retained_clusters': len(rows), 'uplink_bytes_signatures': len(rows) * SIGNATURE_BYTES,
                             'uplink_bytes_raw_candidates': len(q) * WINDOW_BYTES, 'uplink_bytes_raw_all_windows': n_windows * WINDOW_BYTES,
                             'uplink_bytes_fedavg_round': inference_params * FLOAT_BYTES,
                             'client_encode_ms': t_encode_all / len(D.x) * n_windows * 1e3, 'client_rank_ms': t_rank * 1e3, 'client_cluster_ms': t_cluster * 1e3})
        t0 = time.perf_counter(); groups = mnn_groups(all_rows, c['agreement_similarity']); cands = group_candidates(D, all_rows, groups); t_server = time.perf_counter() - t0
        round_out.append({'dataset': dname, 'fold': fid, 'clients': len(clients), 'total_signatures': len(all_rows), 'mnn_pairs': len(groups), 'agreed_candidates': len(cands),
                          'label_queries_max': min(CFG.label_query_budget, len(cands)), 'server_match_ms': t_server * 1e3,
                          'downlink_bytes_new_entries': min(CFG.label_query_budget, len(cands)) * (64 * FLOAT_BYTES + 32),  # signature + label string
                          'round_uplink_bytes_total': sum(r['uplink_bytes_signatures'] for r in rows_out if r['dataset'] == dname and r['fold'] == fid)})
        print(f'{dname} fold {fid}: {len(clients)} clients, {len(all_rows)} signatures, {len(groups)} MNN pairs, server {t_server*1e3:.1f} ms')
clients_df = pd.DataFrame(rows_out); rounds_df = pd.DataFrame(round_out)
clients_df.to_csv(OUT / 'phase5d_client_costs.csv', index=False); rounds_df.to_csv(OUT / 'phase5d_round_costs.csv', index=False)

# %% [markdown]
# ## Summary table

# %%
agg = clients_df.groupby('dataset').agg(client_windows=('client_windows', 'mean'), candidate_windows=('candidate_windows', 'mean'), retained_clusters=('retained_clusters', 'mean'),
                                        uplink_signatures_B=('uplink_bytes_signatures', 'mean'), uplink_raw_candidates_B=('uplink_bytes_raw_candidates', 'mean'),
                                        uplink_raw_all_B=('uplink_bytes_raw_all_windows', 'mean'), uplink_fedavg_B=('uplink_bytes_fedavg_round', 'mean'),
                                        encode_ms=('client_encode_ms', 'mean'), rank_ms=('client_rank_ms', 'mean'), cluster_ms=('client_cluster_ms', 'mean')).reset_index()
agg['reduction_vs_raw_candidates'] = agg.uplink_raw_candidates_B / agg.uplink_signatures_B
agg['reduction_vs_fedavg'] = agg.uplink_fedavg_B / agg.uplink_signatures_B
ragg = rounds_df.groupby('dataset').agg(clients=('clients', 'mean'), total_signatures=('total_signatures', 'mean'), mnn_pairs=('mnn_pairs', 'mean'), agreed_candidates=('agreed_candidates', 'mean'),
                                        label_queries_max=('label_queries_max', 'mean'), server_match_ms=('server_match_ms', 'mean'), round_uplink_KB=('round_uplink_bytes_total', lambda v: v.mean() / 1024)).reset_index()
summary = agg.merge(ragg, on='dataset'); summary.to_csv(OUT / 'phase5d_summary.csv', index=False); display(summary.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
(agg.set_index('dataset')[['uplink_signatures_B', 'uplink_raw_candidates_B', 'uplink_fedavg_B']] / 1024).plot(kind='bar', ax=axes[0], logy=True)
axes[0].set_ylabel('KB per client per round (log)'); axes[0].set_title('Uplink'); axes[0].grid(axis='y', alpha=.25)
agg.set_index('dataset')[['encode_ms', 'rank_ms', 'cluster_ms']].plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_ylabel('ms per client per round (CPU)'); axes[1].set_title('Client compute'); axes[1].grid(axis='y', alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'phase5d_cost.png', dpi=180, bbox_inches='tight'); plt.show()

record = {'phase': 'Phase 5D', 'encoder': enc_cost, 'bytes': {'window': WINDOW_BYTES, 'signature_record': SIGNATURE_BYTES, 'fedavg_round': inference_params * FLOAT_BYTES},
          'summary': json.loads(summary.set_index('dataset').to_json(orient='index')),
          'note': 'Wall times measured in Colab (see cpu_threads / gpu_name); treat as relative, not absolute, device costs. Clusters and signatures come from the locked policy on frozen encoders.'}
(OUT / 'phase5d_record.json').write_text(json.dumps(record, indent=2))
md = ('# FedVocabHAR Phase 5D Systems Cost\n\n'
      f"Encoder: {inference_params:,} inference parameters ({inference_params*4/1024:.0f} KB fp32), {macs/1e6:.2f} M MACs per window, "
      f"CPU {enc_cost['cpu_ms_per_window_batch1']:.2f} ms/window (batch 1), {enc_cost['cpu_ms_per_window_batch256']:.3f} ms/window (batch 256).\n\n"
      '| Dataset | Clients | Windows/client | Candidates/client | Signatures/client | Uplink (B) | Raw candidates (B) | FedAvg (B) | x vs raw | x vs FedAvg | Client ms | Server ms | Queries |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
for _, r in summary.iterrows():
    md += (f"| {r.dataset} | {r.clients:.0f} | {r.client_windows:.0f} | {r.candidate_windows:.0f} | {r.retained_clusters:.1f} | {r.uplink_signatures_B:,.0f} | {r.uplink_raw_candidates_B:,.0f} | "
           f"{r.uplink_fedavg_B:,.0f} | {r.reduction_vs_raw_candidates:.0f} | {r.reduction_vs_fedavg:.0f} | {r.encode_ms + r.rank_ms + r.cluster_ms:.0f} | {r.server_match_ms:.1f} | <= {r.label_queries_max:.1f} |\n")
(OUT / 'FedVocabHAR_Phase5D_Cost.md').write_text(md); print(md)
