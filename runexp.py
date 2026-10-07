"""实验启动封装脚本。

提供命令行参数转发，内部通过子进程调用 `MDGFM.py`，
便于统一入口执行不同数据集与超参数配置。
"""

import subprocess
import sys
import argparse

# Optional MDGFM.py flags set in __main__ (avoids bloating run_experiment signature).
_PASSTHROUGH_MDGFM = []

def run_experiment(
    dataset,
    drop_percent,
    lr,
    downlr,
    epochs,
    shot_num,
    seed=1024,
    feature_adapter='pca',
    rp_dim=512,
    allin_post_norm='none',
    coral_weight=0.0,
    proto_weight=0.0,
    finetune_gcn=False,
    finetune_gcn_lr_scale=0.1,
    downstream_head='downprompt',
    dual_alpha=0.5,
    dual_epochs=400,
    dual_tau=1.0,
    dual_calib=False,
    pgr_weight=0.0,
    pgr_margin=0.2,
    lp_refine=False,
    lp_beta=0.3,
    lp_steps=2,
    self_train_steps=0,
    self_train_thresh=0.9,
    self_train_weight=0.3,
    proto_margin=0.0,
    proto_margin_weight=0.0,
    learn_dual_alpha=False,
    dual_label_smoothing=0.0,
    dual_ensemble=1,
    use_srm=False,
    srm_temp=4.0,
    srm_reg_weight=0.0,
    domain_gate=False,
    domain_gate_gamma=0.3,
    dpc_weight=0.0,
    dpc_temp=0.2,
    use_sgfm=False,
    sgfm_bottleneck=64,
    sgfm_scale=0.2,
    use_hat_adapter=False,
    hat_gate_hidden=16,
    hat_reg_weight=0.0,
    use_band_gsl=False,
    band_init_alpha=0.5,
    band_v2_adaptive_gate=False,
    band_v2_cross_domain=False,
    band_trust_low=0.0,
    band_gate_clamp=False,
    band_cd_weight=0.1,
    band_deg_anchor=0.0,
    band_gate_entropy=0.0,
    use_graph_mae=False,
    mae_weight=0.25,
    mae_mask_ratio=0.2,
    pretrain_ema=False,
    pretrain_ema_decay=0.999,
    use_feat_dv_inv=False,
    feat_dv_weight=0.15,
    feat_dv_dropout=0.25,
    use_film_prompt=False,
    use_film_residual=False,
    film_residual_scale=0.15,
    use_gcil=False,
    gcil_weight=0.1,
    use_gcil_spectral=False,
    gcil_inv_weight=1.0,
    gcil_indep_weight=0.1,
    scgw_p1=False,
    scgw_p2=False,
    scgw_p3=False,
    scgw_p4=False,
    scgw_num_bases=8,
    scgw_base_size=16,
    scgw_tau=1.0,
    scgw_weight=0.1,
    scgw_feat_blend=0.3,
    scgw_prompt_blend=0.3,
    use_scale_gnn=False,
    scale_gnn_hops=3,
    scale_encoder='none',
    scale_residual_gamma=0.15,
    use_mdgfm_ap=False,
    ap_num_hops=3,
    ap_homo_cond=False,
    ap_reg_weight=0.0,
    use_mdgfm_hs=False,
    use_hybrid_spectral_pretrain=False,
    hs_num_prompt=10,
    hs_tau_inner=0.35,
    hs_tau_cross=0.25,
    hs_prompt_epochs=200,
    hs_reg_weight=0.01,
    use_mdgfm_graver=False,
    graver_vocab_size=8,
    graver_ego_hop=1,
    graver_max_ego_nodes=24,
    graver_vocab_noise=0.05,
    graver_router_hidden=64,
    graver_router_epochs=30,
    graver_moe_weight=0.01,
    graver_samples_per_class=20,
    graver_wildcard_per_domain=30,
    graver_global_weight=0.4,
    graver_wildcard_weight=0.25,
    graver_moe_aux_weight=0.1,
    use_bikt=False,
    bikt_weight=0.05,
    use_prograph=False,
    prograph_subspaces=3,
    prograph_view_weight=0.05,
    use_mfgia=False,
    mfgia_domain_dim=32,
    mfgia_refresh_every=0,
    use_mdgfm_tri=False,
    tri_subspaces=3,
    tri_domain_dim=32,
    tri_bikt_weight=0.02,
    tri_view_weight=0.02,
    tri_refresh_every=0,
    result_tag='',
    use_homo_router=False,
    homo_bypass_thresh=0.5,
    homo_dual_domains='',
    homo_router_scope='episode',
    hetero_bikt_scale=1.0,
    hetero_sim_temp=1.0,
    use_hetero_film=False,
    hetero_film_scale=0.15,
    hetero_struct_boost=0.0,
    homo_router_soft=False,
    homo_soft_temp=0.08,
    use_uniprop=False,
    dual_adapted=False,
    eval_episodes=50,
    use_mtg=False,
    mtg_prototypes=4,
    mtg_hetero_only=True,
    f2_gee_branch='off',
    f2_node_w_branch='off',
    f2_node_w_gamma=1.0,
    f2_leaky_branch='off',
    f2_leaky_alpha=0.3,
    f2_leaky_steps=2,
    f2_leaky_homo_cond=True,
    f2_teacher_branch='off',
    f2_teacher_mix=1.0,
    f2_teacher_gamma=1.0,
    f2_subproto_branch='off',
    f2_subproto_k=2,
    f2_lsub_weight=0.1,
    f2_lsmo_branch='off',
    f2_lsmo_weight=0.05,
):
    cmd = [
        sys.executable,
        '-u',
        'MDGFM.py',
        '--dataset', str(dataset),
        '--drop_percent', str(drop_percent),
        '--lr',str(lr),
        '--downstreamlr',str(downlr),
        '--epochs',str(epochs),
        '--shot_num',str(shot_num),
        '--seed', str(seed),
        # [ALL-IN] Optional args (default keeps original pipeline).
        '--feature_adapter', str(feature_adapter),
        '--rp_dim', str(rp_dim),
        '--allin_post_norm', str(allin_post_norm),
        '--coral_weight', str(coral_weight),
        '--proto_weight', str(proto_weight),
        '--finetune_gcn_lr_scale', str(finetune_gcn_lr_scale),
        '--downstream_head', str(downstream_head),
        '--dual_alpha', str(dual_alpha),
        '--dual_epochs', str(dual_epochs),
        '--dual_tau', str(dual_tau),
        '--pgr_weight', str(pgr_weight),
        '--pgr_margin', str(pgr_margin),
        '--lp_beta', str(lp_beta),
        '--lp_steps', str(lp_steps),
        '--self_train_steps', str(self_train_steps),
        '--self_train_thresh', str(self_train_thresh),
        '--self_train_weight', str(self_train_weight),
        '--proto_margin', str(proto_margin),
        '--proto_margin_weight', str(proto_margin_weight),
        '--dual_label_smoothing', str(dual_label_smoothing),
        '--dual_ensemble', str(dual_ensemble),
        '--srm_temp', str(srm_temp),
        '--srm_reg_weight', str(srm_reg_weight),
        '--domain_gate_gamma', str(domain_gate_gamma),
        '--dpc_weight', str(dpc_weight),
        '--dpc_temp', str(dpc_temp),
        '--sgfm_bottleneck', str(sgfm_bottleneck),
        '--sgfm_scale', str(sgfm_scale),
        '--hat_gate_hidden', str(hat_gate_hidden),
        '--hat_reg_weight', str(hat_reg_weight),
        '--band_init_alpha', str(band_init_alpha),
        '--band_trust_low', str(band_trust_low),
        '--band_cd_weight', str(band_cd_weight),
        '--band_deg_anchor', str(band_deg_anchor),
        '--band_gate_entropy', str(band_gate_entropy),
        '--mae_weight', str(mae_weight),
        '--mae_mask_ratio', str(mae_mask_ratio),
        '--pretrain_ema_decay', str(pretrain_ema_decay),
        '--feat_dv_weight', str(feat_dv_weight),
        '--feat_dv_dropout', str(feat_dv_dropout),
        '--film_residual_scale', str(film_residual_scale),
        '--gcil_weight', str(gcil_weight),
        '--gcil_inv_weight', str(gcil_inv_weight),
        '--gcil_indep_weight', str(gcil_indep_weight),
        '--scale_gnn_hops', str(scale_gnn_hops),
        '--scale_encoder', str(scale_encoder),
        '--scale_residual_gamma', str(scale_residual_gamma),
        '--ap_num_hops', str(ap_num_hops),
        '--ap_reg_weight', str(ap_reg_weight),
        '--hs_num_prompt', str(hs_num_prompt),
        '--hs_tau_inner', str(hs_tau_inner),
        '--hs_tau_cross', str(hs_tau_cross),
        '--hs_prompt_epochs', str(hs_prompt_epochs),
        '--hs_reg_weight', str(hs_reg_weight),
        '--bikt_weight', str(bikt_weight),
        '--prograph_subspaces', str(prograph_subspaces),
        '--prograph_view_weight', str(prograph_view_weight),
        '--mfgia_domain_dim', str(mfgia_domain_dim),
        '--mfgia_refresh_every', str(mfgia_refresh_every),
        '--tri_subspaces', str(tri_subspaces),
        '--tri_domain_dim', str(tri_domain_dim),
        '--tri_bikt_weight', str(tri_bikt_weight),
        '--tri_view_weight', str(tri_view_weight),
        '--tri_refresh_every', str(tri_refresh_every),
    ]
    if result_tag:
        cmd.extend(['--result_tag', str(result_tag)])
    if domain_gate:
        cmd.append('--domain_gate')
    if use_srm:
        cmd.append('--use_srm')
    if learn_dual_alpha:
        cmd.append('--learn_dual_alpha')
    if dual_calib:
        cmd.append('--dual_calib')
    if lp_refine:
        cmd.append('--lp_refine')
    if finetune_gcn:
        cmd.append('--finetune_gcn')
    if use_sgfm:
        cmd.append('--use_sgfm')
    if use_hat_adapter:
        cmd.append('--use_hat_adapter')
    if use_band_gsl:
        cmd.append('--use_band_gsl')
    if band_v2_adaptive_gate:
        cmd.append('--band_v2_adaptive_gate')
    if band_v2_cross_domain:
        cmd.append('--band_v2_cross_domain')
    if band_gate_clamp:
        cmd.append('--band_gate_clamp')
    if use_graph_mae:
        cmd.append('--use_graph_mae')
    if pretrain_ema:
        cmd.append('--pretrain_ema')
    if use_feat_dv_inv:
        cmd.append('--use_feat_dv_inv')
    if use_film_prompt:
        cmd.append('--use_film_prompt')
    if use_film_residual:
        cmd.append('--use_film_residual')
    if use_gcil:
        cmd.append('--use_gcil')
    if use_gcil_spectral:
        cmd.append('--use_gcil_spectral')
    if scgw_p1:
        cmd.append('--scgw_p1')
    if scgw_p2:
        cmd.append('--scgw_p2')
    if scgw_p3:
        cmd.append('--scgw_p3')
    if scgw_p4:
        cmd.append('--scgw_p4')
    if scgw_num_bases != 8:
        cmd.extend(['--scgw_num_bases', str(scgw_num_bases)])
    if scgw_base_size != 16:
        cmd.extend(['--scgw_base_size', str(scgw_base_size)])
    if scgw_tau != 1.0:
        cmd.extend(['--scgw_tau', str(scgw_tau)])
    if scgw_weight != 0.1:
        cmd.extend(['--scgw_weight', str(scgw_weight)])
    if scgw_feat_blend != 0.3:
        cmd.extend(['--scgw_feat_blend', str(scgw_feat_blend)])
    if scgw_prompt_blend != 0.3:
        cmd.extend(['--scgw_prompt_blend', str(scgw_prompt_blend)])
    if use_scale_gnn:
        cmd.append('--use_scale_gnn')
    if use_mdgfm_ap:
        cmd.append('--use_mdgfm_ap')
    if ap_homo_cond:
        cmd.append('--ap_homo_cond')
    if use_mdgfm_hs:
        cmd.append('--use_mdgfm_hs')
    if use_hybrid_spectral_pretrain:
        cmd.append('--use_hybrid_spectral_pretrain')
    if use_mdgfm_graver:
        cmd.extend([
            '--use_mdgfm_graver',
            '--graver_vocab_size', str(graver_vocab_size),
            '--graver_ego_hop', str(graver_ego_hop),
            '--graver_max_ego_nodes', str(graver_max_ego_nodes),
            '--graver_vocab_noise', str(graver_vocab_noise),
            '--graver_router_hidden', str(graver_router_hidden),
            '--graver_router_epochs', str(graver_router_epochs),
            '--graver_moe_weight', str(graver_moe_weight),
            '--graver_samples_per_class', str(graver_samples_per_class),
            '--graver_wildcard_per_domain', str(graver_wildcard_per_domain),
            '--graver_global_weight', str(graver_global_weight),
            '--graver_wildcard_weight', str(graver_wildcard_weight),
            '--graver_moe_aux_weight', str(graver_moe_aux_weight),
        ])
    if use_bikt:
        cmd.append('--use_bikt')
    if use_prograph:
        cmd.append('--use_prograph')
    if use_mfgia:
        cmd.append('--use_mfgia')
    if use_mdgfm_tri:
        cmd.append('--use_mdgfm_tri')
    if use_homo_router:
        cmd.extend(['--use_homo_router', '--homo_bypass_thresh', str(homo_bypass_thresh)])
        if homo_dual_domains:
            cmd.extend(['--homo_dual_domains', str(homo_dual_domains)])
        if homo_router_soft:
            cmd.extend(['--homo_router_soft', '--homo_soft_temp', str(homo_soft_temp)])
        if homo_router_scope != 'episode':
            cmd.extend(['--homo_router_scope', str(homo_router_scope)])
    if use_uniprop:
        cmd.append('--use_uniprop')
    if dual_adapted:
        cmd.append('--dual_adapted')
    if use_mtg:
        cmd.append('--use_mtg')
        if mtg_prototypes != 4:
            cmd.extend(['--mtg_prototypes', str(mtg_prototypes)])
        if mtg_hetero_only:
            cmd.append('--mtg_hetero_only')
        else:
            cmd.append('--no_mtg_hetero_only')
    if eval_episodes != 50:
        cmd.extend(['--eval_episodes', str(eval_episodes)])
    if hetero_bikt_scale != 1.0:
        cmd.extend(['--hetero_bikt_scale', str(hetero_bikt_scale)])
    if hetero_sim_temp != 1.0:
        cmd.extend(['--hetero_sim_temp', str(hetero_sim_temp)])
    if use_hetero_film:
        cmd.append('--use_hetero_film')
    if hetero_film_scale != 0.15:
        cmd.extend(['--hetero_film_scale', str(hetero_film_scale)])
    if hetero_struct_boost != 0.0:
        cmd.extend(['--hetero_struct_boost', str(hetero_struct_boost)])
    # F2 / F2+P4 downstream plugins
    if f2_gee_branch and f2_gee_branch != 'off':
        cmd.extend(['--f2_gee_branch', str(f2_gee_branch)])
    if f2_node_w_branch and f2_node_w_branch != 'off':
        cmd.extend(['--f2_node_w_branch', str(f2_node_w_branch)])
        cmd.extend(['--f2_node_w_gamma', str(f2_node_w_gamma)])
    if f2_leaky_branch and f2_leaky_branch != 'off':
        cmd.extend(['--f2_leaky_branch', str(f2_leaky_branch)])
        cmd.extend(['--f2_leaky_alpha', str(f2_leaky_alpha)])
        cmd.extend(['--f2_leaky_steps', str(f2_leaky_steps)])
        if f2_leaky_homo_cond:
            cmd.append('--f2_leaky_homo_cond')
        else:
            cmd.append('--no_f2_leaky_homo_cond')
    if f2_teacher_branch and f2_teacher_branch != 'off':
        cmd.extend(['--f2_teacher_branch', str(f2_teacher_branch)])
        cmd.extend(['--f2_teacher_mix', str(f2_teacher_mix)])
        cmd.extend(['--f2_teacher_gamma', str(f2_teacher_gamma)])
    if f2_subproto_branch and f2_subproto_branch != 'off':
        cmd.extend(['--f2_subproto_branch', str(f2_subproto_branch)])
        cmd.extend(['--f2_subproto_k', str(f2_subproto_k)])
        cmd.extend(['--f2_lsub_weight', str(f2_lsub_weight)])
    if f2_lsmo_branch and f2_lsmo_branch != 'off':
        cmd.extend(['--f2_lsmo_branch', str(f2_lsmo_branch)])
        cmd.extend(['--f2_lsmo_weight', str(f2_lsmo_weight)])
    if _PASSTHROUGH_MDGFM:
        cmd.extend(_PASSTHROUGH_MDGFM)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with specific parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--drop_percent', type=float, default=0.5, help='Drop percentage.')
    #parser.add_argument('--gpu', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--lr', type=float, default=0.002, help='pretrain lr')
    parser.add_argument('--downstreamlr', type=float, default=0.03, help='downstream lr')
    parser.add_argument('--epochs', type=int, default=60, help='epoch')
    parser.add_argument('--shot_num', type=int, default=1, help='shotnum')
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--feature_adapter', type=str, default='pca', choices=['pca', 'allin'],
                        help='[ALL-IN] feature adapter switch')
    parser.add_argument('--rp_dim', type=int, default=512,
                        help='[ALL-IN] random projection width for allin adapter')
    parser.add_argument('--allin_post_norm', type=str, default='none', choices=['none', 'zscore'],
                        help='[ALL-IN] optional post normalization after compression')
    parser.add_argument('--coral_weight', type=float, default=0.0,
                        help='[B2] CORAL regularization weight')
    parser.add_argument('--proto_weight', type=float, default=0.0,
                        help='[B] prototype loss weight in downstream adaptation')
    parser.add_argument('--finetune_gcn', action='store_true',
                        help='[B] enable downstream micro-finetuning on pretrained GCN')
    parser.add_argument('--finetune_gcn_lr_scale', type=float, default=0.1,
                        help='[B] learning-rate scale for GCN params')
    parser.add_argument('--downstream_head', type=str, default='downprompt', choices=['downprompt', 'dual'],
                        help='downstream classifier head')
    parser.add_argument('--dual_alpha', type=float, default=0.5,
                        help='[dual] linear/prototype fusion weight')
    parser.add_argument('--dual_epochs', type=int, default=400,
                        help='[dual] optimization steps per episode')
    parser.add_argument('--dual_tau', type=float, default=1.0,
                        help='[dual] temperature for prototype logits')
    parser.add_argument('--dual_calib', action='store_true',
                        help='[dual] enable learnable logit calibration')
    parser.add_argument('--pgr_weight', type=float, default=0.0,
                        help='[dual] prototype graph regularization weight')
    parser.add_argument('--pgr_margin', type=float, default=0.2,
                        help='[dual] max off-diagonal prototype similarity')
    parser.add_argument('--lp_refine', action='store_true',
                        help='[dual] apply LP refinement on test logits')
    parser.add_argument('--lp_beta', type=float, default=0.3,
                        help='[dual] LP mixing coefficient')
    parser.add_argument('--lp_steps', type=int, default=2,
                        help='[dual] LP propagation steps')
    parser.add_argument('--self_train_steps', type=int, default=0,
                        help='[dual] pseudo-label self-training steps')
    parser.add_argument('--self_train_thresh', type=float, default=0.9,
                        help='[dual] pseudo-label confidence threshold')
    parser.add_argument('--self_train_weight', type=float, default=0.3,
                        help='[dual] pseudo-label loss weight')
    parser.add_argument('--proto_margin', type=float, default=0.0,
                        help='[dual] additive margin for prototype CE')
    parser.add_argument('--proto_margin_weight', type=float, default=0.0,
                        help='[dual] weight of prototype-margin CE')
    parser.add_argument('--learn_dual_alpha', action='store_true',
                        help='[dual] learn linear vs prototype fusion weight')
    parser.add_argument('--dual_label_smoothing', type=float, default=0.0,
                        help='[dual] label smoothing for supervised CE')
    parser.add_argument('--dual_ensemble', type=int, default=1,
                        help='[dual] K independent heads per episode, average logits at test')
    parser.add_argument('--use_srm', action='store_true',
                        help='[dual-v2] enable seed-robust mixture (episode-wise adaptive alpha)')
    parser.add_argument('--srm_temp', type=float, default=4.0,
                        help='[dual-v2] confidence-gap scaling for adaptive alpha')
    parser.add_argument('--srm_reg_weight', type=float, default=0.0,
                        help='[dual-v2] regularization weight for adaptive alpha')
    parser.add_argument('--domain_gate', action='store_true',
                        help='[v1] enable domain-aware mixture prompt in pretraining')
    parser.add_argument('--domain_gate_gamma', type=float, default=0.3,
                        help='[v1] blend ratio for gated shared prompt mixing')
    parser.add_argument('--dpc_weight', type=float, default=0.0,
                        help='[v1] domain prototype contrast weight')
    parser.add_argument('--dpc_temp', type=float, default=0.2,
                        help='[v1] domain prototype contrast temperature')
    parser.add_argument('--use_sgfm', action='store_true',
                        help='[SGRM] support-guided FiLM before dual head')
    parser.add_argument('--sgfm_bottleneck', type=int, default=64,
                        help='[SGRM] FiLM MLP bottleneck')
    parser.add_argument('--sgfm_scale', type=float, default=0.2,
                        help='[SGRM] modulation strength')
    parser.add_argument('--use_hat_adapter', action='store_true',
                        help='[HAT] homophily-adaptive gate in dual head')
    parser.add_argument('--hat_gate_hidden', type=int, default=16,
                        help='[HAT] hidden width of gate MLP')
    parser.add_argument('--hat_reg_weight', type=float, default=0.0,
                        help='[HAT] gate saturation regularization weight')
    parser.add_argument('--use_band_gsl', action='store_true',
                        help='[BandGSL] dual-branch low/high-frequency GSL in pretraining')
    parser.add_argument('--band_init_alpha', type=float, default=0.5,
                        help='[BandGSL] initial low-frequency fusion weight')
    parser.add_argument('--band_v2_adaptive_gate', action='store_true',
                        help='[BandGSL-v2] enable node-adaptive low/high gate')
    parser.add_argument('--band_v2_cross_domain', action='store_true',
                        help='[BandGSL-v2] enable cross-domain low/high consistency regularization')
    parser.add_argument('--band_trust_low', type=float, default=0.0,
                        help='[BandGSL-v3] trust-region mix toward low-only adjacency')
    parser.add_argument('--band_gate_clamp', action='store_true',
                        help='[BandGSL-v3] clamp node gate to [0.15,0.85]')
    parser.add_argument('--band_cd_weight', type=float, default=0.1,
                        help='[BandGSL] cross-domain consistency loss weight')
    parser.add_argument('--band_deg_anchor', type=float, default=0.0,
                        help='[BandGSL] degree anchor weight')
    parser.add_argument('--band_gate_entropy', type=float, default=0.0,
                        help='[BandGSL] gate entropy bonus weight')
    parser.add_argument('--use_graph_mae', action='store_true',
                        help='[Pretrain] Graph-MAE auxiliary reconstruction')
    parser.add_argument('--mae_weight', type=float, default=0.25,
                        help='[Pretrain] MAE loss weight')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.2,
                        help='[Pretrain] MAE mask ratio')
    parser.add_argument('--pretrain_ema', action='store_true',
                        help='[Pretrain] EMA-smooth and save EMA weights')
    parser.add_argument('--pretrain_ema_decay', type=float, default=0.999,
                        help='[Pretrain] EMA decay')
    parser.add_argument('--use_feat_dv_inv', action='store_true',
                        help='[Pretrain] feature view invariance')
    parser.add_argument('--feat_dv_weight', type=float, default=0.15,
                        help='[Pretrain] view invariance weight')
    parser.add_argument('--feat_dv_dropout', type=float, default=0.25,
                        help='[Pretrain] dropout for second view')
    parser.add_argument('--use_film_prompt', action='store_true',
                        help='[Pretrain] FiLM replaces sumtext (node-wise gamma/beta)')
    parser.add_argument('--use_film_residual', action='store_true',
                        help='[Pretrain] sumtext + scaled FiLM residual')
    parser.add_argument('--film_residual_scale', type=float, default=0.15,
                        help='[Pretrain] FiLM residual scale after sumtext')
    parser.add_argument('--use_gcil', action='store_true',
                        help='[GCIL] causal-style invariance/independence pretrain auxiliary')
    parser.add_argument('--gcil_weight', type=float, default=0.1,
                        help='[GCIL] auxiliary loss weight')
    parser.add_argument('--use_gcil_spectral', action='store_true',
                        help='[GCIL] spectral low-pass view for GCIL')
    parser.add_argument('--gcil_inv_weight', type=float, default=1.0,
                        help='[GCIL] invariance term scale')
    parser.add_argument('--gcil_indep_weight', type=float, default=0.1,
                        help='[GCIL] independence term scale')
    parser.add_argument('--scgw_p1', action='store_true',
                        help='[SCGFM-P1] structure-aware feature re-encoding after PCA')
    parser.add_argument('--scgw_p2', action='store_true',
                        help='[SCGFM-P2] geometric-base pretrain reconstruction loss')
    parser.add_argument('--scgw_p3', action='store_true',
                        help='[SCGFM-P3] structural coords into downstream meta prompt')
    parser.add_argument('--scgw_p4', action='store_true',
                        help='[SCGFM-P4] BandGSL low/high coord alignment (needs --use_band_gsl)')
    parser.add_argument('--scgw_num_bases', type=int, default=8)
    parser.add_argument('--scgw_base_size', type=int, default=16)
    parser.add_argument('--scgw_tau', type=float, default=1.0)
    parser.add_argument('--scgw_weight', type=float, default=0.1)
    parser.add_argument('--scgw_feat_blend', type=float, default=0.3)
    parser.add_argument('--scgw_prompt_blend', type=float, default=0.3)
    parser.add_argument('--use_scale_gnn', action='store_true',
                        help='[Scale1] legacy alias for --scale_encoder scale1')
    parser.add_argument('--scale_encoder', type=str, default='none',
                        choices=['none', 'scale1', 'scale2', 'scale_residual', 'gpr_residual'],
                        help='Graph encoder variant')
    parser.add_argument('--scale_gnn_hops', type=int, default=3,
                        help='[Scale] hop count for fusion')
    parser.add_argument('--scale_residual_gamma', type=float, default=0.15,
                        help='[Scale residual] gamma in [0,1] for base+gamma*(scale-base)')
    parser.add_argument('--use_mdgfm_ap', action='store_true',
                        help='[MDGFM-AP] adaptive propagation prompt on downstream adjtot')
    parser.add_argument('--ap_num_hops', type=int, default=3,
                        help='[MDGFM-AP] hop count for adjacency fusion')
    parser.add_argument('--ap_homo_cond', action='store_true',
                        help='[MDGFM-AP] condition hop weights on support homophily')
    parser.add_argument('--ap_reg_weight', type=float, default=0.0,
                        help='[MDGFM-AP] L2 regularization on AP hop logits')
    parser.add_argument('--use_mdgfm_hs', action='store_true',
                        help='[MDGFM-HS] spectral prompt graph downstream + dual re-encode')
    parser.add_argument('--use_hybrid_spectral_pretrain', action='store_true',
                        help='[MDGFM-HS] dual-band fusion in pretrain')
    parser.add_argument('--hs_num_prompt', type=int, default=10,
                        help='[MDGFM-HS] virtual prompt nodes per band')
    parser.add_argument('--hs_tau_inner', type=float, default=0.35,
                        help='[MDGFM-HS] inner-edge cosine threshold')
    parser.add_argument('--hs_tau_cross', type=float, default=0.25,
                        help='[MDGFM-HS] cross-edge cosine threshold')
    parser.add_argument('--hs_prompt_epochs', type=int, default=200,
                        help='[MDGFM-HS] spectral prompt tuning steps per episode')
    parser.add_argument('--hs_reg_weight', type=float, default=0.01,
                        help='[MDGFM-HS] L2 on spectral prompt features')
    parser.add_argument('--use_mdgfm_graver', action='store_true',
                        help='[MDGFM-GRAVER] generative vocabulary support augmentation')
    parser.add_argument('--graver_vocab_size', type=int, default=8)
    parser.add_argument('--graver_ego_hop', type=int, default=1)
    parser.add_argument('--graver_max_ego_nodes', type=int, default=24)
    parser.add_argument('--graver_vocab_noise', type=float, default=0.05)
    parser.add_argument('--graver_router_hidden', type=int, default=64)
    parser.add_argument('--graver_router_epochs', type=int, default=30)
    parser.add_argument('--graver_moe_weight', type=float, default=0.01)
    parser.add_argument('--graver_samples_per_class', type=int, default=20)
    parser.add_argument('--graver_wildcard_per_domain', type=int, default=30)
    parser.add_argument('--graver_global_weight', type=float, default=0.4)
    parser.add_argument('--graver_wildcard_weight', type=float, default=0.25)
    parser.add_argument('--graver_moe_aux_weight', type=float, default=0.1)
    parser.add_argument('--use_bikt', action='store_true')
    parser.add_argument('--bikt_weight', type=float, default=0.05)
    parser.add_argument('--use_prograph', action='store_true')
    parser.add_argument('--prograph_subspaces', type=int, default=3)
    parser.add_argument('--prograph_view_weight', type=float, default=0.05)
    parser.add_argument('--use_mfgia', action='store_true')
    parser.add_argument('--mfgia_domain_dim', type=int, default=32)
    parser.add_argument('--mfgia_refresh_every', type=int, default=0)
    parser.add_argument('--use_mdgfm_tri', action='store_true')
    parser.add_argument('--tri_subspaces', type=int, default=3)
    parser.add_argument('--tri_domain_dim', type=int, default=32)
    parser.add_argument('--tri_bikt_weight', type=float, default=0.02)
    parser.add_argument('--tri_view_weight', type=float, default=0.02)
    parser.add_argument('--tri_refresh_every', type=int, default=0)
    parser.add_argument('--result_tag', type=str, default='')
    parser.add_argument('--use_homo_router', action='store_true',
                        help='Per-episode homophily router: homo>thresh -> Dual, else downprompt/route')
    parser.add_argument('--homo_bypass_thresh', type=float, default=0.5,
                        help='[Homo router] support homo threshold for Dual branch')
    parser.add_argument('--homo_dual_domains', type=str, default='',
                        help='Datasets allowed Dual branch (e.g. Cora); empty = all')
    parser.add_argument('--homo_router_scope', type=str, default='episode',
                        choices=['episode', 'dataset'],
                        help='Homo router: per-episode (default) or dataset-level all Dual/BiKT')
    parser.add_argument('--hetero_bikt_scale', type=float, default=1.0,
                        help='BiKT weight multiplier on heterophilic episodes')
    parser.add_argument('--hetero_sim_temp', type=float, default=1.0,
                        help='Prototype temperature on heterophilic BiKT episodes')
    parser.add_argument('--use_hetero_film', action='store_true',
                        help='Support FiLM on heterophilic BiKT path')
    parser.add_argument('--hetero_film_scale', type=float, default=0.15)
    parser.add_argument('--hetero_struct_boost', type=float, default=0.0,
                        help='Hetero episodes: more learned GSL adj')
    parser.add_argument('--homo_router_soft', action='store_true',
                        help='Soft homo router: blend Dual/downprompt logits')
    parser.add_argument('--homo_soft_temp', type=float, default=0.08,
                        help='Soft router temperature for sigmoid((homo-thresh)/temp)')
    parser.add_argument('--use_uniprop', action='store_true',
                        help='UniProp unified AP re-encode + soft Dual/BiKT blend')
    parser.add_argument('--dual_adapted', action='store_true',
                        help='Homo router: Dual on MDGFM 4.3 prompt+GSL path (no AP)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='Few-shot eval episodes (10 for fast screen)')
    parser.add_argument('--use_mtg', action='store_true',
                        help='[MTG] layer-wise message tuning on frozen GCN (Dual re-encode)')
    parser.add_argument('--mtg_prototypes', type=int, default=4,
                        help='[MTG] prototypes per GCN layer')
    parser.add_argument('--mtg_hetero_only', action='store_true', default=True,
                        help='[MTG] MTG on hetero/BiKT branch only when homo router is on')
    parser.add_argument('--no_mtg_hetero_only', dest='mtg_hetero_only', action='store_false',
                        help='[MTG] wrap GCN globally (legacy Dual reencode)')
    _f2b = ['off', 'dual', 'bikt', 'hetero', 'both']
    parser.add_argument('--f2_gee_branch', type=str, default='off', choices=_f2b,
                        help='[E1/R4] support-GEE concat (default target: dual)')
    parser.add_argument('--f2_node_w_branch', type=str, default='off', choices=_f2b,
                        help='[E2/R3] node weights (default target: bikt)')
    parser.add_argument('--f2_node_w_gamma', type=float, default=1.0,
                        help='[E2] node-weight sharpness')
    parser.add_argument('--f2_leaky_branch', type=str, default='off', choices=_f2b,
                        help='[E3] leaky aggregation (default target: bikt)')
    parser.add_argument('--f2_leaky_alpha', type=float, default=0.3,
                        help='[E3] base leak rate')
    parser.add_argument('--f2_leaky_steps', type=int, default=2,
                        help='[E3] leaky propagation steps')
    parser.add_argument('--f2_leaky_homo_cond', action='store_true', default=True,
                        help='[E3] scale leaky alpha by episode homo')
    parser.add_argument('--no_f2_leaky_homo_cond', dest='f2_leaky_homo_cond', action='store_false',
                        help='[E3] disable homo-conditioned leaky alpha')
    parser.add_argument('--f2_teacher_branch', type=str, default='off', choices=_f2b,
                        help='[R1] weighted teacher prototypes (default: bikt)')
    parser.add_argument('--f2_teacher_mix', type=float, default=1.0)
    parser.add_argument('--f2_teacher_gamma', type=float, default=1.0)
    parser.add_argument('--f2_subproto_branch', type=str, default='off', choices=_f2b,
                        help='[R2] multi-subclass / L_sub (default: dual)')
    parser.add_argument('--f2_subproto_k', type=int, default=2)
    parser.add_argument('--f2_lsub_weight', type=float, default=0.1)
    parser.add_argument('--f2_lsmo_branch', type=str, default='off', choices=_f2b,
                        help='[R5] L_smo smoothness (default: bikt)')
    parser.add_argument('--f2_lsmo_weight', type=float, default=0.05)
    parser.add_argument('--fixed_ckpt', type=str, default='',
                        help='Fixed pretrain ckpt path (no timestamp)')
    parser.add_argument('--load_pretrained', type=str, default='',
                        help='Skip pretrain; load this ckpt for downstream')
    parser.add_argument('--pretrain_only', action='store_true',
                        help='Save pretrain ckpt and exit before downstream')
    args = parser.parse_args()

    _PASSTHROUGH_MDGFM.clear()
    if args.load_pretrained:
        _PASSTHROUGH_MDGFM.extend(['--load_pretrained', args.load_pretrained])
    if args.fixed_ckpt:
        _PASSTHROUGH_MDGFM.extend(['--fixed_ckpt', args.fixed_ckpt])
    if args.pretrain_only:
        _PASSTHROUGH_MDGFM.append('--pretrain_only')

    run_experiment(
        args.dataset,
        args.drop_percent,
        args.lr,
        args.downstreamlr,
        args.epochs,
        args.shot_num,
        args.seed,
        args.feature_adapter,
        args.rp_dim,
        args.allin_post_norm,
        args.coral_weight,
        args.proto_weight,
        args.finetune_gcn,
        args.finetune_gcn_lr_scale,
        args.downstream_head,
        args.dual_alpha,
        args.dual_epochs,
        args.dual_tau,
        args.dual_calib,
        args.pgr_weight,
        args.pgr_margin,
        args.lp_refine,
        args.lp_beta,
        args.lp_steps,
        args.self_train_steps,
        args.self_train_thresh,
        args.self_train_weight,
        args.proto_margin,
        args.proto_margin_weight,
        args.learn_dual_alpha,
        args.dual_label_smoothing,
        args.dual_ensemble,
        args.use_srm,
        args.srm_temp,
        args.srm_reg_weight,
        args.domain_gate,
        args.domain_gate_gamma,
        args.dpc_weight,
        args.dpc_temp,
        args.use_sgfm,
        args.sgfm_bottleneck,
        args.sgfm_scale,
        args.use_hat_adapter,
        args.hat_gate_hidden,
        args.hat_reg_weight,
        args.use_band_gsl,
        args.band_init_alpha,
        args.band_v2_adaptive_gate,
        args.band_v2_cross_domain,
        args.band_trust_low,
        args.band_gate_clamp,
        args.band_cd_weight,
        args.band_deg_anchor,
        args.band_gate_entropy,
        args.use_graph_mae,
        args.mae_weight,
        args.mae_mask_ratio,
        args.pretrain_ema,
        args.pretrain_ema_decay,
        args.use_feat_dv_inv,
        args.feat_dv_weight,
        args.feat_dv_dropout,
        args.use_film_prompt,
        args.use_film_residual,
        args.film_residual_scale,
        args.use_gcil,
        args.gcil_weight,
        args.use_gcil_spectral,
        args.gcil_inv_weight,
        args.gcil_indep_weight,
        args.scgw_p1,
        args.scgw_p2,
        args.scgw_p3,
        args.scgw_p4,
        args.scgw_num_bases,
        args.scgw_base_size,
        args.scgw_tau,
        args.scgw_weight,
        args.scgw_feat_blend,
        args.scgw_prompt_blend,
        args.use_scale_gnn,
        args.scale_gnn_hops,
        args.scale_encoder,
        args.scale_residual_gamma,
        args.use_mdgfm_ap,
        args.ap_num_hops,
        args.ap_homo_cond,
        args.ap_reg_weight,
        args.use_mdgfm_hs,
        args.use_hybrid_spectral_pretrain,
        args.hs_num_prompt,
        args.hs_tau_inner,
        args.hs_tau_cross,
        args.hs_prompt_epochs,
        args.hs_reg_weight,
        args.use_mdgfm_graver,
        args.graver_vocab_size,
        args.graver_ego_hop,
        args.graver_max_ego_nodes,
        args.graver_vocab_noise,
        args.graver_router_hidden,
        args.graver_router_epochs,
        args.graver_moe_weight,
        args.graver_samples_per_class,
        args.graver_wildcard_per_domain,
        args.graver_global_weight,
        args.graver_wildcard_weight,
        args.graver_moe_aux_weight,
        args.use_bikt,
        args.bikt_weight,
        args.use_prograph,
        args.prograph_subspaces,
        args.prograph_view_weight,
        args.use_mfgia,
        args.mfgia_domain_dim,
        args.mfgia_refresh_every,
        args.use_mdgfm_tri,
        args.tri_subspaces,
        args.tri_domain_dim,
        args.tri_bikt_weight,
        args.tri_view_weight,
        args.tri_refresh_every,
        args.result_tag,
        args.use_homo_router,
        args.homo_bypass_thresh,
        args.homo_dual_domains,
        args.homo_router_scope,
        args.hetero_bikt_scale,
        args.hetero_sim_temp,
        args.use_hetero_film,
        args.hetero_film_scale,
        args.hetero_struct_boost,
        args.homo_router_soft,
        args.homo_soft_temp,
        args.use_uniprop,
        args.dual_adapted,
        args.eval_episodes,
        args.use_mtg,
        args.mtg_prototypes,
        args.mtg_hetero_only,
        args.f2_gee_branch,
        args.f2_node_w_branch,
        args.f2_node_w_gamma,
        args.f2_leaky_branch,
        args.f2_leaky_alpha,
        args.f2_leaky_steps,
        args.f2_leaky_homo_cond,
        args.f2_teacher_branch,
        args.f2_teacher_mix,
        args.f2_teacher_gamma,
        args.f2_subproto_branch,
        args.f2_subproto_k,
        args.f2_lsub_weight,
        args.f2_lsmo_branch,
        args.f2_lsmo_weight,
    )