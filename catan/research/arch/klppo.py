"""PPO update with a KL-to-BC penalty (AlphaStar-style anchor to the supervised policy).

loss = PPO loss + kl_bc_coef * KL(pi_bc || pi) over legal actions. The frozen BC
policy's log-probs are computed once per update for the whole batch.
Same as catan.rl.ppo.ppo_update otherwise (copied: it has no hook for an extra term).
shortcut: copy of ppo_update; upgrade by adding an ``extra_loss`` hook upstream.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def make_update(bc_model, coef: float):
    bc_model.eval()

    def ppo_update(model, opt, batch, p, device, gen):
        obs = torch.from_numpy(batch["obs"]).to(device)
        mask = torch.from_numpy(batch["mask"]).to(device)
        act = torch.from_numpy(batch["act"]).to(device)
        old_logp = torch.from_numpy(batch["logp"]).to(device)
        adv_all = torch.from_numpy(batch["adv"]).to(device)
        ret = torch.from_numpy(batch["ret"]).to(device)
        n = len(act)
        with torch.no_grad():
            bc_logp = torch.cat([torch.log_softmax(bc_model(obs[i:i + 2048], mask[i:i + 2048])[0], -1)
                                 for i in range(0, n, 2048)])
        stats = defaultdict(list)
        stop = False
        for _ in range(p["epochs"]):
            perm = torch.randperm(n, generator=gen).to(device)
            for idx in perm.chunk(max(1, n // p["minibatch"])):
                logits, v = model(obs[idx], mask[idx])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, act[idx, None]).squeeze(1)
                probs = logp_all.exp()
                entropy = -(probs * logp_all).sum(-1).mean()
                q = bc_logp[idx]
                kl_bc = (q.exp() * (q - logp_all)).masked_fill(~mask[idx], 0).sum(-1).mean()
                adv = adv_all[idx]
                if p["norm_adv"]:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - old_logp[idx])
                pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
                vloss = 0.5 * ((v - ret[idx]) ** 2).mean()
                loss = pg + p["vf_coef"] * vloss - p["ent_coef"] * entropy + coef * kl_bc
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(vloss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(kl)
                stats["kl_bc"].append(kl_bc.item())
                stats["clipfrac"].append(((ratio - 1).abs() > p["clip"]).float().mean().item())
                stats["grad_norm"].append(float(gn))
                if p.get("target_kl") and kl > 1.5 * p["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["kl_early_stop"] = float(stop)
        y, yhat = batch["ret"], batch["val"]
        out["explained_variance"] = float(1 - np.var(y - yhat) / (np.var(y) + 1e-8))
        return out

    return ppo_update
