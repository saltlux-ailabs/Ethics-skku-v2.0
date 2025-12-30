from functools import partial
import torch
import torch.nn.functional as F
import csv
import pdb

GLOBAL_LOGS = []
current_batch_idx = 0
cnt = 0
def rank_value_vecs(model, toxic_vector):
    """ 
    Rank all value vectors based on similarity vs. toxic_vector.
    toxic_vector: [d_model]
    """
    scores = []
    for layer in range(model.config.n_layer):
        # mlp_outs = model.blocks[layer].mlp.W_out
        # [d_mlp, d_model]
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(mlp_outs, toxic_vector.unsqueeze(0), dim=1)
        _topk = cos_sims.topk(k=100)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return sorted_scores

def get_intervene_vector(model, config):
    def _get_toxic_probe(_config):
        return torch.load(_config["datapath"])

    return {
        "toxic_probe": _get_toxic_probe,
    }[config["type"]](config)

def hook_subtract(model, config):
    intervene_vector = get_intervene_vector(model, config).squeeze()
    
    scale = config["scale"]
    subtract_from = config["subtract_from"]
    hook_timesteps = config["hook_timesteps"]

    def patch(vec, _scale):
        def hook(module, input, output):

            _vec = vec.unsqueeze(0).unsqueeze(0)
            if hook_timesteps == -1:
                _vec = _vec.repeat(output.shape[0], 1, 1)
            else:
                _vec = _vec.repeat(output.shape[0], output.shape[1], 1)
            output[:, hook_timesteps:, :] = output[:, hook_timesteps:, :] - (
                _scale * _vec
            )
            return output

        return hook

    hooks = []
    for layer in subtract_from:
        hook = model.transformer.h[layer].mlp.register_forward_hook(
        # hook = model.model.layers[layer].mlp.register_forward_hook(
            patch(intervene_vector, scale)
        )
        hooks.append(hook)
    return model, hooks

def _top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Create a mask to filter out tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask to the right to keep also the first token above top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    indices_to_keep = ~indices_to_remove  # Boolean mask to keep only the tokens to keep
    return indices_to_keep


def hook_subtract_with_weighted_sum(model, config, log_file="log.csv", mode="input", weight_path="", log_path="", alpha=100):
    class_probe_vectors = torch.load(None) #fill-in

    scale = config["scale"]
    subtract_from = config["subtract_from"]
    hook_timesteps = config["hook_timesteps"]
    attention_mask = config['attention_mask'].bool()
    input_id = config["prompt_input_ids"]
    
    def patch(class_probe_vectors, _scale):
        def hook(module, input, output):
            global GLOBAL_LOGS, current_batch_idx, cnt
            if mode == "input":
                hidden_state = input[0]
            elif mode == "output":
                hidden_state = output[:, hook_timesteps:, :]
            elif mode == "last_token":
                hidden_state = output[:, -1, :]                             
            else:
                raise ValueError(f"Invalid mode: {mode}. Choose from 'input', 'output', or 'last_token'.")

            batch_size = hidden_state.shape[0] 
            seq_length = hidden_state.shape[1]
            batch_attention_mask = attention_mask[:, :seq_length]
            alphas = []

            actual_batch_size = min(hidden_state.shape[0], batch_attention_mask.shape[0])
            for i in range(batch_size):
                similarities = []
                
                if seq_length > 1:
                    valid_positions = batch_attention_mask[i]
                    valid_hidden_state = hidden_state[i][valid_positions, :]
                    
                    for probe_vector in class_probe_vectors:
                        probe_vector = probe_vector.unsqueeze(0)  
                        cos_sim = F.cosine_similarity(valid_hidden_state, probe_vector, dim=-1)
                        similarities.append(cos_sim.mean().item())
                else:
                    for probe_vector in class_probe_vectors:
                        probe_vector = probe_vector.unsqueeze(0)
                        cos_sim = F.cosine_similarity(hidden_state[i:i+1], probe_vector, dim=-1)
                        similarities.append(cos_sim.mean().item())

                similarities = torch.tensor(similarities, device=hidden_state.device)
                
                # if selection_mode == "weighted_sum":
                # similarities = torch.where(similarities > 0, similarities, torch.tensor(float('-inf'), device=hidden_state.device))
                # weights = F.softmax(similarities, dim=0)
                # weights = torch.where(similarities == float('-inf'), torch.tensor(0.0, device=hidden_state.device), weights)
                # weighted_probe_vector = torch.sum(weights.unsqueeze(1) * class_probe_vectors, dim=0)

                # elif selection_mode == "max_similarity":
                max_sim_idx = torch.argmax(similarities)
                max_similarity = similarities[max_sim_idx]
                weighted_probe_vector = class_probe_vectors[max_sim_idx]
                if max_similarity <= 0:
                    weighted_probe_vector = torch.zeros_like(class_probe_vectors[0])

                original_logits = model.lm_head(output[i:i+1, :, :])  
                output[i:i+1, hook_timesteps:, :] -= config['scale'] * weighted_probe_vector.unsqueeze(0).unsqueeze(1)
                steered_logits = model.lm_head(output[i:i+1, :, :])
                output[i:i+1, hook_timesteps:, :] += config['scale'] * weighted_probe_vector.unsqueeze(0).unsqueeze(1)
                
                if seq_length>1:
                    original_last_token_logits = original_logits.squeeze()[-1].detach()
                    steer_last_token_logits = steered_logits.squeeze()[-1].detach()
                else:
                    original_last_token_logits = original_logits.squeeze().detach()
                    steer_last_token_logits = steered_logits.squeeze().detach()                    

                # pdb.set_trace()
                indices_to_select_original = _top_p_filtering(original_last_token_logits, 0.9)
                indices_to_select_steer = _top_p_filtering(steer_last_token_logits, 0.9)
                indices_to_select_merged = indices_to_select_original | indices_to_select_steer

                original_logits_filtered = original_last_token_logits.clone().masked_select(indices_to_select_merged)
                steer_logits_filtered = steer_last_token_logits.clone().masked_select(indices_to_select_merged)
                
                original_softmaxed = F.log_softmax(original_logits_filtered, dim=-1).detach()           # dim: [vocab_size] target distribution
                steer_softmaxed = F.log_softmax(steer_logits_filtered, dim=-1).detach() 

                kl = F.kl_div(
                    steer_softmaxed,   # input
                    original_softmaxed,     # target
                    reduction='batchmean',
                    log_target=True,
                )  

                kl_div = kl.item() * 125000
                # alphas = max(min(kl_div, 100), 0)
                alphas = max(min(kl_div, alpha), 0)

                output[i:i+1, hook_timesteps:, :] -= alphas * weighted_probe_vector.unsqueeze(0).unsqueeze(1)
                # output[i:i+1, hook_timesteps:, :] -= scale * weighted_probe_vector.unsqueeze(0).unsqueeze(1)

                GLOBAL_LOGS.append({
                    "input_index": current_batch_idx+i,
                    # "selected_class": selected_class,
                    # "similarity": selected_similarity,
                    "alpha": alphas,
                    "kl_div": kl_div,
                })
                                
            cnt += 1
            if cnt == 20:
                current_batch_idx += batch_size
                cnt = 0

            return output 

        return hook

    def save_logs():
        """Save the global logs to a CSV file."""
        global GLOBAL_LOGS
        categories = ["insult", "identity_hate", "obscene", "threat", "entire", "other"]
        with open(log_file, "w", newline="") as csvfile:
            fieldnames = ["input_index", "alpha", "kl_div"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for log in GLOBAL_LOGS:
                # 클래스 이름 변환
                category_name = categories[log["selected_class"]]
                # 퍼센트 포맷
                # similarity_percent = f"{log['similarity'] * 100:.2f}%"
                writer.writerow({
                    "input_index": log["input_index"],
                    # "selected_class": category_name,
                    # "similarity": similarity_percent,
                    "alpha": log["alpha"],
                    "kl_div": log["kl_div"],
                })

    hooks = []
    for layer in subtract_from:
        hook = model.transformer.h[layer].mlp.register_forward_hook(
            patch(class_probe_vectors, scale)
        )
        hooks.append(hook)
    model.save_logs = save_logs
    return model, hooks

def dont_hook(model, config):
    return model, []