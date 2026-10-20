
import torch
from common.generation import response_token_logprobs
from task3_grpo.grpo import grpo_policy_loss


def backward_microbatched(
    policy, sequences, attention_mask, prompt_width, response_ids,
    old_logp, seq_adv, train_mask, ref_logp, **kwargs
):
    if policy.training:
        raise ValueError("Microbatched backward requires eval mode.")

    chunks = []
    with torch.no_grad():
        for i in range(sequences.shape[0]):
            logp, logits = response_token_logprobs(
                policy, sequences[i:i+1], attention_mask[i:i+1],
                prompt_width, response_ids[i:i+1]
            )
            chunks.append(logp)
            del logits

    new_logp = torch.cat(chunks).detach().requires_grad_(True)
    loss, diagnostics = grpo_policy_loss(
        new_logp, old_logp, seq_adv, train_mask, ref_logp, **kwargs
    )
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite GRPO loss.")

    logp_gradient, = torch.autograd.grad(loss, new_logp)
    reported_loss = loss.detach()
    del loss, new_logp, chunks

    for i in range(sequences.shape[0]):
        logp, logits = response_token_logprobs(
            policy, sequences[i:i+1], attention_mask[i:i+1],
            prompt_width, response_ids[i:i+1]
        )
        logp.backward(logp_gradient[i:i+1])
        del logp, logits

    return reported_loss, {
        key: value.detach() for key, value in diagnostics.items()
    }
