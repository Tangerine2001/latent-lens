"""Evaluation loop for the tuned lens model."""
from argparse import Namespace
from datasets import Dataset
from collections import defaultdict
from itertools import islice
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from typing import Optional
from tuned_lens.residual_stream import record_residual_stream
from tuned_lens.stats import ResidualStats, LogitStats
from tuned_lens.nn import Decoder, TunedLens
from tuned_lens.utils import (
    maybe_all_cat,
    maybe_all_reduce,
    shift_labels,
    shift_preds,
    pytree_map,
    pytree_stack,
    send_to_device,
)
import torch as th
import torch.distributed as dist

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

# batching and padding, use left pad tokkenizer
class RelatedCollator():
    def __init__(self, tokenizer, pad_to_multiple_of=1):
        self.tokenizer = tokenizer
        self.textCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of)
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        prompt, response, related = zip(*batch)

        # print(prompt)
        self.tokenizer.padding_side = "left"
        prompt = self.textCollator(prompt)
        self.tokenizer.padding_side = "right"
        response = self.textCollator(response)
        self.tokenizer.padding_side = "left"
        related = self.textCollator(related)

        # causing index out of range error
        del prompt["labels"]
        del response["labels"]
        del related["labels"]

        prompt["attention_mask"] = th.where(prompt["input_ids"] == self.pad_token_id, 0, 1)
        response["attention_mask"] = th.where(response["input_ids"] == self.pad_token_id, 0, 1)
        related["attention_mask"] = th.where(related["input_ids"] == self.pad_token_id, 0, 1)

        return prompt, response, related # token index in ["input_ids"], -100 for pad in ["labels"]

@th.autocast("cuda", enabled=th.cuda.is_available(), dtype=th.float16)
@th.no_grad()
def eval_loop_latent(
    args: Namespace,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    data: Dataset,
    lens: Optional[TunedLens],
    nats_to_bpb_ratio: float,
):
    """Trains a TunedLens model against a transformer on a dataset.

    Args:
        args: The command-line arguments see __main__.py train subcommand.
        model: The transformer model to train.
        data: The dataset to train on.
        lens: The TunedLens model to train.
        nats_to_bpb_ratio: The ratio of nats to bits per byte for the dataset.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    # dl = DataLoader(
    #     data.shuffle(seed=args.seed),  # type: ignore[arg-type],
    #     batch_size=args.per_gpu_batch_size,
    # )
    collator = RelatedCollator(tokenizer, data.pad_to_multiple_of)
    dl = DataLoader(data, batch_size=args.per_gpu_batch_size, collate_fn=collator, shuffle=False)
    if lens:
        lens.eval()

    # Running mean & covariance of the hidden states & residuals
    delta_stats = ResidualStats(cov=False)
    stream_stats = ResidualStats(dtype=th.float32)

    if args.limit:
        dl = islice(dl, args.limit)
        total = args.limit
    else:
        total = len(dl)

    root_dir = args.output or args.lens / "eval"
    output_dir = root_dir / f"rank_{local_rank}"
    output_dir.mkdir(exist_ok=True, parents=True)

    _to_logits = lens.to_logits if lens else Decoder(model)
    L = model.config.num_hidden_layers
    batches = []
    transfer_batches = []

    grad_alignments = [[] for _ in range(L)]

    final_logit_stats = LogitStats()
    ll_statistics = [LogitStats() for _ in range(L)]
    tl_statistics = [LogitStats() for _ in range(L)]

    responseNlls = []
    # recalls = [{} for _ in range(L)]
    # precisions = [{} for _ in range(L)]
    globalRecallNum = [{} for _ in range(L)]
    globalRecallDenom = [{} for _ in range(L)]
    globalPrecisionNum = [{} for _ in range(L)]
    globalPrecisionDenom = [{} for _ in range(L)]

    pbar = tqdm(dl, desc="Evaluating", position=local_rank, total=total)
    for batch_t in pbar:
        batch_t = send_to_device(batch_t, th.device(local_rank))
        batch, response, related = batch_t
        # batch is left pad, response is right pad, related is left pad
        # _ _ _ _ prompt | response _ _ _
        with record_residual_stream(model) as stream:
            output = model(**batch)

        # needs an attention mask for full prompt, but input ids dont include past keys, logits are just for response tokens
        full_att = th.cat([batch["attention_mask"], response["attention_mask"]], dim=1)
        responseOutput = model(input_ids=response["input_ids"], attention_mask=full_att, past_key_values=output.past_key_values)

        responseNll = nllResponse(output, responseOutput, batch, response, related, tokenizer.pad_token_id).detach().to("cpu")
        responseNlls.append(responseNll)

        shift = args.token_shift if args.token_shift is not None else 1
        final_lps = output.logits.log_softmax(dim=-1)
        final_probs = final_lps.exp()
        labels = shift_labels(batch["input_ids"], shift)

        batch_output = defaultdict(dict)
        transfer_ces = th.zeros(L, L, device=final_lps.device)
        transfer_kls = th.zeros(L, L, device=final_lps.device)

        batch_output["response_nll"] = responseNll

        # Compute logit lens eval and statistics
        for (j, d), (name, h) in zip(enumerate(stream.residuals()), stream.items()):
            if args.grad_alignment:
                h.requires_grad_(True)
                h.retain_grad()

            with th.set_grad_enabled(args.grad_alignment):
                baseline_lps = _to_logits(h).log_softmax(dim=-1)

                # Note that we don't reduce the loss here, since we want to look at
                # the full distribution of losses across tokens and samples
                losses = th.nn.functional.cross_entropy(
                    shift_preds(baseline_lps, shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                avg_loss = losses.mean()

            if args.grad_alignment:
                avg_loss.backward()

                assert h.grad is not None
                grad_alignments[j].append(
                    th.nn.functional.cosine_similarity(
                        h.grad.flatten(1), d.flatten(1), dim=-1
                    )
                )
                h.grad = None

            ll_statistics[j].update(baseline_lps, assume_normalized=True)

            batch_output["baseline_ce"][name] = losses
            batch_output["baseline_entropy"][name] = th.sum(
                -baseline_lps.exp() * baseline_lps, dim=-1
            )
            batch_output["baseline_kl"][name] = th.sum(
                final_probs * (final_lps - baseline_lps), dim=-1
            )

        # Compute tuned lens eval and statistics if applicable
        if lens:
            for j, (name, h) in zip(range(L), stream.items()):
                lens_lps = lens(h, idx=j).log_softmax(dim=-1)
                lens_probs = lens_lps.exp()

                batch_output["latent_lens_f1"][name] = {}
                batch_output["latent_lens_precision"][name] = {}
                batch_output["latent_lens_recall"][name] = {}

                for thresh in [0.001, 0.003, 0.005, 0.01, 0.03, 0.05]:
                    precision, recall, f1, precNum, precDen, recallNum, recallDen = precisionRecallLens(lens_lps, batch, response, related, tokenizer.pad_token_id, thresh=thresh)

                    batch_output["latent_lens_f1"][name][thresh] = th.tensor(f1, dtype=th.float32)
                    batch_output["latent_lens_precision"][name][thresh] = th.tensor(precision, dtype=th.float32)
                    batch_output["latent_lens_recall"][name][thresh] = th.tensor(recall, dtype=th.float32)

                    # if thresh not in precisions[j]:
                    #     precisions[j][thresh] = [precision]
                    # else:
                    #     precisions[j][thresh].append(precision)

                    # if thresh not in recalls[j]:
                    #     recalls[j][thresh] = [recall]
                    # else:
                    #     recalls[j][thresh].append(recall)

                    if thresh not in globalPrecisionNum[j]:
                        globalPrecisionNum[j][thresh] = precNum
                    else:
                        globalPrecisionNum[j][thresh] += precNum
                    if thresh not in globalPrecisionDenom[j]:
                        globalPrecisionDenom[j][thresh] = precDen
                    else:
                        globalPrecisionDenom[j][thresh] += precDen

                    if thresh not in globalRecallNum[j]:
                        globalRecallNum[j][thresh] = recallNum
                    else:
                        globalRecallNum[j][thresh] += recallNum
                    if thresh not in globalRecallDenom[j]:
                        globalRecallDenom[j][thresh] = recallDen
                    else:
                        globalRecallDenom[j][thresh] += recallDen
                    
                    

                batch_output["lens_ce"][name] = th.nn.functional.cross_entropy(
                    shift_preds(lens_lps, shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                batch_output["lens_entropy"][name] = th.sum(
                    -lens_probs * lens_lps, dim=-1
                )
                batch_output["lens_kl"][name] = th.sum(
                    final_probs * (final_lps - lens_lps), dim=-1
                )
                tl_statistics[j].update(lens_lps, assume_normalized=True)

                if args.transfer:
                    # Each iteration of the loop processes a different *probe* layer i
                    # for the test layer j.
                    for i in range(L):
                        transfer_lps = lens(h, idx=i).log_softmax(dim=-1)
                        transfer_ces[i, j] = th.nn.functional.cross_entropy(
                            shift_preds(transfer_lps, shift).flatten(0, 1),
                            labels.flatten(),
                        )
                        transfer_kls[i, j] = th.sum(
                            lens_probs * (lens_lps - transfer_lps), dim=-1
                        ).mean()

        final_logit_stats.update(final_lps, assume_normalized=True)
        if args.residual_stats:
            # Drop the first token because it's weird
            rest = stream.map(lambda x: x[:, 1:])

            delta_stats.update(rest.residuals())
            stream_stats.update(rest)

        batch_output["baseline_ce"]["final"] = th.nn.functional.cross_entropy(
            shift_preds(final_lps, shift).flatten(0, 1),
            labels.flatten(),
            reduction="none",
        )
        batch_output["baseline_entropy"]["final"] = th.sum(
            -final_probs * final_lps, dim=-1
        )
        th.save(batch_output, output_dir / f"batch_{pbar.n}.pt")

        batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]
        transfer_batches.append(
            {
                "transfer_ce": transfer_ces,
                "transfer_kl": transfer_kls,
            }
        )

    pbar.close()
    agg = pytree_map(lambda x: nats_to_bpb_ratio * x.mean(), pytree_stack(batches))
    agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
    if local_rank == 0:
        th.save(agg, root_dir / "aggregate_metrics.pt")

    if args.transfer:
        agg_transfer = pytree_map(
            lambda x: nats_to_bpb_ratio * x.mean(0), pytree_stack(transfer_batches)
        )
        agg_transfer = pytree_map(lambda x: maybe_all_reduce(x), agg_transfer)
        if local_rank == 0:
            th.save(agg_transfer, root_dir / "aggregate_transfer_metrics.pt")

    # first_token_stats.all_reduce_()
    delta_stats.all_reduce_()
    stream_stats.all_reduce_()
    for stats in ll_statistics:
        stats.all_reduce_()

    if lens:
        for stats in tl_statistics:
            stats.all_reduce_()

        responsePerplexity = th.exp(th.stack(responseNlls).mean())
        globalPrecision = [{} for _ in range(L)]
        globalRecall = [{} for _ in range(L)]
        globalF1 = [{} for _ in range(L)]
        for i in range(L):
            for thresh in globalPrecisionNum[i].keys():
                if globalPrecisionDenom[i][thresh] == 0:
                    globalPrecision[i][thresh] = 1
                else:
                    globalPrecision[i][thresh] = globalPrecisionNum[i][thresh] / globalPrecisionDenom[i][thresh]
                if globalRecallDenom[i][thresh] == 0:
                    globalRecall[i][thresh] = 0
                else:
                    globalRecall[i][thresh] = globalRecallNum[i][thresh] / globalRecallDenom[i][thresh]
                if globalPrecision[i][thresh] + globalRecall[i][thresh] == 0:
                    globalF1[i][thresh] = 0
                else:
                    globalF1[i][thresh] = 2 * (globalPrecision[i][thresh] * globalRecall[i][thresh]) / (globalPrecision[i][thresh] + globalRecall[i][thresh])

        latent_stats = {
            "precision": globalPrecision,
            "recall": globalRecall,
            "f1": globalF1,
            "response_perplexity": responsePerplexity,
        }
        print(latent_stats)
        th.save(latent_stats, root_dir / "latent_stats.pt")

    if args.grad_alignment:
        grad_alignments = [maybe_all_cat(th.cat(x, dim=0)) for x in grad_alignments]
        if local_rank == 0:
            th.save(grad_alignments, root_dir / "grad_alignments.pt")

    if local_rank == 0:
        th.save(delta_stats, root_dir / "delta_stats.pt")
        th.save(stream_stats, root_dir / "stream_stats.pt")

        th.save(final_logit_stats, root_dir / "final_logit_stats.pt")
        th.save(ll_statistics, root_dir / "ll_logit_stats.pt")
        if lens:
            th.save(tl_statistics, root_dir / "tl_logit_stats.pt")

def nllResponse(outputs, responseOutput, prompt, response, related, pad_token_id):
    promptLogits = outputs.logits
    responseLogits = responseOutput.logits
    # add last logit in prompt to response logits
    responseLogits = th.cat((promptLogits[:, -1, :].unsqueeze(1), responseLogits), dim=1)[:,:-1,:]
  
    nll = th.nn.functional.cross_entropy(responseLogits.flatten(0, 1), response["input_ids"].flatten(0, 1), ignore_index=pad_token_id)
    return nll

def precisionRecallLens(logprob, prompt, response, related, pad_token_id, thresh=0.001):
        
    # print(hidden_lps.layers[0].shape)
    # print(responseOutput.logits.shape)

    num_related = related["input_ids"].shape[1]
    num_prompt = prompt["input_ids"].shape[1]

    num_related_unique = th.unique(related["input_ids"]).shape[0]

    # trueP = [[0] * num_prompt] * num_layers
    # predN = [[0] * num_prompt] * num_layers
    uniqueTrueP = set()

    recall = 0
    precision = 0
    f1 = 0

    # print(related["input_ids"].shape)
    if True:
        # thresh = 0.001 # 0.5%
        prob = logprob.exp()
        # print("max prob", prob.max().item())
        Ns = (prob > thresh).sum(dim=-1)
        totalN = Ns.sum().item()
        # print("Ns", Ns.shape)
        totalTrueP = 0
        for k in range(num_prompt):
            if prompt["input_ids"][0, k] == pad_token_id:
                continue
            N = Ns[0, k].item()
            topN = th.argsort(logprob, descending=True, dim=-1)[0, k, :N]
            # print("topN", topN.shape)
            for i in range(num_related):
                if related["input_ids"][0, i] in topN:
                    # trueP[j][k] += 1
                    totalTrueP += 1
                    uniqueTrueP.add(related["input_ids"][0, i].item())
            
            # predN[j][k] = N

        # predList = list(uniqueTrueP[j])
        # if len(predList) > 0:
        #     predStrList = []
        #     for token in predList:
        #         predStrList.append(tokenizer.decode([token]))
        #     print(j, "pred", predList, predStrList)
        # print(j, "unique", len(uniqueTrueP[j]), "trueP", totalTrueP, "N", totalN, "num_related", num_related)

        recall = len(uniqueTrueP) / num_related_unique
        if totalN == 0:
            precision = 1
        else:
            precision = totalTrueP / totalN
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return (precision, recall, f1, 
                    totalTrueP, totalN, 
                    len(uniqueTrueP), num_related_unique)
