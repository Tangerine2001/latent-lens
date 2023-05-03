"""Train or evaluate a tuned lens for a language model."""

from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from hashlib import md5
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from tuned_lens import TunedLens
from tuned_lens.data import (
    chunk_and_tokenize,
    compute_nats_to_bpb_ratio,
    silence_datasets_messages,
)
from tuned_lens.model_surgery import get_transformer_layers
from tuned_lens.scripts import (
    downstream_loop,
    eval_loop,
    eval_loop_latent,
    train_loop,
    train_loop_latent,
)
import json
import pickle
import shutil
import torch as th
import torch.distributed as dist
import torch.utils.data.dataloader as dataloader


# json with strings to tuple of tensors
class RelatedDataset(Dataset):
    def __init__(self, dataJson, tokenizer, pad_to_multiple_of=1):
        self.dataJson = dataJson
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.maxLenPrompt = 0
        self.maxLenResponse = 0
        self.maxLenRelated = 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.dataJson)

    def __getitem__(self, index):
        prompt, response, related = self.formatText(index)

        promptT = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        responseT = self.tokenizer(response, return_tensors="pt")["input_ids"].squeeze(0)
        relatedT = self.tokenizer(related, return_tensors="pt")["input_ids"].squeeze(0)

        return promptT, responseT, relatedT

    def formatText(self, idx):
        # print("data index", idx)
        prompt = self.dataJson[idx]['requested_rewrite']['prompt']
        prompt = prompt.replace("{}", self.dataJson[idx]['requested_rewrite']['subject'])
        response = self.dataJson[idx]['requested_rewrite']['target_true']["str"] + "."
        related = self.dataJson[idx]['relatedWords']
        relatedStr = related[0]
        for r in related[1:]:
            relatedStr += " " + r

        relatedStr = relatedStr.replace(".","")
        relatedStr = relatedStr.replace(",","")
        relatedStr = relatedStr.replace("  "," ")

        return prompt, response, relatedStr

def main(args):
    """The main entry point for the command line script."""
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    # Deterministically choose a temporary cache directory shared
    # by all ranks using MD5 hash of the command line arguments
    if args.no_cache:
        cache_dir = f"/tmp/{md5(pickle.dumps(args)).hexdigest()}"
    else:
        cache_dir = None

    if args.random_model:
        print(f"Sampling random weights for model '{args.model_name}'...")
        config = AutoConfig.from_pretrained(args.model_name, revision=args.revision)
        model = AutoModelForCausalLM.from_config(  # type: ignore
            config, torch_dtype=th.float16
        )
    else:
        print(f"Loading pretrained weights for '{args.model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            args.model_name,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            revision=args.revision,
            torch_dtype="auto",
        )

    # Make sure all ranks have loaded the model, then delete the cache
    if cache_dir:
        if dist.is_initialized():
            dist.barrier()
        if local_rank == 0:
            shutil.rmtree(cache_dir)

    assert isinstance(model, PreTrainedModel)
    model.eval()
    model.requires_grad_(False)

    #th.cuda.set_device(local_rank)

    # Can be set either in eval or in training; in eval it's required
    if getattr(args, "lens", None):
        lens = TunedLens.load(args.lens, map_location="cpu")
    elif args.command in ("downstream", "eval"):
        lens = None
    else:
        lens = TunedLens(
            model,
            extra_layers=args.extra_layers,
            reuse_unembedding=not args.separate_unembeddings,
        ).float()

    if lens:
        #lens = lens.to(device=th.device("cuda", local_rank))
        lens = lens.to(device=th.device("cpu", local_rank))
        print(f"Using lens with config: {json.dumps(lens.config, indent=2)}")
    else:
        print("No tuned lens provided, using logit lens.")

    if args.fsdp:
        _, layers = get_transformer_layers(model)
        layer_cls = type(layers[0])
        print(f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy.")

        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
            ),
            cpu_offload=CPUOffload(offload_params=args.cpu_offload),
            device_id=local_rank,
            # This turns out to be important for training speed
            forward_prefetch=True,
            mixed_precision=MixedPrecision(
                param_dtype=th.float16,
                reduce_dtype=th.float16,
                buffer_dtype=th.float16,
            ),
        )
    elif args.cpu_offload:
        raise ValueError("CPU offload requires FSDP.")
    else:
        model.to(local_rank)

    from torchinfo import summary
    print(summary(model))
    print(summary(lens))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model_name,
        revision=args.revision,
        use_fast=not args.slow_tokenizer,
        tokenizer_type=args.tokenizer_type,
        padding_side='left'
    )
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # pad to eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    if args.command == "downstream":
        downstream_loop(args, model, lens, tokenizer)
        return

    # print(f"Loading dataset '{' '.join(args.dataset)}'")
    # if len(args.dataset) == 1 and args.dataset[0].endswith(".jsonl"):
    #     dataset = Dataset.from_json(args.dataset[0])
    #     assert isinstance(dataset, Dataset)
    # else:
    #     dataset = load_dataset(*args.dataset, split=args.split)
    #     if not isinstance(dataset, (Dataset, DatasetDict)):
    #         raise ValueError("Only Dataset and DatasetDict instances are supported.")

    # processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    # nats_to_bpb = compute_nats_to_bpb_ratio(dataset, processed)

    dataDir = "datasets/"
    with open(dataDir +"related.json", "r") as f:
        relatedJson = json.load(f)

    print("related dataset", len(relatedJson))
    
    # print(tokenizer.pad_token_id)
    processed = RelatedDataset(relatedJson, tokenizer, pad_to_multiple_of=32)

    nats_to_bpb = 1.0 # unused
    print(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

    assert isinstance(processed, Dataset)
    if dist.is_initialized():
        processed = processed.shard(dist.get_world_size(), local_rank)

    if args.command == "train":
        train_loop_latent(args, tokenizer, model, processed, lens, float(nats_to_bpb))
    elif args.command == "eval":
        eval_loop_latent(args, tokenizer, model, processed, lens, float(nats_to_bpb))
    else:
        raise ValueError(f"Unknown command: {args.command}")
