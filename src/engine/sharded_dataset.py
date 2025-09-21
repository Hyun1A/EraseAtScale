# src/engine/sharded_dataset.py
import os, math, gc, random
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Optional, Any
import torch
from torch.utils.data import IterableDataset



def build_global_shuffled_shards(
    targets: List[str],
    in_dir: str = "./dataset/train_pairs",
    out_dir: str = "./dataset/train_shards",
    domain_map: dict = 'all',
    num_shards: int = 8,
    n_batch: int = 64,
    total_iterations: Optional[int] = None, 
    seed: int = 42,
    anchors_filename: str = "anchors.pt",  
) -> List[str]:

    os.makedirs(out_dir, exist_ok=True)
    assert len(targets) > 0, "Empty targets"

    first_domain = domain_map[targets[0]]
    fp0 = os.path.join(in_dir, first_domain, f"{targets[0]}.pt")
    try:
        data0 = torch.load(fp0, map_location="cpu", weights_only=True)
    except TypeError:
        data0 = torch.load(fp0, map_location="cpu")
    sample_shape = tuple(data0["target"][0].shape)
    dtype = data0["target"].dtype
    per_concept_len = int(data0["target"].size(0))
    del data0
    gc.collect()

    global_indices: List[Tuple[str, int]] = [(t, i) for t in targets for i in range(per_concept_len)]
    rng = random.Random(seed)
    rng.shuffle(global_indices)
    if total_iterations is not None and total_iterations > 0:
        needed = min(len(global_indices), int(total_iterations) * int(n_batch))
        global_indices = global_indices[:needed]

    N = len(global_indices)
    if N == 0:
        return []

    shard_sizes = [N // num_shards] * num_shards
    for i in range(N % num_shards):
        shard_sizes[i] += 1

    shard_slices: List[Tuple[int, int]] = []
    start = 0
    for sz in shard_sizes:
        shard_slices.append((start, start + sz))
        start += sz

    shard_paths: List[str] = []
    need_build = False
    for si, _ in enumerate(shard_slices):
        shard_path = os.path.join(out_dir, f"gshard_{si:03d}.pt")
        shard_paths.append(shard_path)
        if not os.path.isfile(shard_path):
            need_build = True
    anchors_path = os.path.join(out_dir, anchors_filename)

    if not need_build and os.path.isfile(anchors_path):
        return sorted(shard_paths)

    schedule: Dict[str, Dict[int, Tuple[List[int], List[int]]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    shard_sizes_map: Dict[int, int] = {}
    for si, (st, ed) in enumerate(shard_slices):
        pairs = global_indices[st:ed]
        shard_sizes_map[si] = len(pairs)
        for local_pos, (c, j) in enumerate(pairs):
            pos_list, idx_list = schedule[c][si]
            pos_list.append(local_pos)
            idx_list.append(j)

    shards: Dict[int, Dict[str, Any]] = {}
    for si, n in shard_sizes_map.items():
        shards[si] = dict(
            targets=torch.empty((n, *sample_shape), dtype=dtype),
            mappings=torch.empty((n, *sample_shape), dtype=dtype),
            concept_ids=torch.full((n,), fill_value=-1, dtype=torch.long),
            concept_names=[],
            local_id={},
        )

    anchors_dict: Dict[str, torch.Tensor] = {}

    concepts = list(schedule.keys())
    total_c = len(concepts)
    for k, c in enumerate(concepts, 1):
        domain = domain_map[c]
        c_fp = os.path.join(in_dir, domain, f"{c}.pt")
        try:
            c_obj = torch.load(c_fp, map_location="cpu", weights_only=True)
        except TypeError:
            c_obj = torch.load(c_fp, map_location="cpu")

        Xt = c_obj["target"]   # (B, 2, d/2)
        Xm = c_obj["mapping"]  # (B, 2, d/2)
        Xa = c_obj["anchor"]   # (A, 2, d/2) 

        if c not in anchors_dict:
            anchors_dict[c] = Xa.cpu() if isinstance(Xa, torch.Tensor) else torch.as_tensor(Xa).cpu()

        for si, (pos_list, idx_list) in schedule[c].items():
            buf = shards[si]
            if c not in buf["local_id"]:
                buf["local_id"][c] = len(buf["concept_names"])
                buf["concept_names"].append(c)
            pos_t = torch.as_tensor(pos_list, dtype=torch.long)
            buf["targets"][pos_t]  = torch.stack([Xt[j] for j in idx_list], dim=0)
            buf["mappings"][pos_t] = torch.stack([Xm[j] for j in idx_list], dim=0)
            buf["concept_ids"][pos_t] = buf["local_id"][c]

        del Xt, Xm, Xa, c_obj
        if (k % max(1, total_c // 100)) == 0 or k == total_c:
            print(f"[Concept 1-pass] processed {k}/{total_c} concepts")
    gc.collect()

    for si in sorted(shards.keys()):
        out_path = os.path.join(out_dir, f"gshard_{si:03d}.pt")
        buf = shards[si]
        shard_obj = {
            "targets": buf["targets"].contiguous(),
            "mappings": buf["mappings"].contiguous(),
            "concept_ids": buf["concept_ids"].contiguous().to(torch.long),
            "concept_names": buf["concept_names"],
        }
        torch.save(shard_obj, out_path)

    torch.save({"anchors": anchors_dict}, anchors_path)
    print(f"[Anchors] saved to: {anchors_path} (concepts={len(anchors_dict)})")

    return sorted(shard_paths)







class GlobalShuffledShardDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: List[str],
        n_batch: int = 64,
        n_anc: int = 4,
        total_iterations: Optional[int] = None,  
        seed: int = 42,
        shuffle_within_shard: bool = False,      
        drop_last: bool = True,
        anchors_path: Optional[str] = None,   
    ):
        super().__init__()
        assert len(shard_paths) > 0, "No shard paths provided."
        self.shard_paths = list(sorted(shard_paths))
        self.n_batch = int(n_batch)
        self.n_anc = int(n_anc)
        self.total_iterations = None if total_iterations in (None, -1) else int(total_iterations)
        self.seed = int(seed)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.drop_last = bool(drop_last)

        self.anchors: Dict[str, torch.Tensor] = {}
        if anchors_path is not None:
            obj = torch.load(anchors_path, map_location="cpu")
            self.anchors = obj["anchors"] if isinstance(obj, dict) and "anchors" in obj else obj
            for k, v in self.anchors.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    self.anchors[k] = v.cpu()
        else:
            raise ValueError("anchors_path must be provided to use global anchors dictionary.")

    def _load_shard(self, path: str):
        shard = torch.load(path, map_location="cpu")
        targets: torch.Tensor = shard["targets"]           # (N, 2, d/2)
        mappings: torch.Tensor = shard["mappings"]         # (N, 2, d/2)
        concept_ids: torch.Tensor = shard["concept_ids"]   # (N,)
        concept_names: List[str] = shard["concept_names"]

        name2id = {n: i for i, n in enumerate(concept_names)}
        id2anchor: Dict[int, torch.Tensor] = {}
        for name, cid in name2id.items():
            if name not in self.anchors:
                raise KeyError(f"Anchor pool missing for concept '{name}' in anchors file.")
            id2anchor[cid] = self.anchors[name]  # (A, 2, d/2)

        return targets, mappings, concept_ids, id2anchor

    def _free_shard(self, obj):
        del obj
        gc.collect()

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        order = list(range(len(self.shard_paths)))
        epoch = 0
        steps_done = 0

        while True:
            if epoch > 0:
                rng.shuffle(order)

            for oi in order:
                path = self.shard_paths[oi]
                targets, mappings, concept_ids, id2anchor = self._load_shard(path)

                N = targets.size(0)

                if N == 0:
                    self._free_shard((targets, mappings, concept_ids, id2anchor))
                    continue

                idxs = list(range(N))
                if self.shuffle_within_shard:
                    rng.shuffle(idxs)

                shard_steps = max( (N // self.n_batch) if self.drop_last else math.ceil(N / self.n_batch), 1)

                for b in range(shard_steps):
                    st = b * self.n_batch
                    ed = st + self.n_batch
                    if ed > N:
                        if self.drop_last:
                            break
                        ed = N

                    sel = idxs[st:ed]
                    bt = targets[sel]             # (B, 2, d/2)
                    bm = mappings[sel]            # (B, 2, d/2)
                    bcid = concept_ids[sel]       # (B,)

                    a_list = []
                    

                    for cid in bcid.tolist():
                        pool: torch.Tensor = id2anchor[cid]  # (A, 2, d/2)
                        A = pool.size(0)
                        idxs_anc = [rng.randrange(0, A) for _ in range(self.n_anc)]
                        a_list.append(pool[idxs_anc])        # (n_anc, 2, d/2)
                    ba = torch.cat(a_list, dim=0)            # (B*n_anc, 2, d/2)

                    yield bt, bm, ba
                    steps_done += 1

                    if self.total_iterations is not None and steps_done >= self.total_iterations:
                        self._free_shard((targets, mappings, concept_ids, id2anchor))
                        return

                self._free_shard((targets, mappings, concept_ids, id2anchor))

            epoch += 1
            if self.total_iterations is not None and steps_done >= self.total_iterations:
                return
