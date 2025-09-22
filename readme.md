<<<<<<< HEAD
## Installation

**OS**: Ubuntu 18.04.5 LTS

**Python**: 3.10.14

<pre>
conda create -n eas python=3.10
</pre>

<pre>
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2
</pre>

<pre>
pip install -r requirements.txt
</pre>


## Distillation from gates
### Training gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/celebs/train_gate_single.sh
</pre>

Characters
<pre>
bash ./shell_scripts/distill_gate/characters/train_gate_single.sh
</pre>

Artists
<pre>
bash ./shell_scripts/distill_gate/artists/train_gate_single.sh
</pre>

### Distillation from gates
<pre>
bash ./shell_scripts/distill_gate/hetero/train_distill_gate.sh
</pre>

### Distillation with noise
<pre>
bash ./shell_scripts/distill_gate/hetero/train_distill_gate_w_noise.sh
</pre>

### Generation with gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/celebs/generate_gate.sh
</pre>

### Generation after distillation from gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/hetero/generate_distill_gate.sh
</pre>

### Generation after distillation with noise
Celebrities
<pre>
bash ./shell_scripts/distill_gate/hetero/generate_distill_gate_w_noise.sh
</pre>


## Direct training
### Extract samples
Celebrities
<pre>
bash ./shell_scripts/direct_train/celebs/extract_single_last_token.sh
</pre>

### Train at once
<pre>
bash ./shell_scripts/direct_train/hetero/train_at_once.sh
</pre>

### Train at once with noise
<pre>
bash ./shell_scripts/direct_train/hetero/train_at_once_w_noise.sh
</pre>

### Generation
<pre>
bash ./shell_scripts/direct_train/hetero/generate_direct.sh
</pre>

### Generation after noise injection
<pre>
bash ./shell_scripts/direct_train/hetero/generate_direct_w_noise.sh
</pre>


## Generating images from original, or noise-injected model
Original
<pre>
bash ./shell_scripts/org/generate_org.sh
</pre>

After noise injection
<pre>
bash ./shell_scripts/org/generate_org_w_noise.sh
</pre>
=======
## Installation

**OS**: Ubuntu 18.04.5 LTS

**Python**: 3.10.14

<pre>
conda create -n eas python=3.10
</pre>

<pre>
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2
</pre>

<pre>
pip install -r requirements.txt
</pre>


## Distillation from gates
### Training gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/celebs/train_gate_single.sh
</pre>

Characters
<pre>
bash ./shell_scripts/distill_gate/characters/train_gate_single.sh
</pre>

Artists
<pre>
bash ./shell_scripts/distill_gate/artists/train_gate_single.sh
</pre>

### Distillation from gates
<pre>
bash ./shell_scripts/distill_gate/hetero/train_distill_gate.sh
</pre>

### Distillation with noise
<pre>
bash ./shell_scripts/distill_gate/hetero/train_distill_gate_w_noise.sh
</pre>

### Generation with gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/celebs/generate_gate.sh
</pre>

### Generation after distillation from gates
Celebrities
<pre>
bash ./shell_scripts/distill_gate/hetero/generate_distill_gate.sh
</pre>

### Generation after distillation with noise
Celebrities
<pre>
bash ./shell_scripts/distill_gate/hetero/generate_distill_gate_w_noise.sh
</pre>


## Direct training
### Extract samples
Celebrities
<pre>
bash ./shell_scripts/direct_train/celebs/extract_single_last_token.sh
</pre>

### Train at once
<pre>
bash ./shell_scripts/direct_train/hetero/train_at_once.sh
</pre>

### Train at once with noise
<pre>
bash ./shell_scripts/direct_train/hetero/train_at_once_w_noise.sh
</pre>

### Generation
<pre>
bash ./shell_scripts/direct_train/hetero/generate_direct.sh
</pre>

### Generation after noise injection
<pre>
bash ./shell_scripts/direct_train/hetero/generate_direct_w_noise.sh
</pre>


## Generating images from original, or noise-injected model
Original
<pre>
bash ./shell_scripts/org/generate_org.sh
</pre>

After noise injection
<pre>
bash ./shell_scripts/org/generate_org_w_noise.sh
</pre>
>>>>>>> 00cc7c0e7d2b051b6a5bcf4a75e01eb8a91df4b2
