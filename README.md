# MCoT_hallucination: Activation Decoding for Hallucination Mitigation in Multimodal Chain-of-Thought Models
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.27201)

Multimodal Chain-of-Thought (MCoT) models have demonstrated impressive capability in complex visual reasoning tasks. Unfortunately, recent studies reveal that they suffer from severe hallucination problems due to diminished visual attention during the generation process. However, visual attention decay is a well-studied problem in Large Vision-Language Models (LVLMs). Considering the fundamental differences in reasoning processes between MCoT models and traditional LVLMs, we raise a basic question: **Whether MCoT models have unique causes of hallucinations?** To answer this question, we systematically investigate the hallucination patterns of MCoT models and find that fabricated texts are primarily generated in associative reasoning steps, which we term divergent thinking. Leveraging these insights, we introduce a simple yet effective strategy that can effectively localize divergent thinking steps and intervene in the decoding process to mitigate hallucinations. Extensive experiments show that our method outperforms existing methods by a large margin. More importantly, our proposed method can be conveniently integrated with other hallucination mitigation methods and further boost their performance. 


<div align="center">
<img src=images\motivation.jpg width="60%">
</div>

## Method
For each decoding step, the model first extracts visual token representations from a selected hidden layer and projects them through the language head. These activations are used to build token-wise visual entropy scores over visual positions. The entropy is normalized by `log(m)`, where `m` is the number of visual positions, and the current generative token is checked against a threshold `gamma`. When the token exceeds the threshold, decoding is adjusted with an `alpha`-scaled entropy penalty.

<div align="center">
<img src=images\method.jpg width="60%">
</div>

## Code Structure
- `generate_chair.py`: caption generation with MCoT models
- `chair.py`: CHAIR evaluator
- `chair.pkl`: CHAIR cache for evaluation

## Data
The generation input is a JSON list with `image_id` and `instruction` fields:

```json
[
  {
    "image_id": "COCO_val2014_000000000042.jpg",
    "instruction": "Describe this image."
  }
]
```

An example file is included in `examples/coco_images.example.json`.

## Installation
Please follow the official `Qwen2.5-VL` installation instructions: [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).


## Usage
### Generate Captions
```bash
python generate_chair.py \
  --input /path/to/coco_images.json \
  --output /path/to/results/grit3b_chair.json \
  --model_id /path/to/GRIT-3B \
  --image_root /path/to/val2014 \
  --num_samples 500 \
  --max_new_tokens 256 \
  --activation_alpha 0.75 \
  --activation_info_layer -1 \
  --activation_threshold 0.5
```

### Run CHAIR Evaluation
```bash
python chair.py \
  --cap_file /path/to/results/grit3b_chair.json \
  --image_id_key image_id \
  --caption_key caption \
  --cache /path/to/chair.pkl \
  --save_path /path/to/results/grit3b_chair.metrics.json
```

## Main Arguments
- `--activation_alpha`: decoding intervention strength
- `--activation_info_layer`: hidden-state layer used for visual entropy computation
- `--activation_threshold`: threshold `gamma` for current-token intervention
- `--num_samples`: number of samples to run
- `--num_chunks` and `--chunk_index`: split evaluation support

## Output
Generation results are stored as JSON records with the original sample fields and decoded outputs. CHAIR evaluation outputs a JSON file with:

- `sentences`: per-sample CHAIR annotations
- `overall_metrics`: CHAIR metrics for thinking text
- `overall_metrics_answer`: CHAIR metrics for answer text

## Citation
If this repository is useful in your research, please cite:

```bibtex
@article{ma2026understanding,
  title={Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models},
  author={Ma, Ji and Suo, Wei and Wang, Peng and Zhang, Yanning},
  journal={arXiv preprint arXiv:2603.27201},
  year={2026}
}
```

## Acknowledgements
This codebase is based on [Activation_Decoding](https://github.com/hkust-nlp/Activation_Decoding), [middle_layers_indicating_hallucinations](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations), [R1-Onevision](https://github.com/Fancy-MLLM/R1-Onevision), [GRIT](https://github.com/eric-ai-lab/GRIT) and [PixelReasoner](https://github.com/TIGER-AI-Lab/Pixel-Reasoner). Many thanks to the authors for generously sharing their codes!
