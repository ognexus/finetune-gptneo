# Finetune GPT Neo 1.3B Model using custom Dataset on AI Platform Notebook

## Finetune GPT-NEO (1.3B Parameters)

This 1.3B GPT Neo model is fine tuned on a custom dataset. The model training is done on GCP's AI Platform JupyterLab Notebook. 

https://cloud.google.com/notebooks

I have used a 60GB RAM and 1 GPU (NVIDIA Tesla T4 - 16GB). 

```markdown
start = time.time()
print("Training Started")


!deepspeed --num_gpus=1 run_clm.py \
--deepspeed ds_config_gptneo.json \
--model_name_or_path EleutherAI/gpt-neo-1.3B \
--train_file train.txt \
--validation_file validation.txt \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 5 \
--eval_steps 15 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 4 \
--use_fast_tokenizer False \
--learning_rate 5e-06 \
--warmup_steps 10 \
--save_strategy="epoch"

end = time.time()
print("Training complete!")
```

## Generate text with a GPT-NEO 1.3B Parameters model

```python
# credit to Suraj Patil - https://github.com/huggingface/transformers/pull/10848 - modified to create multiple texts and use deepspeed inference

from transformers import GPTNeoForCausalLM, AutoTokenizer
import deepspeed

# casting to fp16 "half" gives a large speedup during model loading
model = GPTNeoForCausalLM.from_pretrained("finetuned").half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained("finetuned")

# using deepspeed inference is optional: it gives about a 2x speed up
deepspeed.init_inference(model, mp_size=1, dtype=torch.half, replace_method='auto')

texts = ["From off a hill whose concave", "Paralell text 2"]

ids = tokenizer(texts, padding=padding, return_tensors="pt").input_ids.to("cuda")


gen_tokens = model.generate(
  ids,
  do_sample=True,
  min_length=0,
  max_length=200,
  temperature=1.0,
  top_p=0.8,
  use_cache=True
)
gen_text = tokenizer.batch_decode(gen_tokens)
print(gen_text)

```

Credits for the repo (already in the License.MD) :- https://github.com/Xirider/finetune-gpt2xl
