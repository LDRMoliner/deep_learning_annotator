from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch
import json
from datasets import load_dataset

model_id = "HiTZ/Latxa-Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)


# Para completion_only_loss

def prepare_dataset_modern(example):
    all_keys = list(example.keys())
    try:
        start_idx = all_keys.index("content_integrity")
        metadata_keys = all_keys[start_idx:]
    except ValueError:
        metadata_keys = []

    prompt_text = example['text']

    answer_dict = {key: example[key] for key in metadata_keys}
    completion_text = json.dumps(answer_dict, ensure_ascii=True)

    return {
        "prompt": f"{prompt_text}\n",
        "completion": f"{completion_text}"
    }


data_files = {
    "train": "/home/mmolina030/propella_annotations/colossal-oscar-annotated-v2/train_00000.jsonl",
    "validation": "/home/mmolina030/propella_annotations/colossal-oscar-annotated-v2/valid_00000.jsonl"
}

dataset = load_dataset("json", data_files=data_files)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

train_dataset = train_dataset.map(prepare_dataset_modern, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(prepare_dataset_modern, remove_columns=val_dataset.column_names)

print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(val_dataset)}")

print("Instance example:")
print(train_dataset[0])

######


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="./latxa-lora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_total_limit=20,
    num_train_epochs=1,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    save_strategy="steps",
    max_length=2048,
    completion_only_loss=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    peft_config=lora_config,
    processing_class=processor
)

trainer.train()
trainer.save_model("./latxa-qwen-final-adapter")
