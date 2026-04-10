# Latxa-Qwen-Propella-Annotator

This repository contains the scripts for fine-tuning and deploying **Latxa-Qwen**, specifically adapted for structured document annotation. By combining **LoRA (Low-Rank Adaptation)** for efficient training and **vLLM** for high-performance inference, this project transforms Basque (Euskara) text into validated, machine-readable JSON metadata.

The project provides a pipeline to annotate documents using the `HiTZ/Latxa-Qwen3-VL-4B-Instruct` base model. It is designed to handle complex Basque linguistic structures and output data according to the **Propella** schema.

### Key Components
- **Training:** Uses `TRL` and `SFTTrainer` with LoRA to adapt the model to specific JSON structures.
- **Inference:** Optimized via `vLLM` using `StructuredOutputsParams` for guaranteed JSON validity.
- **Robustness:** Includes a retry-loop mechanism and token-aware truncation logic.

## 📖 Example Usage

As the model was too heavy for GitHub, download it from the following [link](https://upvehueus-my.sharepoint.com/:f:/g/personal/mmolina030_ikasle_ehu_eus/IgD4ezvep7UdS6Iq4TJdvpggAZkFMV9Hi_OUma_vp5GLCvM?e=yJiIGg) and add it to the root folder.

This example demonstrates how to use the fine-tuned LoRA adapter with **vLLM** to process a Basque text and extract structured metadata.

```python
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams
from propella import (
    create_messages, 
    AnnotationResponse, 
    get_annotation_response_schema
)

# 1. Setup paths and parameters
MODEL_ID = "HiTZ/Latxa-Qwen3-VL-4B-Instruct"
ADAPTER_PATH = "latxa-qwen-final-adapter"

# 2. Initialize the model with LoRA support
llm = LLM(
    model=MODEL_ID, 
    max_model_len=8192, 
    enable_lora=True, 
    max_loras=1, 
    gpu_memory_utilization=0.8
)

# 3. Define the structured output schema
schema = get_annotation_response_schema(flatten=True)
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json=schema), 
    max_tokens=300
)
lora_request = LoRARequest("latxa-adapter", 1, ADAPTER_PATH)

# 4. Prepare your input (Basque text)
input_text = "Nahiz eta herri arrantzalea izan, Ondarroan porturik ez zegoen; itsasontziak itsasadarrean amarratzen ziren, Zubi Zaharraren inguruan.<ref>{{Erreferentzia|izenburua=Ondarroako Udala - Monumentos e itinerarios|url=http://www.ondarroa.eus/eu-ES/Turismoa/MonumenIbilbide_monumentuaktos_itinerarios/Orrialdeak/default.aspx|aldizkaria=www.ondarroa.eus|sartze-data=2019-05-22}}</ref> [[1934|1934an]]an Ondarroako kanpo portu berriaren lanak hasi ziren, Juan Egidazu injeniariaren zuzendaritzapean. Lehenengo zabaldura Kanttoipetik Alamedaraino egin zen; bigarren fase batean, Alamedatik Arta muturreraino.\n[[Espainiako Gerra Zibila|Espainiako Gerra Zibilaren]]ren ondoren, portu berri horri esker Ondarroako portuak hazkunde handia bizi izan zuen."
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    create_messages(input_text), 
    tokenize=False, 
    add_generation_prompt=True
)

# 5. Generate and Validate
output = llm.generate([prompt], sampling_params, lora_request=lora_request)
raw_text = output[0].outputs[0].text

# Use the Propella Pydantic model to ensure data integrity
result = AnnotationResponse.model_validate_json(raw_text)
print(json.dumps(result.model_dump(), indent=4, ensure_ascii=False))
```

**Input**

Nahiz eta herri arrantzalea izan, Ondarroan porturik ez zegoen; itsasontziak itsasadarrean amarratzen ziren, Zubi Zaharraren inguruan. 1934an Ondarroako kanpo portu berriaren lanak hasi ziren, Juan Egidazu injeniariaren zuzendaritzapean. Lehenengo zabaldura Kanttoipetik Alamedaraino egin zen; bigarren fase batean, Alamedatik Arta muturreraino. Espainiako Gerra Zibilaren ondoren, portu berri horri esker Ondarroako portuak hazkunde handia bizi izan zuen.




**Output**

```json
{
    "text": "Nahiz eta herri arrantzalea izan, Ondarroan porturik ez zegoen; itsasontziak itsasadarrean amarratzen ziren, Zubi Zaharraren inguruan.<ref>{{Erreferentzia|izenburua=Ondarroako Udala - Monumentos e itinerarios|url=http://www.ondarroa.eus/eu-ES/Turismoa/MonumenIbilbide_monumentuaktos_itinerarios/Orrialdeak/default.aspx|aldizkaria=www.ondarroa.eus|sartze-data=2019-05-22}}</ref> [[1934|1934an]]an Ondarroako kanpo portu berriaren lanak hasi ziren, Juan Egidazu injeniariaren zuzendaritzapean. Lehenengo zabaldura Kanttoipetik Alamedaraino egin zen; bigarren fase batean, Alamedatik Arta muturreraino.\n[[Espainiako Gerra Zibila|Espainiako Gerra Zibilaren]]ren ondoren, portu berri horri esker Ondarroako portuak hazkunde handia bizi izan zuen.",
    "content_integrity": "mostly_complete",
    "content_ratio": "mostly_content",
    "content_length": "minimal",
    "one_sentence_description": "Historical description of the new port construction on Ondarroa's coastline following the 1934 battle.",
    "content_type": [
        "reference",
        "instructional"
    ],
    "business_sector": [
        "government_public",
        "transportation_logistics"
    ],
    "technical_content": [
        "non_technical"
    ],
    "information_density": "adequate",
    "content_quality": "good",
    "audience_level": "general",
    "commercial_bias": "none",
    "time_sensitivity": "evergreen",
    "content_safety": "safe",
    "educational_value": "basic",
    "reasoning_indicators": "basic_reasoning",
    "pii_presence": "no_pii",
    "regional_relevance": [
        "european"
    ],
    "country_relevance": [
        "spain"
    ]
}
```

**Evaluation**

We evaluate our model in a test set containing 20 documents. These documents were human-annotated, which explains the little amount of instances. However, it is to be extended in the foreseeable future.

![jaccard](https://github.com/LDRMoliner/deep_learning_annotator/blob/main/jaccard_comparison_grouped.png)

![qwl](https://github.com/LDRMoliner/deep_learning_annotator/blob/main/qwk_comparison_grouped.png)




