import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams
from propella import (
    create_messages,
    AnnotationResponse,
    get_annotation_response_schema,
)

# Setting up the adapter, tokens, and model parameters

ADAPTER_PATH = "latxa-qwen-final-adapter"
SYSTEM_PROMPT_TOKENS = 685 + 25
MAX_TOKENS = 300
MAX_MODEL_LEN = 8192
TRIES = 3
# Document to annotate

input_text = "Nahiz eta herri arrantzalea izan, Ondarroan porturik ez zegoen; itsasontziak itsasadarrean amarratzen ziren, Zubi Zaharraren inguruan.<ref>{{Erreferentzia|izenburua=Ondarroako Udala - Monumentos e itinerarios|url=http://www.ondarroa.eus/eu-ES/Turismoa/MonumenIbilbide_monumentuaktos_itinerarios/Orrialdeak/default.aspx|aldizkaria=www.ondarroa.eus|sartze-data=2019-05-22}}</ref> [[1934|1934an]]an Ondarroako kanpo portu berriaren lanak hasi ziren, Juan Egidazu injeniariaren zuzendaritzapean. Lehenengo zabaldura Kanttoipetik Alamedaraino egin zen; bigarren fase batean, Alamedatik Arta muturreraino.\n[[Espainiako Gerra Zibila|Espainiako Gerra Zibilaren]]ren ondoren, portu berri horri esker Ondarroako portuak hazkunde handia bizi izan zuen."

# --- Initialize Model ---
llm = LLM(model="HiTZ/Latxa-Qwen3-VL-4B-Instruct", max_model_len=8192, enable_lora=True, max_loras=1, gpu_memory_utilization=0.8)
tokenizer = llm.get_tokenizer()

schema = get_annotation_response_schema(flatten=True)
guided_json = StructuredOutputsParams(json=schema)
sampling_params = SamplingParams(structured_outputs=guided_json, max_tokens=MAX_TOKENS)
lora_request = LoRARequest("latxa-adapter", 1, ADAPTER_PATH)

# --- Token Truncation Logic ---
text_tokens = len(tokenizer.encode(input_text))
if text_tokens + SYSTEM_PROMPT_TOKENS + MAX_TOKENS > MAX_MODEL_LEN:
    input_text = tokenizer.decode(tokenizer.encode(input_text)[:MAX_MODEL_LEN - SYSTEM_PROMPT_TOKENS - MAX_TOKENS])

# --- Inference ---


prompt = tokenizer.apply_chat_template(create_messages(input_text), tokenize=False, add_generation_prompt=True)

# Sometimes, the model might not return valid JSON due to generation issues. To handle this, we can implement a retry mechanism that attempts inference multiple times until we get a valid response or exhaust our attempts. This is especially useful when working with structured outputs, as it ensures we can still obtain the desired information even if the model's output is occasionally malformed.

for i in range(TRIES):

    print(f"\n--- Inference Attempt {i+1} ---")

    output = llm.generate([prompt], sampling_params, lora_request=lora_request)

    # --- Process Result ---
    try:
        raw_text = output[0].outputs[0].text
        response_content = AnnotationResponse.model_validate_json(raw_text)
        break  # Exit loop if parsing is successful

    except Exception as e:
        print(f"\n--- Error processing inference ---")
        print(f"Error Message: {e}")
        print(f"Raw output: '{output[0].outputs[0].text}'")

            
# Merge input with the validated JSON response
final_result = {"text": input_text} | response_content.model_dump(mode='json')

print("\n--- Inference Result ---")
print(json.dumps(final_result, indent=4, ensure_ascii=False))