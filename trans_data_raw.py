# fast_translate_csv_m2m100.py
# Run trên Kaggle (P100)
import os, gc, sys, time
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ---------- CONFIG ----------
INPUT_CSV = "/kaggle/input/bitext-retail/bitext-retail-banking-llm-chatbot-training-dataset.csv"
OUTPUT_CSV = "/kaggle/working/bitext_translated_safe.csv"
COL_INSTR = "instruction"
COL_RESP = "response"

START_BATCH_SIZE = 256   # <-- bạn thay giá trị theo GPU
CHUNK_ROWS = 2048
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 512
NUM_BEAMS = 1            # greedy = nhanh
SRC_LANG = "en"
TGT_LANG = "vi"
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Start batch size (user):", START_BATCH_SIZE)

# load model & tokenizer M2M100-418M
print("Loading M2M100-418M model/tokenizer...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)
if DEVICE == "cuda":
    model = model.half()
model.eval()

# set tokenizer source language
tokenizer.src_lang = SRC_LANG
forced_bos_token_id = tokenizer.get_lang_id(TGT_LANG)
print(f"Forced target language: {TGT_LANG} (id={forced_bos_token_id})")

# helpers
def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _generate_on_device(input_ids, attention_mask, device, gen_kwargs):
    with torch.inference_mode():
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

def translate_list_with_oom_handling(texts, start_batch, device, max_input_length=512, max_output_length=512, num_beams=1):
    n = len(texts)
    if n == 0:
        return []
    results = []
    i = 0
    cur_batch = max(1, int(start_batch))
    gen_kwargs = dict(max_length=max_output_length, num_beams=num_beams, forced_bos_token_id=forced_bos_token_id)
    moved_to_cpu = False

    try:
        while i < n:
            b = min(cur_batch, n - i)
            batch_texts = texts[i:i+b]
            try:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                out = _generate_on_device(input_ids, attention_mask, device, gen_kwargs)
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                results.extend(decoded)
                del input_ids, attention_mask, inputs, out
                if device == "cuda":
                    torch.cuda.empty_cache()
                i += b
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    print(f"[OOM] GPU OOM at batch_size={cur_batch}, reducing batch...")
                    free_gpu()
                    if cur_batch > 1:
                        cur_batch = max(1, cur_batch // 2)
                        continue
                    else:
                        if device == "cuda":
                            print("[OOM] Single-item OOM on GPU. Fallback CPU per item.")
                            model.to("cpu")
                            moved_to_cpu = True
                            device = "cpu"
                            continue
                        else:
                            raise
                else:
                    raise
    finally:
        if moved_to_cpu and torch.cuda.is_available():
            try:
                model.to("cuda")
                free_gpu()
            except Exception as e:
                print("Warning: failed to move model back to CUDA:", e)

    return results

# main loop
def process_csv(input_csv, output_csv, start_batch, chunk_rows=2048):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    reader = pd.read_csv(input_csv, chunksize=chunk_rows, iterator=True, dtype=str, keep_default_na=False, encoding="utf-8")
    wrote_header = False
    total = 0

    for chunk in tqdm(reader, desc="chunks"):
        if COL_INSTR not in chunk.columns or COL_RESP not in chunk.columns:
            raise KeyError(f"Input CSV must contain columns '{COL_INSTR}' and '{COL_RESP}'")
        n = len(chunk)
        total += n

        texts_instr = chunk[COL_INSTR].fillna("").tolist()
        texts_resp  = chunk[COL_RESP].fillna("").tolist()

        # Translate instructions
        translated_instr = translate_list_with_oom_handling(texts_instr, start_batch, DEVICE,
                                                           max_input_length=MAX_INPUT_LENGTH,
                                                           max_output_length=MAX_OUTPUT_LENGTH,
                                                           num_beams=NUM_BEAMS)
        # Translate responses
        translated_resp = translate_list_with_oom_handling(texts_resp, start_batch, DEVICE,
                                                          max_input_length=MAX_INPUT_LENGTH,
                                                          max_output_length=MAX_OUTPUT_LENGTH,
                                                          num_beams=NUM_BEAMS)
        # assign back
        chunk[COL_INSTR] = translated_instr
        chunk[COL_RESP]  = translated_resp

        # write CSV
        if not wrote_header:
            chunk.to_csv(output_csv, index=False, mode="w", header=True, encoding="utf-8-sig")
            wrote_header = True
        else:
            chunk.to_csv(output_csv, index=False, mode="a", header=False, encoding="utf-8-sig")

        del chunk, texts_instr, texts_resp, translated_instr, translated_resp
        free_gpu()
        time.sleep(0.05)

    print("Done. Processed rows:", total)
    print("Output:", output_csv)

# Run
if __name__ == "__main__":
    print("Begin processing CSV with START_BATCH_SIZE =", START_BATCH_SIZE)
    process_csv(INPUT_CSV, OUTPUT_CSV, START_BATCH_SIZE, chunk_rows=CHUNK_ROWS)
