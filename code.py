"""
Veena TTS — Full Implementation with Name Pronunciation
=========================================================
Uses `indic-transliteration` (pure Python, no API) to convert
names inside <> tags to Devanagari phonetics before TTS.

Install once:
    pip install snac transformers soundfile indic-transliteration
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import gc
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Audio
from snac import SNAC
import snac.layers as _snac_layers
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ─────────────────────────────────────────────────────────────────────────────
# 1. SNAKE1D PATCH
# ─────────────────────────────────────────────────────────────────────────────

def _snake_eager(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class _Snake1dEager(nn.Module):
    def __init__(self, original: _snac_layers.Snake1d):
        super().__init__()
        self.alpha = original.alpha

    def forward(self, x):
        return _snake_eager(x, self.alpha)

def _patch_snake(module: nn.Module):
    for name, child in list(module.named_children()):
        if type(child) is _snac_layers.Snake1d:
            setattr(module, name, _Snake1dEager(child))
        else:
            _patch_snake(child)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {device}")
if torch.cuda.is_available():
    print(f"GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NAME PRONUNCIATION  (pure Python, no API)
# ─────────────────────────────────────────────────────────────────────────────

# Manual overrides for tricky names — add more as needed
NAME_OVERRIDES = {
    # "EnglishSpelling": "देवनागरी उच्चारण",
    "pramit":    "प्रमित",
    "shriyaa":   "श्रिया",
    "sundarraj": "सुन्दरराज",
    "arjun":     "अर्जुन",
    "kavya":     "काव्या",
    "riya":      "रिया",
    "rohan":     "रोहन",
    "ananya":    "अनन्या",
    "aditya":    "आदित्य",
    "ishaan":    "ईशान",
    "nisha":     "निशा",
    "vivek":     "विवेक",
    "pooja":     "पूजा",
    "rahul":     "राहुल",
    "priya":     "प्रिया",
    "amit":      "अमित",
    "sneha":     "स्नेहा",
    "vishal":    "विशाल",
    "meera":     "मीरा",
    "karan":     "करण",
}

def name_to_devanagari(name: str) -> str:
    """
    Convert a Roman-script name to Devanagari pronunciation.

    Strategy (in order):
      1. Check manual override dict (case-insensitive)
      2. Try ITRANS → Devanagari via indic-transliteration
         (works well for Sanskrit-origin names written phonetically)
      3. Fallback: return original name unchanged
    """
    key = name.strip().lower()

    # 1. Manual override
    if key in NAME_OVERRIDES:
        return NAME_OVERRIDES[key]

    # Handle multi-word names (e.g. "Shriyaa Sundarraj")
    parts = name.strip().split()
    if len(parts) > 1:
        return " ".join(name_to_devanagari(p) for p in parts)

    # 2. ITRANS transliteration (works for phonetically written Indian names)
    try:
        # Normalize: lowercase, map common English spellings to ITRANS
        itrans_name = _english_to_itrans(key)
        devanagari  = transliterate(itrans_name, sanscript.ITRANS, sanscript.DEVANAGARI)
        # Sanity check: must contain actual Devanagari characters
        if any('\u0900' <= c <= '\u097F' for c in devanagari):
            return devanagari
    except Exception:
        pass

    # 3. Fallback: return as-is (TTS will handle it as English)
    return name.strip()


def _english_to_itrans(name: str) -> str:
    """
    Map common English phonetic patterns → ITRANS so transliteration works.
    e.g.  'sh' → 'sh', 'aa' stays 'aa', 'ee' → 'ii', 'oo' → 'uu'
    """
    n = name.lower()
    # Common English → ITRANS vowel mappings
    replacements = [
        ("oo",  "uu"),   # Pooja → puuja
        ("ee",  "ii"),   # Meera → miira
        ("ou",  "u"),    # Gourav → gurav
        ("ksh", "kSh"),  # Daksha → dkSha
        ("gn",  "gn"),
        ("ph",  "ph"),
        ("th",  "th"),
        ("dh",  "dh"),
        ("bh",  "bh"),
        ("gh",  "gh"),
        ("jh",  "jh"),
        ("kh",  "kh"),
        ("ch",  "ch"),
        ("sh",  "sh"),
        ("zh",  "z"),
    ]
    for src, dst in replacements:
        n = n.replace(src, dst)
    return n


def replace_names_with_pronunciation(text: str) -> str:
    """
    Replace all <Name> tags in text with Devanagari pronunciation.
    e.g. "नमस्ते <Pramit>!" → "नमस्ते प्रमित!"
    """
    def process_match(match):
        name          = match.group(1).strip()
        pronunciation = name_to_devanagari(name)
        print(f"  <{name}> → {pronunciation}")
        return pronunciation

    return re.sub(r"<(.*?)>", process_match, text)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEXT PRE-PROCESSING  (numbers → Hindi words)
# ─────────────────────────────────────────────────────────────────────────────

hindi_ones = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
    "सत्रह", "अठारह", "उन्नीस"
]
hindi_tens = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]

def number_to_hindi(n):
    if n == 0:   return "शून्य"
    if n < 20:   return hindi_ones[n]
    if n < 100:
        t, o = divmod(n, 10)
        return hindi_tens[t] + (" " + hindi_ones[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return hindi_ones[h] + " सौ" + (" " + number_to_hindi(r) if r else "")
    if n < 100000:
        th, r = divmod(n, 1000)
        return number_to_hindi(th) + " हज़ार" + (" " + number_to_hindi(r) if r else "")
    if n < 10000000:
        l, r = divmod(n, 100000)
        return number_to_hindi(l) + " लाख" + (" " + number_to_hindi(r) if r else "")
    cr, r = divmod(n, 10000000)
    return number_to_hindi(cr) + " करोड़" + (" " + number_to_hindi(r) if r else "")

def preprocess_hindi(text):
    def rupee_convert(m):
        return number_to_hindi(int(m.group(1).replace(",", ""))) + " रुपये"
    text = re.sub(r'₹\s*(\d[\d,]*)', rupee_convert, text)
    text = text.replace("₹", "")
    text = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group().replace(",", ""), text)
    text = re.sub(r'\b(\d+)\b', lambda m: number_to_hindi(int(m.group())), text)
    return re.sub(r'\s+', ' ', text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD VEENA-TTS MODEL
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "maya-research/veena-tts"

print("\n" + "="*60)
print("Loading Veena-TTS in float16 on GPU …")
print("="*60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    free = torch.cuda.get_device_properties(0).total_memory / 1e9 - used
    print(f"✅ Veena-TTS loaded  | GPU used: {used:.2f} GB | GPU free: {free:.2f} GB")
else:
    print("✅ Veena-TTS loaded on CPU")


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOAD SNAC DECODER
# ─────────────────────────────────────────────────────────────────────────────

snac_device = torch.device("cpu")

print("\nLoading SNAC decoder on CPU …")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(snac_device).eval()
_patch_snake(snac_model)

n_patched   = sum(1 for m in snac_model.modules() if isinstance(m, _Snake1dEager))
n_remaining = sum(1 for m in snac_model.modules() if type(m) is _snac_layers.Snake1d)
print(f"✅ SNAC ready | {n_patched} Snake1d patched | {n_remaining} remaining")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VEENA CONSTANTS & SPEAKERS
# ─────────────────────────────────────────────────────────────────────────────

START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN   = 128258
START_OF_HUMAN_TOKEN  = 128259
END_OF_HUMAN_TOKEN    = 128260
START_OF_AI_TOKEN     = 128261
END_OF_AI_TOKEN       = 128262
AUDIO_CODE_BASE       = 128266
AUDIO_CODE_MAX        = AUDIO_CODE_BASE + 7 * 4096
SAMPLE_RATE           = 24000

ALL_SPEAKERS = ["kavya", "agastya", "maitri", "vinaya"]


# ─────────────────────────────────────────────────────────────────────────────
# 8. SNAC DECODE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def decode_snac_tokens(snac_tokens: list):
    n = len(snac_tokens)
    if n == 0:
        raise ValueError("Empty SNAC token list")
    snac_tokens = snac_tokens[:(n // 7) * 7]
    if not snac_tokens:
        raise ValueError("No complete SNAC frames")

    offsets = [AUDIO_CODE_BASE + i * 4096 for i in range(7)]
    lvl0, lvl1, lvl2 = [], [], []
    for i in range(0, len(snac_tokens), 7):
        lvl0.append(snac_tokens[i]   - offsets[0])
        lvl1.append(snac_tokens[i+1] - offsets[1])
        lvl1.append(snac_tokens[i+4] - offsets[4])
        lvl2.append(snac_tokens[i+2] - offsets[2])
        lvl2.append(snac_tokens[i+3] - offsets[3])
        lvl2.append(snac_tokens[i+5] - offsets[5])
        lvl2.append(snac_tokens[i+6] - offsets[6])

    codes = []
    for lvl_name, lvl in [("level-0", lvl0), ("level-1", lvl1), ("level-2", lvl2)]:
        t = torch.tensor(lvl, dtype=torch.long, device=snac_device).unsqueeze(0)
        if torch.any((t < 0) | (t > 4095)):
            raise ValueError(f"Out-of-range SNAC values in {lvl_name}")
        codes.append(t)

    with torch.no_grad():
        audio_hat = snac_model.decode(codes)
    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 9. GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_audio(text: str, speaker: str):
    if speaker not in ALL_SPEAKERS:
        raise ValueError(f"Unknown speaker '{speaker}'. Choose from: {ALL_SPEAKERS}")

    prompt        = f"<spk_{speaker}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN,
    ]
    input_ids      = torch.tensor([input_tokens], device=model.device)
    max_new_tokens = min(int(len(text) * 1.3) * 7 + 21, 2048)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            use_cache=True,
        )

    new_tokens  = generated_ids[0][len(input_tokens):].tolist()
    snac_tokens = [t for t in new_tokens if AUDIO_CODE_BASE <= t < AUDIO_CODE_MAX]

    if not snac_tokens:
        raise ValueError("Model produced no audio tokens — try increasing max_new_tokens.")

    return decode_snac_tokens(snac_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SCRIPT — edit your RAW_SCRIPT here, wrap names in < >
# ─────────────────────────────────────────────────────────────────────────────

RAW_SCRIPT = (
    "नमस्ते ! आपको यहाँ देखकर अच्छा लगा <tejas>! मैं इस app में आपको investment "
    "शुरू करने के लिये guide कर सकती हूँ। यहाँ पर आप S.I.P शुरू कर सकते हैं, या , "
    "हमारे best performing funds देख सकते हैं, या अपने हिसाब से funds भी explore कर "
    "सकते हैं। आप क्या करना चाहेंगे?"
)

# Pipeline: names → numbers → TTS
print("\nResolving names...")
NAME_RESOLVED = replace_names_with_pronunciation(RAW_SCRIPT)
CLEAN_SCRIPT  = preprocess_hindi(NAME_RESOLVED)

print(f"\nRaw     : {RAW_SCRIPT}")
print(f"Names   : {NAME_RESOLVED}")
print(f"Cleaned : {CLEAN_SCRIPT}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. GENERATE — all 4 speakers
# ─────────────────────────────────────────────────────────────────────────────

OUT_BASE = "/kaggle/working/veena_audio/investment_greeting"
for spk in ALL_SPEAKERS:
    os.makedirs(f"{OUT_BASE}/{spk}", exist_ok=True)

results = {}

print("\n" + "="*60)
print(f"GENERATING  ({len(ALL_SPEAKERS)} speakers)")
print("="*60)

for idx, speaker in enumerate(ALL_SPEAKERS, 1):
    print(f"\n[{idx}/{len(ALL_SPEAKERS)}] Speaker: {speaker.upper()}")
    try:
        torch.cuda.empty_cache()
        gc.collect()

        audio_arr = generate_audio(CLEAN_SCRIPT, speaker=speaker)
        duration  = len(audio_arr) / SAMPLE_RATE
        out_path  = f"{OUT_BASE}/{speaker}/investment_greeting.wav"
        sf.write(out_path, audio_arr, SAMPLE_RATE)

        results[speaker] = ("ok", out_path)
        print(f"  ✓ Saved: {out_path}  ({duration:.1f}s)")

    except Exception as e:
        results[speaker] = ("fail", str(e))
        print(f"  ✗ Failed: {e}")
        torch.cuda.empty_cache()
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 12. PLAYBACK
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PLAYBACK")
print("="*60)

for speaker in ALL_SPEAKERS:
    status, val = results[speaker]
    print(f"\n▶ {speaker.upper()} ({'✓ Generated' if status == 'ok' else '✗ Failed'})")
    if status == "ok":
        display(Audio(val))
    else:
        print(f"  Error: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

ok_list   = [s for s, (st, _) in results.items() if st == "ok"]
fail_list = [s for s, (st, _) in results.items() if st == "fail"]

print("\n" + "="*60)
print(f"  ✓ Generated : {len(ok_list)}/{len(ALL_SPEAKERS)} — {ok_list}")
if fail_list:
    print(f"  ✗ Failed    : {len(fail_list)}/{len(ALL_SPEAKERS)} — {fail_list}")
print("="*60)
