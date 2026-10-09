"""Retention suite: the frozen item generator, sealed draws and the freshness harvest. No model, no tokenizer (except the
optional report-only --tokens audit, which reads the tokenizer file through the `tokenizers` library), no torch.

  python -B scripts/ret_gen.py freeze --approval "<who, when, DECISIONS entry>"   # RET_FREEZE.json (create-only)
  python -B scripts/ret_gen.py preview                           # pre-freeze audit of BOTH dev draws (one harvest pass)
                                                                 # -> DEV_DRAW_PREVIEW.json, DEV_DRAW_PREVIEW_SD.json
  python -B scripts/ret_gen.py mint --purpose dev                # the dev draw: fixed DEV_SEED, 8 categories x 3 levels x 75
  python -B scripts/ret_gen.py mint --purpose dev --surface-diversity    # the variant's dev draw: fixed DEV_SEED_SD
  python -B scripts/ret_gen.py mint --purpose confirm            # a sealed confirmation draw: fresh OS-entropy seed, the
                                                                 # dev selection's categories at their selected levels
  python -B scripts/ret_gen.py verify --draw <draw.jsonl>        # regenerate from the seal; sha256 must match
  python -B scripts/ret_gen.py stats  --draw <draw.jsonl> [--tokens]
  options: --surface-diversity (paraphrased instructions/system prompts and field-name variants; OFF by default;
           recorded in the seal; each setting has its own pre-registered dev seed and its own dev selection, and a
           confirmation draw uses the selection of its setting), --no-scan (tests only: the seal records the scan as
           skipped and the runner refuses such a draw outside tests)

Determinism. A draw is a pure function of (seed, plan {category: [levels]}, surface_diversity, this code). Every cell
(category, level) has its own random.Random seeded with the string f"{seed}|{category}|L{level}|sd{0|1}" (str seeds are
hashed with SHA-512 by CPython, so the stream is independent of PYTHONHASHSEED and of the other cells). The file is
UTF-8 JSONL, one prompt per line, fixed key order, '\\n' newlines. Same seed -> same bytes -> same sha256.

Items. 8 categories x 75 items per level (docs/retention/DESIGN.md section 3). unknown_twin items are PAIRS (an
answerable twin 'a' and an unanswerable twin 'u' over the same passage): 150 prompts, and the item is correct only if
both twins are. Every other item is one prompt. Answers are computed by code; ret_grade.py grades them. Subtypes are
STRATIFIED: each cell's 75 items are dealt round-robin over the cell's strata (STRATA) and shuffled, so every draw of
a cell (dev or confirmation) has the same subtype mix. Questions about a passage record are in the present tense, so
the gold of a corrected record (its latest value) is the only reading.

Freshness (the seal records it; minting refuses on any gated hit):
  F1 gate   no prompt's normalised user turn equals a harvested string or a user turn of any earlier draw
  F2 gate   no item-unique sentence of >= 8 words (a sentence that occurs in exactly one item of the draw) equals a
            sentence of a harvested string; no passage sentence of >= 8 words is shared with an earlier draw
  F3 report the same checks for 6-7-word and template (shared) sentences; operative-content overlaps with earlier
            draws (category, subtype, level and the numbers/words that decide the answer)
  F0 gate   no prompt contains a transition payload ({"before", "event"}): the programme's payload harvest cannot
            overlap a retention item, and this is checked rather than assumed.
The harvest walks the retention-records tree exactly as scripts/l1_make_ood.harvest_historical does (rglob *.jsonl then
*.json; files named HSKILL_CONFIRM_V2* are skipped BY NAME before any stat or open, so the reserved HSKILL-CONFIRM-002
is never opened or hashed (DECISIONS I-28); files > 60 MB skipped), and in addition RCWM-T1's data/ and docs/l1,
docs/prereg3 prompt sets and the Rineya training / lab corpora listed in SCAN_SOURCES. Every string value of >= 16
characters is normalised (ret_common.norm_text) and compared by exact equality, whole and sentence by sentence.
Nothing is selected on harvested content: a hit only refuses the mint (mint again with a new seed).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if __name__ == "__main__":                         # as a CLI nothing here touches the driver (empty string, never "-1");
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # as an import (ret_run.py's GPU path) the environment is left alone
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ret_common as RC  # noqa: E402
import ret_grade as RG  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEN_CODE = ("scripts/ret_common.py", "scripts/ret_gen.py", "scripts/ret_grade.py")
BUSINESS = Path(r"C:/Users/clclc/BusinessProject")
RECORDS = BUSINESS / "rineya-dataset-retention-records"      # == l1_make_ood.RECORDS
CB = BUSINESS / "Chat-Bubble"
SCAN_SOURCES = (   # (name, root, walk identical to l1_make_ood.harvest_historical)
    ("records", RECORDS, True),
    ("rcwm_data", ROOT / "data", False),
    ("rcwm_docs_l1", ROOT / "docs" / "l1", False),
    ("rcwm_docs_prereg3", ROOT / "docs" / "prereg3", False),
    ("corpus_v2", CB / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0", False),
    ("corpus_v3", CB / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0", False),
    ("short_response_corpus", CB / "Rineya_Short_Response_Behavior_Corpus_v1.0", False),
    ("short_response_model_data", CB / "short-response-model-phase-0-1-2" / "data", False),
    ("long_context_demo", CB / "Rineya_Long_Context_Demo_v1", False),
    ("puzzle_51k", CB / "Rineya-Puzzle-Reasoning-51k", False),
    ("lcr_sft", CB / "Rineya_LCR_SFT_v1", False),
    ("invisible_limit_lab", CB / "Rineya_Invisible_Limit_Lab_v1", False),
    ("bwm_bundle", BUSINESS / "WorldModel" / "Rineya_BWM_001_Research_Bundle", False),   # BWM data, banks, runs
)
SKIP_DIRS = {".git", ".claude", ".venv", "venv", "node_modules", "__pycache__"}
MAX_JSON_BYTES = 60_000_000
MIN_STR = 16
GATE_WORDS, REPORT_WORDS = 8, 6
TOKENIZER_JSON = RC.TOKENIZER_DIR / "tokenizer.json"                 # pinned (RC.TOKENIZER_PINS)
PASSAGE_TOKENS = (150, 300)

# ------------------------------------------------------------------------------------------------ vocabularies
FIRST = ("Ines Tomas Priya Kofi Mei Arjun Leila Mateo Sanna Dario Ama Yusuf Hana Bram Oksana Ravi Elif Jonas Noor "
         "Tariq Lena Felipe Aiko Bjorn Chioma Diego Esra Farah Goran Hamid Ilse Jasper Kiri Lucia Malik Nadia Olu Paz "
         "Quentin Rosa Sven Tala Uma Viktor Wen Ximena Yara Zane Abebe Bianca Cyrus Dina Emeka Freya Gustavo Hiro "
         "Imani Joaquin Kalani Leon Maren Nikhil Odile Pavel Qiana Rafael Siobhan Teodor Ulla Valentina Wiremu Xavier "
         "Yasmin Zofia Anouk Bastian Celeste Desmond Eszter Fintan Greta Hector Ingrid Jun Katya Lorenzo Marisol Nils "
         "Orla Pedro Rania Soren Tamsin Ugo Vera Wanda Yosef Zara Alma Benedikt Carmen Dmitri Elena Femi Gita Hugo "
         "Isak Jana Kenji Lior Mira Nuno Oren Petra Rhea Selin Tobias Una Vikram Wilma Yuki Zeynep Ayaan Brigid Caio "
         "Dalia Erik Fatima Gael Hedda Ilan Juno Kasia Luca Mosi Neha Otto Pilar Ronan Saba Thiago Ulrike Vida Wout "
         "Yael Zuri").split()
LAST = ("Okafor Varga Mbeki Lindqvist Arslan Moreau Tanaka Haddad Kowalski Ferreira Nakamura Osei Petrov Quispe "
        "Rahman Sandoval Takahashi Udeh Vasquez Wojcik Yilmaz Zamora Abara Bergstrom Castillo Dimitrov Eze Falk "
        "Gallo Hoang Ibsen Jovanovic Kader Lacroix Mensah Novak Ortega Park Ruiz Sato Tembo Ulloa Vidal Weber Xu "
        "Yamada Zielinski Achebe Bauer Cruz Dahl Engel Fonseca Grimaldi Holm Iqbal Janssen Kaur Lund Mwangi Nyberg "
        "Oyelaran Pires Rossi Silva Thorsen Uddin Valdez Wiik Yoon Zahir Adeyemi Brandt Cortes Delgado Ekstrom "
        "Fiore Gupta Hayes Ishikawa Jensen Kamau Lopes Mori Nieminen Obi Paredes Rinaldi Strand Tavares Unger "
        "Vogel Whitaker Yeo Zanetti Ansah Bianchi Costa Duarte Erdem Fischer Garza Hallberg Ito Juarez Kiplagat "
        "Lehtonen Marsh Ngata Ozturk Pellegrini Rasmussen Soto Toivonen Urquhart Varela Wren Yusupov Zubiri").split()
PLACES = ("Harlow Bay", "Elmstead", "Corran Vale", "Brindle Cove", "Ashby Fen", "Kestrel Point", "Larkfield",
          "Morrow Heath", "Nettlecombe", "Oakhurst", "Pennick", "Quarry Bend", "Rushmere", "Saltway", "Thornbury Mill",
          "Upwell", "Vellacott", "Wexcombe", "Yarrow Glen", "Alder Reach", "Birchmoor", "Coldharbour", "Dunmere",
          "Eastwick Down", "Fallowfield", "Greystone", "Hollins Cross", "Ivybridge Lane", "Juniper Hill", "Kingsmere")
OBJS = (("stamps", "stamp"), ("marbles", "marble"), ("seashells", "seashell"), ("postcards", "postcard"),
        ("stickers", "sticker"), ("pencils", "pencil"), ("beads", "bead"), ("buttons", "button"), ("apples", "apple"),
        ("eggs", "egg"), ("tickets", "ticket"), ("bottles", "bottle"), ("cards", "card"), ("batteries", "battery"),
        ("candles", "candle"), ("books", "book"), ("jars", "jar"), ("crayons", "crayon"), ("rocks", "rock"),
        ("keys", "key"), ("plums", "plum"), ("badges", "badge"), ("ribbons", "ribbon"), ("envelopes", "envelope"),
        ("lemons", "lemon"), ("bricks", "brick"), ("tiles", "tile"), ("cups", "cup"), ("spoons", "spoon"),
        ("magnets", "magnet"))
WORDS = ("anchor basket candle dragon engine falcon garden hammer island jacket kettle ladder magnet needle orange "
         "pepper quiver rocket saddle tunnel velvet walnut yogurt zipper bridge castle desert forest glacier harbor "
         "lantern meadow orchard pillow river shadow temple valley window button carpet donkey feather ginger helmet "
         "insect jungle koala lemon mirror napkin oyster parrot rabbit salmon tomato violin wizard beacon cactus "
         "dagger ember fossil goblet hornet icicle jigsaw kitten locket marble nectar otter pebble quartz raisin "
         "sparrow thistle urchin vessel willow yarrow zephyr acorn bamboo cobalt dolphin easel fiddle gravel hazel "
         "iris jasmine kayak lagoon mango nutmeg olive puffin quill radish spruce turnip vapor wombat badger canyon "
         "dahlia granite hollow ivory juniper kernel lilac mortar nickel onion prism rattle sorrel tulip beetle comet "
         "denim funnel gazelle hamlet inkwell jester kiosk lobster muffin nomad orbit pigeon quarry ribbon satin "
         "trumpet saffron tinsel pickle pretzel biscuit crayon compass cushion shovel saucer thimble wallet whistle "
         "sandal pumpkin blanket chimney cricket meteor pirate tractor buckle cabbage dimple goblin harness jelly "
         "lizard monkey noodle peanut puzzle rascal sponge timber velcro waffle").split()
CITIES = ("Lisbon Oslo Quito Nairobi Hanoi Lima Tunis Dakar Perth Bergen Cusco Accra Kyoto Porto Riga Tallinn Malmo "
          "Turku Gdansk Leeds Cork Galway Lyon Nantes Graz Brno Split Varna Izmir Pune Mysore Kandy Davao Cebu Hobart "
          "Nelson Dunedin Halifax Regina Tucson Boise Fresno Akron Salem Laredo").split()
CROPS = ("carrots leeks beans kale peas squash onions radishes beets chard lettuce garlic potatoes peppers tomatoes "
         "cabbage spinach turnips parsnips fennel celery melons").split()
COLOURS = ("amber crimson teal violet olive navy ochre coral ivory maroon scarlet indigo silver bronze lime peach plum "
           "slate rust mint").split()
INSTRUMENTS = ("cello violin oboe flute drums banjo harp piano tuba clarinet trumpet viola bassoon ukulele saxophone "
               "accordion mandolin trombone").split()
EVENTS = ("concert", "workshop", "ferry crossing", "board meeting", "pottery class", "chess match", "bake sale",
          "science fair", "film screening", "choir rehearsal", "book club", "yoga class", "planning session",
          "garden tour", "debate", "craft fair", "lecture", "swim meet", "quiz night", "dance class")
THINGS = ("lamp", "backpack", "kettle", "scarf", "chair", "teapot", "bicycle bell", "notebook", "rug", "clock",
          "toolbox", "radio", "jacket", "umbrella", "blender")
ORD_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}

# ------------------------------------------------------------------------------------------------ surface forms
# Index 0 is the default (surface diversity OFF). With --surface-diversity every item draws a variant per slot.
SYSTEM = ("Answer the user's question. Give only the final answer in the requested format, with no explanation.",
          "Respond with the final answer only, in exactly the format the user asks for. Do not explain.",
          "Give just the answer, formatted as requested, and nothing else.")
I_NUM = ("Answer with a number only.", "Reply with just the number.", "Give only the number as your answer.")
I_CLOCK = ("Give the time in 24-hour HH:MM format.", "Reply with the time only, as HH:MM on a 24-hour clock.",
           "Answer with the 24-hour time in the form HH:MM.")
I_WEEKDAY = ("Answer with the day of the week only.", "Reply with just the name of the day.",
             "Give only the day of the week.")
I_DATE = ("Answer with the month and day only.", "Reply with just the month and the day.",
          "Give only the month and the day of the month.")
I_SORT_NUM = ("Reply with the sorted numbers separated by commas.", "Give the numbers in that order, separated by commas.",
              "Answer with the sorted list, using commas between numbers.")
I_SORT_WORD = ("Reply with the sorted words separated by commas.", "Give the words in that order, separated by commas.",
               "Answer with the sorted list, using commas between words.")
I_RATIO = ("Reply in the form a:b.", "Give the answer as a:b.", "Answer in the format a:b.")
I_KV = ("Reply with the answer only.", "Give just the requested value.", "Answer with the value only.")
I_UNK = ("Answer using only the passage. If the passage does not contain the answer, reply with exactly UNKNOWN.",
         "Use only the passage above. When the passage does not give the answer, reply with exactly UNKNOWN.",
         "Rely only on the passage. If it does not state the answer, reply with exactly UNKNOWN.")
I_YESNO = ("Answer with one word: yes or no.", "Reply with only yes or no.", "Give a one-word answer, yes or no.")


class Surf:
    """Surface-form chooser: variant 0 unless surface diversity is on."""

    def __init__(self, rng: random.Random, diversity: bool):
        self.rng, self.on = rng, diversity

    def pick(self, opts):
        return opts[self.rng.randrange(len(opts))] if self.on else opts[0]


# ------------------------------------------------------------------------------------------------ helpers
def full_name(rng, used: set | None = None, first_pool=FIRST, last_pool=LAST) -> str:
    for _ in range(1000):
        n = f"{rng.choice(first_pool)} {rng.choice(last_pool)}"
        if used is None or n not in used:
            if used is not None:
                used.add(n)
            return n
    raise RuntimeError("name pool exhausted")


def two_people(rng):
    a = rng.choice(FIRST)
    b = rng.choice([x for x in FIRST if x != a])
    return a, b


def two_full_names(rng):
    """Two full names with different first names (the item's person is identified by a full name, so short templates
    have a large space and independent draws practically never share a user turn)."""
    a = full_name(rng)
    b = full_name(rng, first_pool=tuple(x for x in FIRST if x != a.split()[0]))
    return a, b


def hhmm(m: int) -> str:
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def fmt_num(x) -> str:
    """Integers plain; finite decimals without trailing zeros (exact)."""
    if isinstance(x, int):
        return str(x)
    from fractions import Fraction
    f = Fraction(x)
    if f.denominator == 1:
        return str(f.numerator)
    s = f"{float(f):.6f}".rstrip("0").rstrip(".")
    if RG.parse_rational(s) != f:
        raise ValueError(f"not a short finite decimal: {x}")
    return s


def num_gold(value, units=()) -> dict:
    return {"kind": "number", "value": fmt_num(value), "units": sorted({u.casefold() for u in units})}


def msgs(system: str, user: str) -> list:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def one(sub, user, gold, op, surf) -> dict:
    return {"subtype": sub, "op": op, "prompts": [{"twin": None, "messages": msgs(surf.pick(SYSTEM), user), "gold": gold}]}


# ------------------------------------------------------------------------------------------------ 1 arith_word
def gen_arith_word(rng, level, surf, stratum):
    """Multi-step integer word problem: L1 2 ops (+/-), L2 3 ops (+/-, one x2/x3), L3 4 ops (+/-, one xk, one /k)."""
    plural, single = rng.choice(OBJS)
    a, b = two_people(rng)
    ops_n = {1: 2, 2: 3, 3: 4}[level]
    for _ in range(500):
        n0 = rng.randint(5, {1: 30, 2: 40, 3: 60}[level])
        kinds = ["add", "sub"]
        seq = [rng.choice(kinds) for _ in range(ops_n)]
        if level == 2:
            seq[rng.randrange(1, ops_n)] = "mul"
        if level == 3:
            i, j = rng.sample(range(1, ops_n), 2)
            seq[i], seq[j] = "mul", "div"
        cur, lines, ok, trace = n0, [f"{a} has {n0} {plural}."], True, [n0]
        for op in seq:
            if op == "add":
                k = rng.randint(2, {1: 15, 2: 25, 3: 40}[level])
                cur += k
                lines.append(rng.choice([f"{a} buys {k} more {plural}.", f"{b} gives {a} {k} more {plural}.",
                                         f"{a} finds {k} more {plural}."]))
            elif op == "sub":
                if cur <= 3:
                    ok = False
                    break
                k = rng.randint(1, min(cur - 1, {1: 15, 2: 25, 3: 40}[level]))
                cur -= k
                lines.append(rng.choice([f"{a} gives {k} {plural} to {b}.", f"{a} uses {k} of the {plural}.",
                                         f"{a} sells {k} {plural}."]))
            elif op == "mul":
                k = rng.choice((2, 3)) if level == 2 else rng.choice((2, 3, 4))
                cur *= k
                lines.append({2: f"Then {a} doubles the number of {plural}.",
                              3: f"Then {a} triples the number of {plural}.",
                              4: f"Then the number of {plural} {a} has becomes four times as large."}[k])
            else:
                divs = [k for k in (2, 3, 4, 5) if cur % k == 0 and cur // k >= 2]
                if not divs:
                    ok = False
                    break
                k = rng.choice(divs)
                cur //= k
                lines.append(f"{a} splits the {plural} into {k} equal groups and keeps only one group.")
            trace.append(f"{op}{k}")
        if ok and 1 <= cur <= 999:
            user = " ".join(lines) + f" How many {plural} does {a} have now? " + surf.pick(I_NUM)
            return one(f"{ops_n}ops", user, num_gold(cur, (plural, single)), f"{level}|{trace}", surf)
    raise RuntimeError("arith_word: no valid problem")


# ------------------------------------------------------------------------------------------------ 2 unit_conv
UNITS = {
    "m": ("meters", "meter", ("m", "meter", "meters", "metre", "metres")),
    "cm": ("centimeters", "centimeter", ("cm", "centimeter", "centimeters", "centimetre", "centimetres")),
    "mm": ("millimeters", "millimeter", ("mm", "millimeter", "millimeters", "millimetre", "millimetres")),
    "km": ("kilometers", "kilometer", ("km", "kilometer", "kilometers", "kilometre", "kilometres")),
    "kg": ("kilograms", "kilogram", ("kg", "kilogram", "kilograms")),
    "g": ("grams", "gram", ("g", "gram", "grams")),
    "l": ("liters", "liter", ("l", "liter", "liters", "litre", "litres")),
    "ml": ("milliliters", "milliliter", ("ml", "milliliter", "milliliters", "millilitre", "millilitres")),
    "h": ("hours", "hour", ("h", "hr", "hrs", "hour", "hours")),
    "min": ("minutes", "minute", ("min", "mins", "minute", "minutes")),
    "s": ("seconds", "second", ("s", "sec", "secs", "second", "seconds")),
    "day": ("days", "day", ("day", "days")),
    "week": ("weeks", "week", ("week", "weeks")),
    "ft": ("feet", "foot", ("ft", "foot", "feet")),
    "in": ("inches", "inch", ("in", "inch", "inches")),
    "yd": ("yards", "yard", ("yd", "yard", "yards")),
    "lb": ("pounds", "pound", ("lb", "lbs", "pound", "pounds")),
    "oz": ("ounces", "ounce", ("oz", "ounce", "ounces")),
}
CONV = (("m", "cm", 100), ("km", "m", 1000), ("kg", "g", 1000), ("l", "ml", 1000), ("h", "min", 60), ("min", "s", 60),
        ("day", "h", 24), ("week", "day", 7), ("ft", "in", 12), ("yd", "ft", 3), ("lb", "oz", 16), ("cm", "mm", 10))
CHAINS = (("h", "min", "s"), ("yd", "ft", "in"), ("week", "day", "h"), ("m", "cm", "mm"), ("km", "m", "cm"),
          ("day", "h", "min"))
FACTOR = {(a, b): f for a, b, f in CONV}
# The quantity lives in a sentence with the person's name (item-unique, >= 8 words: freshness gate F2); the question
# sentence is a shared template. Units: (sentence with {who} {x} {src}, question with {dst}).
FRAME_U = {
    "length": ("{who} measured a garden path that is {x} {src} long.", "How long is the path in {dst}?"),
    "mass": ("{who} weighed a sack of flour and it came to {x} {src}.", "How heavy is the sack in {dst}?"),
    "volume": ("{who} poured {x} {src} of water into a rain barrel.", "How much water is that in {dst}?"),
    "time": ("{who} spent {x} {src} working on a community project.", "How long is that in {dst}?"),
    "imperial": ("{who} cut a strip of curtain fabric that is {x} {src} long.", "How long is the strip in {dst}?"),
    "weight": ("{who} packed a parcel for the post office weighing {x} {src}.", "How heavy is the parcel in {dst}?"),
}
UTYPE = {"m": "length", "cm": "length", "mm": "length", "km": "length", "kg": "mass", "g": "mass", "l": "volume",
         "ml": "volume", "h": "time", "min": "time", "s": "time", "day": "time", "week": "time", "ft": "imperial",
         "in": "imperial", "yd": "imperial", "lb": "weight", "oz": "weight"}


def gen_unit_conv(rng, level, surf, stratum):
    from fractions import Fraction
    a = full_name(rng)
    for _ in range(500):
        if level == 1:
            src, dst, f = rng.choice(CONV)
            x = Fraction(rng.randint(2, 15 if f >= 100 else 40))
            y, sub = x * f, "forward"
        elif level == 2:
            src, dst, f = rng.choice(CONV)
            if stratum == "reverse":                     # reverse: "how many meters are in 4500 centimeters?"
                y = Fraction(rng.randint(2, 60))
                x, src, dst, sub = y * f, dst, src, "reverse"
            else:                                        # decimal source with an integer result
                if f % 2:
                    continue
                x = Fraction(rng.randint(1, 9)) + Fraction(1, 2)
                y, sub = x * f, "decimal"
        else:
            ch = rng.choice(CHAINS)
            f = FACTOR[(ch[0], ch[1])] * FACTOR[(ch[1], ch[2])]
            if stratum == "chain":
                x = Fraction(rng.randint(2, 9)) + (Fraction(1, 2) if rng.random() < 0.5 else 0)
                src, dst, y, sub = ch[0], ch[2], x * f, "chain"
            else:
                y = Fraction(rng.randint(2, 12))
                src, dst, x, sub = ch[2], ch[0], y * f, "chain_reverse"
        if y.denominator != 1 or x <= 1 or y <= 1:
            continue
        sp, dp = UNITS[src][0], UNITS[dst][0]
        fa, fq = FRAME_U[UTYPE[src]]
        user = fa.format(who=a, x=fmt_num(x), src=sp) + " " + fq.format(dst=dp) + " " + surf.pick(I_NUM)
        return one(sub, user, num_gold(y, UNITS[dst][2]), f"{level}|{src}>{dst}|{fmt_num(x)}", surf)
    raise RuntimeError("unit_conv")


# ------------------------------------------------------------------------------------------------ 3 date_time
DAYS = [d.capitalize() for d in RG.WEEKDAYS]
MONTH_DAYS_2027 = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _dur_text(mins: int) -> str:
    h, m = divmod(mins, 60)
    if h and m:
        return f"{h} hour{'s' if h > 1 else ''} and {m} minutes"
    if h:
        return f"{h} hour{'s' if h > 1 else ''}"
    return f"{m} minutes"


def gen_date_time(rng, level, surf, stratum):
    who = full_name(rng)
    ev = rng.choice(EVENTS)
    sub = stratum
    for _ in range(500):
        if sub == "clock_add":
            start = rng.randrange(6 * 60, 20 * 60, 5)
            d = rng.randrange(5, 56, 5) if level == 1 else rng.randrange(65, 5 * 60 + 56, 5)
            end = start + d
            if end >= 24 * 60 or (level == 2 and not (start < 12 * 60 <= end)):
                continue
            user = f"{who}'s {ev} starts at {hhmm(start)} and lasts {_dur_text(d)}. At what time does it end? " + surf.pick(I_CLOCK)
            return one(sub, user, {"kind": "clock", "minute": end}, f"{level}|{sub}|{start}|{d}", surf)
        if sub == "clock_midnight":
            start = rng.randrange(19 * 60, 23 * 60 + 56, 5)
            d = rng.randrange(65, 8 * 60 + 56, 5)
            end = start + d
            if end < 24 * 60 or end >= 48 * 60:
                continue
            user = (f"{who}'s night shift starts at {hhmm(start)} and lasts {_dur_text(d)}. At what time does it end? "
                    + surf.pick(I_CLOCK))
            return one(sub, user, {"kind": "clock", "minute": end - 24 * 60}, f"{level}|{sub}|{start}|{d}", surf)
        if sub == "duration_between":
            start = rng.randrange(7 * 60, 16 * 60, 5)
            d = rng.randrange(60, 7 * 60, 5)
            if start + d >= 23 * 60:
                continue
            user = (f"{who}'s {ev} runs from {hhmm(start)} to {hhmm(start + d)} on the same day. How many minutes long "
                    "is it? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(d, ("minute", "minutes", "min", "mins")), f"{level}|{sub}|{start}|{d}", surf)
        if sub in ("weekday_add", "weekday_back"):
            w0 = rng.randrange(7)
            if sub == "weekday_add":
                n = rng.choice([x for x in range(1, 14) if x != 7]) if level == 1 else rng.choice(
                    [x for x in range(15, 61) if x % 7])
                ans, rel = (w0 + n) % 7, f"{n} days later"
            else:
                n = rng.choice([x for x in range(8, 41) if x % 7])
                ans, rel = (w0 - n) % 7, f"{n} days earlier"
            user = (f"{who}'s {ev} is on a {DAYS[w0]}. What day of the week is it {rel}? " + surf.pick(I_WEEKDAY))
            return one(sub, user, {"kind": "weekday", "value": RG.WEEKDAYS[ans]}, f"{level}|{sub}|{w0}|{n}", surf)
        if sub == "date_add":
            mon = rng.randrange(12)
            day = rng.randint(1, MONTH_DAYS_2027[mon])
            n = rng.randint(5, 45)
            doy = sum(MONTH_DAYS_2027[:mon]) + day - 1 + n
            if doy >= 365:
                continue
            m2 = 0
            while doy >= MONTH_DAYS_2027[m2]:
                doy -= MONTH_DAYS_2027[m2]
                m2 += 1
            if m2 == mon:
                continue                                  # L3 dates always cross a month boundary
            mname = RG.MONTHS[mon].capitalize()
            user = (f"In 2027, {who}'s {ev} is on {mname} {day}. What date is {n} days later? " + surf.pick(I_DATE))
            return one(sub, user, {"kind": "date", "month": m2 + 1, "day": doy + 1, "year": 2027},
                       f"{level}|{sub}|{mon}|{day}|{n}", surf)
    raise RuntimeError("date_time")


# ------------------------------------------------------------------------------------------------ 4 count_sort
def gen_count_sort(rng, level, surf, stratum):
    n = {1: 5, 2: 7, 3: 9}[level]
    sub = stratum
    who = rng.choice(FIRST)
    lo = -50 if level == 3 else 1
    if sub == "count_word":
        pool = rng.sample(WORDS, 3 if level < 3 else 4)
        target = pool[0]
        c = rng.randint(1, n - 2)
        items = [target] * c + [rng.choice(pool[1:]) for _ in range(n - c)]
        rng.shuffle(items)
        user = (f"{who} wrote this list of words: {', '.join(items)}. How many times does the word \"{target}\" appear "
                "in the list? " + surf.pick(I_NUM))
        return one(sub, user, num_gold(c, ("time", "times")), f"{level}|{sub}|{items}|{target}", surf)
    if sub == "count_gt":
        items = [rng.randint(lo, 99) for _ in range(n)]
        t = rng.randint(lo + 10, 80)
        c = sum(x > t for x in items)
        user = (f"{who} wrote these numbers on a whiteboard: {', '.join(map(str, items))}. How many of them are greater "
                f"than {t}? " + surf.pick(I_NUM))
        return one(sub, user, num_gold(c, ("number", "numbers")), f"{level}|{sub}|{items}|{t}", surf)
    if sub in ("sort_num_asc", "sort_num_desc"):
        items = rng.sample(range(lo, 100), n)
        if level == 3:
            items[-1] = items[0]                           # one duplicate value at level 3
            rng.shuffle(items)
        desc = sub == "sort_num_desc"
        order = "from largest to smallest" if desc else "from smallest to largest"
        user = (f"{who} wrote these numbers on a whiteboard: {', '.join(map(str, items))}. Sort them {order}. "
                + surf.pick(I_SORT_NUM))
        return one(sub, user, {"kind": "num_list", "value": sorted(items, reverse=desc)}, f"{level}|{sub}|{sorted(items)}", surf)
    # sort_words
    if level == 3:
        by_letter = defaultdict(list)
        for w in WORDS:
            by_letter[w[0]].append(w)
        letters = [k for k, v in by_letter.items() if len(v) >= 2]
        for _ in range(100):
            ls = rng.sample(letters, 4)
            items = [w for l in ls for w in rng.sample(by_letter[l], 2)][: n - 1] + [rng.choice(WORDS)]
            if len(set(items)) == n:
                break
    else:
        items = rng.sample(WORDS, n)
    rng.shuffle(items)
    user = (f"{who} wrote this list of words: {', '.join(items)}. Sort the words in alphabetical order. "
            + surf.pick(I_SORT_WORD))
    return one(sub, user, {"kind": "word_list", "value": sorted(items), "vocab": sorted(set(items))},
               f"{level}|{sub}|{sorted(items)}", surf)


# ------------------------------------------------------------------------------------------------ 5 pct_ratio
COIN = ("coin", "coins")
PCT_UNITS = ("%", "percent", "per cent")


def gen_pct_ratio(rng, level, surf, stratum):
    from fractions import Fraction
    a, b = two_full_names(rng)
    sub = stratum
    for _ in range(500):
        if sub in ("pct_of", "pct_of_hard"):
            p = Fraction(rng.choice((10, 20, 25, 50))) if sub == "pct_of" else Fraction(rng.choice(
                ("15", "35", "45", "12.5", "37.5", "62.5")))
            nmax = 400 if sub == "pct_of" else 800
            cands = [N for N in range(20, nmax + 1, 4) if (N * p / 100).denominator == 1]
            N = rng.choice(cands)
            user = (f"A receipt that {a} is checking shows a total of {N} coins. What is {fmt_num(p)}% of that total, "
                    "in coins? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(N * p / 100, COIN), f"{level}|{sub}|{p}|{N}", surf)
        if sub == "ratio_scale":
            plural, single = rng.choice(OBJS)
            u, k1 = rng.randint(2, 12), rng.randint(2, 6)
            k2 = rng.choice([k for k in range(3, 16) if k != k1])
            user = (f"At {a}'s stall, {k1} {plural} cost {k1 * u} coins. At the same price for each, how many coins do "
                    f"{k2} {plural} cost? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(k2 * u, COIN), f"{level}|{sub}|{u}|{k1}|{k2}", surf)
        if sub == "pct_change":
            P = rng.randrange(40, 401, 20)
            p = rng.choice((5, 10, 15, 20, 25, 30, 40, 50))
            up = rng.random() < 0.5
            y = Fraction(P) * (100 + (p if up else -p)) / 100
            if y.denominator != 1:
                continue
            user = (f"A {rng.choice(THINGS)} at {a}'s shop costs {P} coins. The price goes {'up' if up else 'down'} by "
                    f"{p}%. What is the new price in coins? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(y, COIN), f"{level}|{sub}|{P}|{p}|{up}", surf)
        if sub == "what_pct":
            tot = rng.choice((20, 25, 40, 50, 80))
            k = rng.randint(1, tot - 1)
            y = Fraction(100 * k, tot)
            if y.denominator != 1:
                continue
            user = (f"{a} answered {k} of {tot} quiz questions correctly. What percentage of the questions did {a} answer "
                    "correctly? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(y, PCT_UNITS), f"{level}|{sub}|{k}|{tot}", surf)
        if sub == "ratio_simplify":
            g = rng.randint(2, 9)
            x, y = rng.randint(1, 9), rng.randint(1, 9)
            if x == y or math.gcd(x, y) != 1:
                continue
            user = (f"{a} mixes paint in the ratio {g * x}:{g * y}. Write this ratio in simplest form. " + surf.pick(I_RATIO))
            return one(sub, user, {"kind": "ratio", "a": x, "b": y}, f"{level}|{sub}|{g}|{x}|{y}", surf)
        if sub == "successive":
            P = rng.randrange(100, 801, 100)
            p1, p2 = rng.choice((10, 20, 25, 50)), rng.choice((10, 20, 25, 50))
            y = Fraction(P) * (100 + p1) / 100 * (100 - p2) / 100
            if y.denominator != 1:
                continue
            user = (f"A {rng.choice(THINGS)} at {a}'s shop costs {P} coins. Its price rises by {p1}%, and then the new "
                    f"price falls by {p2}%. What is the final price in coins? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(y, COIN), f"{level}|{sub}|{P}|{p1}|{p2}", surf)
        if sub == "ratio_split":
            x, y = rng.randint(1, 7), rng.randint(1, 7)
            if x == y or math.gcd(x, y) != 1:
                continue
            u = rng.randint(3, 30)
            T = (x + y) * u
            user = (f"{a} and {b} share {T} coins in the ratio {x}:{y}. How many coins does {a} get? " + surf.pick(I_NUM))
            return one(sub, user, num_gold(x * u, COIN), f"{level}|{sub}|{x}|{y}|{u}", surf)
    raise RuntimeError("pct_ratio")


# ------------------------------------------------------------------------------------------------ 6/7 passages
def _code4(rng):
    return str(rng.randint(1000, 9999))


def _code3(rng):
    return str(rng.randint(100, 999))


# 'fn': field-name variants (index 0 = default; surface diversity draws one per passage and uses it in the records,
# the corrections and the question alike). Every question 'q' asks for the CURRENT value in the present tense (the
# corrections are past events), so a corrected record has one reading: its latest value (ret_test_graders checks the
# tense of every template and re-derives every gold from the prompt text).
THEMES = {
    "lockers": {"intro": "These are the locker assignments for the {place} swimming club.",
                "fn": ("door code", "entry code", "keypad code"),
                "rec": "{who} has locker {n}, and its {fn} is {v}.",
                "q": ("What is the {fn} of {who}'s locker?", "What {fn} opens {who}'s locker?"),
                "mode": "code", "val": _code4,
                "corr": "Later, the {fn} of locker {n}, used by {who}, was changed to {v}.",
                "alt": ("phone extension", "parking bay number", "membership level")},
    "vans": {"intro": "The {place} delivery team assigned vans and depots for the week.",
             "fn": ("depot", "base", "yard"),
             "rec": "{who} drives van {n} and starts every route at the {v} {fn}.",
             "q": ("At which {fn} does {who} start every route?", "Which {fn} does {who} start every route from?"),
             "mode": "word", "pool": CITIES,
             "corr": "Later, {who} and van {n} were moved to the {v} {fn}.",
             "alt": ("radio channel", "fuel card number", "uniform size")},
    "garden": {"intro": "The {place} community garden lists its plot holders.",
               "fn": ("plot", "bed", "allotment"),
               "rec": "{who} rents {fn} {n} and is growing {v} this season.",
               "q": ("What is {who} growing this season?", "Which crop is {who} growing this season?"),
               "mode": "word", "pool": CROPS,
               "corr": "Later, {who} switched {fn} {n} to growing {v}.",
               "alt": ("gate key number", "water allowance", "compost bin number")},
    "bikes": {"intro": "The {place} cycling club recorded its loan bikes.",
              "fn": ("bike", "cycle", "loan bike"),
              "rec": "{who} borrowed {fn} {n}, which is painted {v}.",
              "q": ("What colour is the {fn} that {who} borrowed?", "Which colour is the {fn} {who} borrowed?"),
              "mode": "word", "pool": COLOURS,
              "corr": "Later, {fn} {n}, borrowed by {who}, was repainted {v}.",
              "alt": ("helmet size", "membership number", "emergency contact")},
    "lab": {"intro": "The {place} lab logbook lists where sample trays are stored.",
            "fn": ("freezer", "cold store", "storage freezer"),
            "rec": "{who} logged sample tray {n} in {fn} {v}.",
            "q": ("In which {fn} is {who}'s sample tray stored?", "Which {fn} holds the sample tray {who} logged?"),
            "mode": "code", "val": _code3,
            "corr": "Later, sample tray {n} from {who} was moved to {fn} {v}.",
            "alt": ("bench number", "safety badge number", "project code")},
    "music": {"intro": "The {place} music school booked its practice rooms.",
              "fn": ("room", "studio", "practice room"),
              "rec": "{who} booked {fn} {n} for {v} practice.",
              "q": ("Which instrument is {who}'s {fn} booked for?", "For which instrument is {who}'s {fn} currently booked?"),
              "mode": "word", "pool": INSTRUMENTS,
              "corr": "Later, {who} changed the {fn} {n} booking to {v} practice.",
              "alt": ("locker number", "lesson fee", "teacher's name")},
}
THEME_ORDER = tuple(THEMES)
N_CORR = {1: 0, 2: 0, 3: 2}
# Passage length control without a tokenizer: per (theme, field name) bounds on characters per Qwen3 token, measured on
# 250 generated passages each with the Titan tokenizer and widened by 0.05, so chars / max_ratio is a lower and
# chars / min_ratio an upper bound on the token count. Records are added until the lower bound reaches the level's
# band; a passage whose upper bound leaves the band is redrawn. mint() re-checks every passage with the real tokenizer
# (gate: 150 <= tokens <= 300).
THEME_RATIO = {
    ("lockers", "door code"): (2.66, 3.11), ("lockers", "entry code"): (2.69, 3.16), ("lockers", "keypad code"): (2.74, 3.19),
    ("vans", "depot"): (3.65, 4.23), ("vans", "base"): (3.59, 4.16), ("vans", "yard"): (3.56, 4.13),
    ("garden", "plot"): (3.65, 4.19), ("garden", "bed"): (3.61, 4.13), ("garden", "allotment"): (3.71, 4.24),
    ("bikes", "bike"): (3.52, 4.13), ("bikes", "cycle"): (3.60, 4.19), ("bikes", "loan bike"): (3.61, 4.14),
    ("lab", "freezer"): (3.01, 3.47), ("lab", "cold store"): (2.96, 3.46), ("lab", "storage freezer"): (3.23, 3.74),
    ("music", "room"): (3.49, 4.18), ("music", "studio"): (3.59, 4.32), ("music", "practice room"): (3.76, 4.48),
}
PASSAGE_BAND = {1: (152, 230), 2: (190, 265), 3: (230, 298)}
CLOSINGS = ("The list was last checked on {day}.", "All entries were confirmed on {day}.",
            "The office updated this list on {day}.")


def _assemble(rng, th, level, nrec, shared_first, fn):
    used = set()
    if shared_first:                                   # some first names repeat, so the full name is needed
        firsts = rng.sample(FIRST, nrec - 3)
        firsts += rng.sample(firsts, 3)
        rng.shuffle(firsts)
    else:
        firsts = rng.sample(FIRST, nrec)
    names = [full_name(rng, used, first_pool=(f,)) for f in firsts]
    keys = rng.sample(range(2, 99), nrec)
    vals_seen = set()

    def new_val():
        for _ in range(1000):
            v = th["val"](rng) if th["mode"] == "code" else rng.choice(th["pool"])
            if v not in vals_seen:
                vals_seen.add(v)
                return v
        raise RuntimeError("value pool exhausted")

    recs = [{"who": w, "n": k, "v": new_val(), "old": None} for w, k in zip(names, keys)]
    sents = [th["intro"].format(place=rng.choice(PLACES))]
    sents += [th["rec"].format(who=r["who"], n=r["n"], v=r["v"], fn=fn) for r in recs]
    for r in rng.sample(recs, N_CORR[level]):
        r["old"], r["v"] = r["v"], new_val()
        sents.append(th["corr"].format(who=r["who"], v=r["v"], n=r["n"], fn=fn))
    sents.append(rng.choice(CLOSINGS).format(day=rng.choice(DAYS)))
    return " ".join(sents), recs, sorted(vals_seen)


def build_passage(rng, level, surf, shared_first: bool, theme_name: str):
    """One theme's records: (passage text, records [{who, n, v, old}], theme, every value incl. superseded ones,
    field name)."""
    th = THEMES[theme_name]
    fn = surf.pick(th["fn"])
    lo, hi = PASSAGE_BAND[level]
    rmin, rmax = THEME_RATIO[(theme_name, fn)]
    for _ in range(200):
        sub_seed = rng.getrandbits(64)
        for nrec in range(6, 30):
            text, recs, vals = _assemble(random.Random(sub_seed), th, level, nrec, shared_first, fn)
            if len(text) / rmax >= lo:
                if len(text) / rmin <= hi:
                    return text, recs, theme_name, vals, fn
                break
    raise RuntimeError(f"no passage fits the band for {theme_name} L{level}")


def _kv_gold(th, rec, values):
    return {"kind": "kv", "mode": th["mode"], "value": rec["v"], "distractors": [v for v in values if v != rec["v"]]}


def gen_kv_recall(rng, level, surf, stratum):
    """Key-value recall from a 150-300-token passage (records are added until the level's token band is reached:
    L1 ~150-230, L2 ~190-265, L3 ~230-298 tokens). L2: some first names repeat, so the full name is needed. L3: also
    two later corrections; the stratum '<theme>_corrected' asks about a corrected record (gold: the latest value; the
    superseded value is a distractor), the stratum '<theme>' about an uncorrected one."""
    tname = stratum[: -len("_corrected")] if stratum.endswith("_corrected") else stratum
    passage, recs, tname, values, fn = build_passage(rng, level, surf, level >= 2, tname)
    th = THEMES[tname]
    corrected = [r for r in recs if r["old"] is not None]
    plain = [r for r in recs if r["old"] is None]
    rec = rng.choice(corrected if stratum.endswith("_corrected") else plain)
    q = surf.pick(th["q"]).format(who=rec["who"], fn=fn)
    user = f"{passage}\n\n{q} {surf.pick(I_KV)}"
    sub = f"{tname}{'_corrected' if rec['old'] is not None else ''}"
    return one(sub, user, _kv_gold(th, rec, values), f"{level}|{tname}|{rec['who']}|{rec['v']}", surf)


def gen_unknown_twin(rng, level, surf, stratum):
    """A PAIR over one passage of theme `stratum`: 'a' asks a value the passage states (gold: the value), 'u' asks one
    it does not (gold: the literal UNKNOWN). L1 'u' asks about a person absent from the passage; L2 asks the same
    person's attribute the passage never mentions; L3 asks about a near-miss name (one record's first name with another
    record's last name; that combination is absent)."""
    passage, recs, tname, values, fn = build_passage(rng, level, surf, level == 3, stratum)
    th = THEMES[tname]
    rec = rng.choice(recs)
    instr = surf.pick(I_UNK)
    qt = surf.pick(th["q"])                              # one question template for both twins
    qa = qt.format(who=rec["who"], fn=fn)
    names = {r["who"] for r in recs}
    if level == 1:
        firsts = {r["who"].split()[0] for r in recs}
        lasts = {r["who"].split()[1] for r in recs}
        ghost = full_name(rng, set(), first_pool=tuple(f for f in FIRST if f not in firsts),
                          last_pool=tuple(l for l in LAST if l not in lasts))
        qu = qt.format(who=ghost, fn=fn)
        sub = "absent_person"
    elif level == 2:
        alt = rng.choice(th["alt"])
        qu = f"What is {rec['who']}'s {alt}?"
        sub = "absent_attribute"
    else:
        for _ in range(1000):
            r1, r2 = rng.sample(recs, 2)
            ghost = f"{r1['who'].split()[0]} {r2['who'].split()[1]}"
            if ghost not in names:
                break
        qu = qt.format(who=ghost, fn=fn)
        sub = "near_miss_name"
    sysm = surf.pick(SYSTEM)
    pa = {"twin": "a", "messages": msgs(sysm, f"{passage}\n\n{instr}\n\n{qa}"), "gold": _kv_gold(th, rec, values)}
    pu = {"twin": "u", "messages": msgs(sysm, f"{passage}\n\n{instr}\n\n{qu}"),
          "gold": {"kind": "unknown", "passage_values": values}}
    return {"subtype": sub, "op": f"{level}|{tname}|{rec['who']}|{qu}", "prompts": [pa, pu]}


# ------------------------------------------------------------------------------------------------ 8 format_follow
def gen_format_follow(rng, level, surf, stratum):
    who = full_name(rng)
    sub = stratum
    w = rng.choice(WORDS)
    ctx = rng.choice((f"{who} is labelling storage bins.", f"{who} is testing a new label printer.",
                      f"{who} is preparing cards for a word game.", f"{who} is checking a printed list."))
    if sub == "upper_echo":
        user = f"{ctx} Reply with only the word \"{w}\" written in capital letters."
        return one(sub, user, {"kind": "exact", "value": w.upper(), "case": True}, f"{level}|{sub}|{w}", surf)
    if sub == "yes_no":
        plural, _ = rng.choice(OBJS)
        x, y = rng.sample(range(3, 99), 2)
        other = full_name(rng, first_pool=tuple(f for f in FIRST if f != who.split()[0]))
        user = (f"{who} has {x} {plural} and {other} has {y} {plural}. Does {who} have more {plural} than {other}? "
                + surf.pick(I_YESNO))
        return one(sub, user, {"kind": "exact", "value": "yes" if x > y else "no", "case": False},
                   f"{level}|{sub}|{x}|{y}", surf)
    if sub in ("nth_word", "last_word"):
        sent = rng.sample(WORDS, rng.randint(7, 9))
        if sub == "nth_word":
            k = rng.randint(2, 6)
            user = f"{ctx} Here is a sequence of words: \"{' '.join(sent)}\". Reply with only the {ORD_WORDS[k]} word of that sequence."
            ans = sent[k - 1]
        else:
            user = f"{ctx} Here is a sequence of words: \"{' '.join(sent)}\". Reply with only the last word of that sequence."
            ans = sent[-1]
        return one(sub, user, {"kind": "exact", "value": ans, "case": False}, f"{level}|{sub}|{sent}|{ans}", surf)
    if sub == "first_letter":
        user = f"{ctx} Reply with only the first letter of the word \"{w}\", as a single lowercase letter."
        return one(sub, user, {"kind": "exact", "value": w[0], "case": True}, f"{level}|{sub}|{w}", surf)
    if sub == "reverse":
        while w == w[::-1]:
            w = rng.choice(WORDS)
        user = f"{ctx} Write the word \"{w}\" backwards in lowercase letters. Reply with only that reversed word."
        return one(sub, user, {"kind": "exact", "value": w[::-1], "case": True}, f"{level}|{sub}|{w}", surf)
    if sub == "json":
        n = rng.randint(2, 99)
        user = (f"{ctx} Reply with only a JSON object that has the key \"item\" set to the string \"{w}\" and the key "
                f"\"count\" set to the number {n}.")
        return one(sub, user, {"kind": "json", "value": {"item": w, "count": n}}, f"{level}|{sub}|{w}|{n}", surf)
    user = (f"{ctx} Spell the word \"{w}\" in capital letters with a hyphen between the letters, as in C-A-T. "
            "Reply with only the spelled word.")
    return one(sub, user, {"kind": "exact", "value": "-".join(w.upper()), "case": True}, f"{level}|{sub}|{w}", surf)


# Strata per (category, level): each cell's items are dealt round-robin over these and shuffled (same mix every draw).
STRATA = {
    "arith_word": {1: ("2ops",), 2: ("3ops",), 3: ("4ops",)},
    "unit_conv": {1: ("forward",), 2: ("reverse", "decimal"), 3: ("chain", "chain_reverse")},
    "date_time": {1: ("clock_add", "weekday_add"), 2: ("clock_add", "weekday_add", "duration_between"),
                  3: ("date_add", "clock_midnight", "weekday_back")},
    "count_sort": {1: ("count_word", "count_gt", "sort_num_asc"), 2: ("count_word", "count_gt", "sort_num_desc", "sort_words"),
                   3: ("count_word", "count_gt", "sort_num_asc", "sort_words")},
    "pct_ratio": {1: ("pct_of", "ratio_scale"), 2: ("pct_change", "what_pct", "ratio_simplify"),
                  3: ("pct_of_hard", "successive", "ratio_split")},
    "kv_recall": {1: THEME_ORDER, 2: THEME_ORDER, 3: tuple(f"{t}{c}" for t in THEME_ORDER for c in ("", "_corrected"))},
    "unknown_twin": {1: THEME_ORDER, 2: THEME_ORDER, 3: THEME_ORDER},
    "format_follow": {1: ("upper_echo", "yes_no"), 2: ("nth_word", "first_letter", "last_word"),
                      3: ("reverse", "json", "hyphen_spell")},
}


def stratum_schedule(rng, cat: str, level: int, n: int) -> list:
    st = STRATA[cat][level]
    sched = [st[i % len(st)] for i in range(n)]
    rng.shuffle(sched)
    return sched


GENERATORS = {"arith_word": gen_arith_word, "unit_conv": gen_unit_conv, "date_time": gen_date_time,
              "count_sort": gen_count_sort, "pct_ratio": gen_pct_ratio, "kv_recall": gen_kv_recall,
              "unknown_twin": gen_unknown_twin, "format_follow": gen_format_follow}
assert tuple(GENERATORS) == RC.CATEGORIES


# ------------------------------------------------------------------------------------------------ draws
def draw_id(purpose: str, seed: int) -> str:
    return f"{purpose}_{seed}"


def gen_cell(seed: int, cat: str, level: int, n: int, diversity: bool) -> tuple[list, dict]:
    """75 items of one (category, level) cell. Within-cell: user turns unique (always); operative contents unique
    where 50 redraws allow it (remaining duplicates are counted)."""
    rng = random.Random(f"{seed}|{cat}|L{level}|sd{int(bool(diversity))}")
    surf = Surf(rng, diversity)
    sched = stratum_schedule(rng, cat, level, n)
    items, users, ops = [], set(), Counter()
    op_dups = 0
    while len(items) < n:
        it = None
        stratum = sched[len(items)]
        for attempt in range(50):
            cand = GENERATORS[cat](rng, level, surf, stratum)
            cand["stratum"] = stratum
            us = tuple(RC.norm_text(RC.user_text(p["messages"])) for p in cand["prompts"])
            if any(u in users for u in us):
                continue
            it = cand
            if cand["op"] not in ops:
                break
        if it is None:
            raise RuntimeError(f"{cat} L{level}: could not draw a unique item")
        if it["op"] in ops:
            op_dups += 1
        ops[it["op"]] += 1
        for p in it["prompts"]:
            users.add(RC.norm_text(RC.user_text(p["messages"])))
        items.append(it)
    return items, {"operative_duplicates_within_cell": op_dups}


def generate(seed: int, purpose: str, plan: dict, diversity: bool = False, n: int = RC.N_PER_CELL) -> tuple[list, dict]:
    """plan {category: [levels]} -> (rows, notes). One row per prompt, fixed key order."""
    did = draw_id(purpose, seed)
    rows, notes = [], {}
    for cat in RC.CATEGORIES:
        for level in sorted(plan.get(cat, [])):
            items, nt = gen_cell(seed, cat, level, n, diversity)
            notes[f"{cat}|L{level}"] = nt
            for i, it in enumerate(items):
                iid = f"{did}/{cat}/L{level}/{i:03d}"
                for p in it["prompts"]:
                    RG.validate_gold(p["gold"])
                    rows.append({"pid": iid + (f"/{p['twin']}" if p["twin"] else ""), "item": iid, "category": cat,
                                 "level": level, "subtype": it["subtype"], "stratum": it["stratum"], "twin": p["twin"],
                                 "messages": p["messages"], "gold": p["gold"], "op": it["op"]})
    return rows, notes


def serialize(rows: list) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def dev_plan() -> dict:
    return {c: list(RC.LEVELS) for c in RC.CATEGORIES}


def selection_plan(sel: dict) -> dict:
    return {c: [int(l)] for c, l in sorted((sel.get("selected") or {}).items())}


# ------------------------------------------------------------------------------------------------ self-test of golds
def gold_selftest(rows: list) -> dict:
    """Every canonical answer grades correct (primary); negatives grade wrong: empty, not terminated, a different
    item's canonical answer of the same kind with a different target, UNKNOWN on answerable prompts, the answerable
    value on unanswerable prompts, and the canonical answer followed by an explanation."""
    pos = neg = pos_ok = neg_ok = 0
    fails = []
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["gold"]["kind"]].append(r)
    for r in rows:
        g = r["gold"]
        ans = RG.canonical_answer(g)
        pos += 1
        if RG.grade(g, ans)["correct"] and RG.grade_lenient(g, ans):
            pos_ok += 1
        else:
            fails.append(("pos", r["pid"], ans))
        negs = ["", ans + " because it follows from the question", ans + " or maybe not"]
        others = [o for o in by_kind[g["kind"]] if RG.target_of(o["gold"]) != RG.target_of(g)
                  and RG.canonical_answer(o["gold"]).casefold() != ans.casefold()]
        if others:
            negs.append(RG.canonical_answer(others[int(RC.sha256_bytes(r["pid"].encode())[:8], 16) % len(others)]["gold"]))
        if g["kind"] != "unknown":
            negs.append("UNKNOWN")
        else:
            negs += [str(v) for v in g.get("passage_values", [])[:2]] + ["unknown", "Unknown."]
        for x in negs:
            neg += 1
            if not RG.grade(g, x)["correct"]:
                neg_ok += 1
            else:
                fails.append(("neg", r["pid"], x))
        neg += 1
        if not RG.grade(g, ans, terminated=False)["correct"]:
            neg_ok += 1
        else:
            fails.append(("neg_unterminated", r["pid"], ans))
    return {"positives": pos, "positives_correct": pos_ok, "negatives": neg, "negatives_rejected": neg_ok,
            "ok": pos_ok == pos and neg_ok == neg, "failures": fails[:20]}


# ------------------------------------------------------------------------------------------------ freshness
def item_unique_sentences(rows: list, min_words: int) -> dict:
    """{sentence: [pids]} for sentences (of normalised user turns) with >= min_words words occurring in ONE item."""
    owners = defaultdict(set)
    where = defaultdict(list)
    for r in rows:
        for s in set(RC.split_sentences(RC.norm_text(RC.user_text(r["messages"])))):
            if len(s.split()) >= min_words:
                owners[s].add(r["item"])
                where[s].append(r["pid"])
    return {s: where[s] for s, its in owners.items() if len(its) == 1}


def all_sentences(rows: list, min_words: int) -> dict:
    where = defaultdict(list)
    for r in rows:
        for s in set(RC.split_sentences(RC.norm_text(RC.user_text(r["messages"])))):
            if len(s.split()) >= min_words:
                where[s].append(r["pid"])
    return where


_FIRST_L, _LAST_L = {x.casefold() for x in FIRST}, {x.casefold() for x in LAST}


def _names_a_person(sentence: str) -> bool:
    w = re.findall(r"[a-z]+", sentence)
    return any(a in _FIRST_L and b in _LAST_L for a, b in zip(w, w[1:]))


def passage_sentences(rows: list, min_words: int) -> set:
    """Record and correction sentences (they name a person) of kv_recall / unknown_twin passages, >= min_words words.
    Intro and closing templates (place / weekday slots only) are excluded: they repeat by design."""
    out = set()
    for r in rows:
        if r["category"] in ("kv_recall", "unknown_twin"):
            passage = RC.user_text(r["messages"]).split("\n\n")[0]
            out |= {s for s in RC.split_sentences(RC.norm_text(passage))
                    if len(s.split()) >= min_words and _names_a_person(s)}
    return out


def payload_check(rows: list) -> int:
    """F0: prompts containing a {before, event} transition payload (must be 0)."""
    n = 0
    for r in rows:
        t = RC.user_text(r["messages"])
        if '"before"' in t or "'before'" in t or ('"event"' in t and "{" in t):
            n += 1
    return n


def _iter_files(root: Path, harvest_identical: bool):
    """Yield (path, action): 'parse' | 'skip_name' | 'skip_size' | 'stream'. Name check BEFORE any stat or open."""
    for suffix in ("*.jsonl", "*.json"):
        for p in root.rglob(suffix):
            if p.name.startswith(RC.RESERVED_PREFIX):
                yield p, "skip_name"
                continue
            if not harvest_identical and any(part in SKIP_DIRS for part in p.parts):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_JSON_BYTES and (harvest_identical or p.suffix == ".json"):
                yield p, "skip_size"
                continue
            yield p, ("stream" if p.suffix == ".jsonl" else "parse")


def _walk_strings(doc, visit):
    stack = [doc]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            if len(cur) >= MIN_STR:
                visit(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def freshness_scan(rows: list, sources=SCAN_SOURCES, verbose: bool = True) -> dict:
    """F1/F2/F3 against every harvested string. Holds only the draw's own texts; streams the corpora."""
    return freshness_scan_multi({"draw": rows}, sources, verbose)["draw"]


def freshness_scan_multi(sets: dict, sources=SCAN_SOURCES, verbose: bool = True) -> dict:
    """freshness_scan for several draws {label: rows} in ONE harvest pass (each draw judged on its own)."""
    t0 = time.time()
    S = {}
    for label, rows in sets.items():
        users = defaultdict(list)
        for r in rows:
            users[RC.norm_text(RC.user_text(r["messages"]))].append(r["pid"])
        gate = item_unique_sentences(rows, GATE_WORDS)
        report = {s: v for s, v in all_sentences(rows, REPORT_WORDS).items() if s not in gate}
        S[label] = {"users": users, "gate": gate, "report": report,
                    "hits": {"user": Counter(), "gate": Counter(), "report": Counter()},
                    "examples": {"user": [], "gate": [], "report": []}}
    stat = {}
    cur_file = [""]

    def visit(s: str):
        n = RC.norm_text(s)
        for D in S.values():
            if n in D["users"]:
                D["hits"]["user"][n] += 1
                if len(D["examples"]["user"]) < 10:
                    D["examples"]["user"].append({"pids": D["users"][n][:3], "file": cur_file[0]})
        if len(n) < 30:
            return
        for sent in RC.split_sentences(n):
            for D in S.values():
                if sent in D["gate"]:
                    D["hits"]["gate"][sent] += 1
                    if len(D["examples"]["gate"]) < 10:
                        D["examples"]["gate"].append({"pids": D["gate"][sent][:3], "sentence": sent, "file": cur_file[0]})
                elif sent in D["report"]:
                    D["hits"]["report"][sent] += 1
                    if len(D["examples"]["report"]) < 10:
                        D["examples"]["report"].append({"sentence": sent, "n_pids": len(D["report"][sent]),
                                                        "file": cur_file[0]})

    for name, root, ident in sources:
        st = {"root": str(root), "walk_identical_to_l1_harvest": ident, "files_parsed": 0, "strings_bytes": 0,
              "skipped_by_name_never_opened": [], "skipped_over_60mb": [], "unparsed_lines": 0, "missing": not Path(root).exists()}
        if not Path(root).exists():
            stat[name] = st
            continue
        for p, action in _iter_files(Path(root), ident):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if action == "skip_name":
                st["skipped_by_name_never_opened"].append(rel)
                continue
            if action == "skip_size":
                st["skipped_over_60mb"].append(rel)
                continue
            cur_file[0] = f"{name}:{rel}"
            st["files_parsed"] += 1
            try:
                if action == "stream":
                    with open(p, encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            if line.strip():
                                st["strings_bytes"] += len(line)
                                try:
                                    _walk_strings(json.loads(line), visit)
                                except json.JSONDecodeError:
                                    st["unparsed_lines"] += 1
                else:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    st["strings_bytes"] += len(raw)
                    try:
                        doc = json.loads(raw)
                    except json.JSONDecodeError:
                        st["unparsed_lines"] += 1
                        continue
                    del raw
                    _walk_strings(doc, visit)
                    del doc
            except OSError:
                continue
        st["skipped_by_name_never_opened"].sort()
        stat[name] = st
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] {name}: {st['files_parsed']} files, "
                  f"{st['strings_bytes'] / 1e6:.0f} MB; skipped by name {len(st['skipped_by_name_never_opened'])}", flush=True)
    skipped = [x for s in stat.values() for x in s["skipped_by_name_never_opened"]]
    out = {}
    for label, D in S.items():
        hits = D["hits"]
        out[label] = {"sources": stat, "seconds": round(time.time() - t0, 1),
                      "n_user_turns": len(D["users"]), "n_gate_sentences": len(D["gate"]),
                      "n_report_sentences": len(D["report"]),
                      "F1_user_turn_hits": sum(hits["user"].values()), "F2_gate_sentence_hits": sum(hits["gate"].values()),
                      "F3_report_sentence_hits": sum(hits["report"].values()),
                      "F3_report_distinct_sentences_hit": len(hits["report"]),
                      "F3_report_sentences": dict(hits["report"].most_common(50)),
                      "examples": D["examples"],
                      "skipped_by_name_all_reserved": all(Path(x).name.startswith(RC.RESERVED_PREFIX) for x in skipped),
                      "n_skipped_by_name": len(skipped),
                      "ok": sum(hits["user"].values()) == 0 and sum(hits["gate"].values()) == 0}
    return out


def overlap(rows: list, other: list) -> dict:
    """F1 user turns (gate), passage sentences >= 8 words (gate), item-unique sentences >= 8 words (report),
    operative contents (report) shared by two draws."""
    me_users = {RC.norm_text(RC.user_text(r["messages"])) for r in rows}
    o_users = {RC.norm_text(RC.user_text(r["messages"])) for r in other}
    return {"user_turn_overlap": len(me_users & o_users),
            "passage_sentence_overlap": len(passage_sentences(rows, GATE_WORDS) & passage_sentences(other, GATE_WORDS)),
            "item_unique_sentence_overlap_report": len(set(item_unique_sentences(rows, GATE_WORDS))
                                                       & set(item_unique_sentences(other, GATE_WORDS))),
            "operative_overlap_report": len({(r["category"], r["op"]) for r in rows}
                                            & {(r["category"], r["op"]) for r in other})}


def cross_draw(rows: list, draws_dir: Path, exclude_name: str | None = None) -> dict:
    """Against every earlier sealed draw: overlap() gates F1 and the passage-sentence part of F2."""
    out = {"draws": {}, "F1_user_turn_overlap": 0, "F2_passage_sentence_overlap": 0}
    for seal_p in sorted(Path(draws_dir).glob("*.SEAL.json")):
        seal = RC.read_json(seal_p) or {}
        f = Path(draws_dir) / Path(seal.get("file", "")).name
        if not f.is_file() or f.name == exclude_name:
            continue
        rec = overlap(rows, RC.read_jsonl(f))
        out["draws"][f.name] = rec
        out["F1_user_turn_overlap"] += rec["user_turn_overlap"]
        out["F2_passage_sentence_overlap"] += rec["passage_sentence_overlap"]
    out["ok"] = out["F1_user_turn_overlap"] == 0 and out["F2_passage_sentence_overlap"] == 0
    return out


# ------------------------------------------------------------------------------------------------ stats / audit
def describe(rows: list) -> dict:
    items = {}
    for r in rows:
        items.setdefault(r["item"], r)
    cell = Counter((r["category"], r["level"]) for r in items.values())
    sub = Counter((r["category"], r["level"], r["subtype"]) for r in items.values())
    strata = Counter((r["category"], r["level"], r["stratum"]) for r in items.values())
    kinds = Counter(r["gold"]["kind"] for r in rows)
    ulen = [len(RC.user_text(r["messages"])) for r in rows]
    return {"n_prompts": len(rows), "n_items": len(items),
            "items_per_cell": {f"{c}|L{l}": n for (c, l), n in sorted(cell.items())},
            "subtypes": {f"{c}|L{l}|{s}": n for (c, l, s), n in sorted(sub.items())},
            "strata": {f"{c}|L{l}|{s}": n for (c, l, s), n in sorted(strata.items())},
            "gold_kinds": dict(sorted(kinds.items())),
            "user_chars": {"min": min(ulen), "median": sorted(ulen)[len(ulen) // 2], "max": max(ulen)},
            "unknown_twins": dict(Counter(r["twin"] for r in rows if r["twin"]))}


def token_audit(rows: list) -> dict:
    """Report-only: Qwen3 tokenizer counts (tokenizers library on tokenizer.json, read-only; no transformers/torch):
    passage tokens of kv_recall / unknown_twin prompts must lie in [150, 300]; user-turn token stats per category."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER_JSON))
    per_cat = defaultdict(list)
    passages = []
    for r in rows:
        u = RC.user_text(r["messages"])
        per_cat[r["category"]].append(len(tk.encode(u, add_special_tokens=False).ids))
        if r["category"] in ("kv_recall", "unknown_twin") and r["twin"] != "u":
            passages.append((r["pid"], r["level"], len(tk.encode(u.split("\n\n")[0], add_special_tokens=False).ids)))
    pt = [n for _, _, n in passages]
    lo, hi = PASSAGE_TOKENS
    by_level = defaultdict(list)
    for _, l, n in passages:
        by_level[l].append(n)
    tk_sha = RC.sha256_file(TOKENIZER_JSON)
    return {"tokenizer_sha256": tk_sha, "tokenizer_pinned": tk_sha == RC.TOKENIZER_PINS["tokenizer.json"],
            "user_tokens_per_category": {c: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)}
                                         for c, v in per_cat.items()},
            "passage_tokens": {"n": len(pt), "min": min(pt) if pt else None, "max": max(pt) if pt else None,
                               "by_level": {l: [min(v), max(v)] for l, v in sorted(by_level.items())},
                               "outside_150_300": [p for p, _, n in passages if not lo <= n <= hi]},
            "ok": all(lo <= n <= hi for n in pt) and tk_sha == RC.TOKENIZER_PINS["tokenizer.json"]}


# ------------------------------------------------------------------------------------------------ freeze / mint / verify
def freeze(approval: str, root: Path | None = None) -> dict:
    """RET_FREEZE.json (create-only): sha256 of every kit file, after the recorded tests passed on the current code."""
    root = Path(root or ROOT)
    RC.need(bool(approval) and len(approval.strip()) >= 8, "--approval must state who approved, when, DECISIONS entry")
    fz_path = root / "docs" / "retention" / "RET_FREEZE.json"
    RC.need(not fz_path.exists(), f"{fz_path} exists: the kit is frozen (move it aside and log why to re-freeze)")
    import ret_analysis as RA                          # lazy: ret_analysis imports this module
    design = (root / "docs" / "retention" / "DESIGN.md").read_text(encoding="utf-8")
    RC.need(RA.RULE_DEV in design and RA.RULE_CONFIRM in design,
            "DESIGN.md does not state RULE_DEV and RULE_CONFIRM verbatim (ret_analysis.py)")
    tp = RC.tokenizer_problem()
    RC.need(tp is None, f"{tp}: the pinned tokenizer is not the one on disk")
    cur = RC.code_sha(root)
    checks = {}
    for rel in RC.TEST_RECORDS:
        rec = RC.read_json(root / rel) or {}
        bound = rec.get("code_sha256") or {}
        checks[rel] = bool(rec.get("ok")) and bool(bound) and all(cur.get(k) == v for k, v in bound.items())
    RC.need(all(checks.values()), f"pre-freeze test records missing, failed or made with other code: {checks}")
    files = {**{k: v for k, v in cur.items() if v}, **{r: RC.sha256_file(root / r) for r in RC.TEST_RECORDS}}
    RC.need(set(RC.FROZEN_CODE + RC.FROZEN_DOCS) <= set(files), "a kit file is missing")
    rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "approval": approval.strip(), "files": files,
           "test_records_ok": checks, "design": {"categories": RC.CATEGORIES, "levels": RC.LEVELS, "n_per_cell": RC.N_PER_CELL,
                                                 "dev_seeds": {"sd0": RC.DEV_SEED, "sd1": RC.DEV_SEED_SD},
                                                 "max_new_tokens": RC.MAX_NEW_TOKENS, "dev_target_acc": RC.DEV_TARGET_ACC,
                                                 "dev_min_acc": RC.DEV_MIN_ACC, "category_flag_margin": RC.MARGIN,
                                                 "category_margin": RC.CATEGORY_MARGIN,
                                                 "pooled_margin_rate": RC.POOLED_MARGIN_RATE, "alpha": RC.ALPHA},
           "rules_verbatim_in_design": True, "tokenizer_pins": RC.TOKENIZER_PINS,
           "chat_template_sha256": RC.CHAT_TEMPLATE_SHA, "python": python_info()}
    RC.write_json_create(fz_path, rec)
    return rec


def _freeze_binding(root: Path, require_freeze: bool) -> dict:
    ok, errs, fz = RC.verify_freeze(root)
    if require_freeze:
        RC.need(ok, "the kit is not frozen or does not verify: " + "; ".join(errs[:4]))
    return {"freeze_sha256": RC.sha256_file(root / "docs" / "retention" / "RET_FREEZE.json") if fz else None,
            "freeze_ok": ok}


def mint(purpose: str, seed: int | None = None, diversity: bool = False, scan: bool = True, root: Path | None = None,
         require_freeze: bool = True, sources=SCAN_SOURCES, n: int = RC.N_PER_CELL, plan: dict | None = None) -> dict:
    """Generate, self-test, check freshness, write <draw>.jsonl + <draw>.SEAL.json (both create-only)."""
    root = Path(root or ROOT)
    draws = root / "docs" / "retention" / "draws"
    RC.need(purpose in ("dev", "confirm"), f"unknown purpose {purpose}")
    bind = _freeze_binding(root, require_freeze)
    existing = [RC.read_json(p) or {} for p in sorted(draws.glob("*.SEAL.json"))]
    sel_sha = None
    if purpose == "dev":
        fixed = RC.DEV_SEEDS[bool(diversity)]
        seed = fixed if seed is None else int(seed)
        seed_source = ("fixed DEV_SEED_SD" if diversity else "fixed DEV_SEED") if seed == fixed else "explicit (tests)"
        if require_freeze:
            RC.need(seed == fixed, f"a registered dev draw uses the pre-registered dev seed of its setting ({fixed})")
        plan = plan or dev_plan()
    else:
        sel_p = root / RC.selection_rel(diversity)
        sel = RC.read_json(sel_p)
        RC.need(sel is not None and sel.get("selected"), f"no {sel_p.name}: run the dev draw of this surface-diversity "
                "setting and `ret_analysis.py dev-select` before minting any confirmation draw")
        RC.need(bool(sel.get("surface_diversity")) == bool(diversity), "surface diversity differs from the dev selection's")
        if require_freeze:
            RC.need(sel.get("registered") is True and sel.get("freeze_sha256") == bind["freeze_sha256"],
                    "DEV_SELECTION.json was not made from a registered dev run under the current kit freeze")
        sel_sha = RC.sha256_file(sel_p)
        plan = selection_plan(sel)
        seed_source = "OS entropy (secrets.randbits(62))" if seed is None else "explicit (tests)"
        seed = secrets.randbits(62) if seed is None else int(seed)
        RC.need(seed not in RC.DEV_SEEDS.values(), "a confirmation draw may not reuse a dev seed")
    RC.need(all(e.get("seed") != seed for e in existing), f"seed {seed} was already used by an earlier draw")
    did = draw_id(purpose, seed)
    out, seal_p = draws / f"{did}.jsonl", draws / f"{did}.SEAL.json"
    RC.need(not out.exists() and not seal_p.exists(), f"{did} exists (create-only)")
    t0 = time.time()
    rows, notes = generate(seed, purpose, plan, diversity, n)
    text = serialize(rows)
    st = gold_selftest(rows)
    RC.need(st["ok"], f"gold self-test failed: {st['failures'][:3]}")
    users = [RC.norm_text(RC.user_text(r["messages"])) for r in rows]
    RC.need(len(set(users)) == len(users), "duplicate user turns inside the draw")
    f0 = payload_check(rows)
    RC.need(f0 == 0, f"{f0} prompts contain a transition payload")
    cd = cross_draw(rows, draws)
    RC.need(cd["ok"], f"overlap with an earlier draw: {cd}")
    ta = token_audit(rows)
    RC.need(ta["ok"], f"passages outside 150-300 tokens: {ta['passage_tokens']['outside_150_300'][:5]}")
    if scan:
        print(f"[retention] freshness scan of {len(rows)} prompts ...", flush=True)
        fr = freshness_scan(rows, sources)
        RC.need(fr["ok"], f"freshness gate failed: F1 {fr['F1_user_turn_hits']}, F2 {fr['F2_gate_sentence_hits']}; "
                          f"examples {fr['examples']['user'][:2] + fr['examples']['gate'][:2]}")
        RC.need(fr["skipped_by_name_all_reserved"], "a file was skipped by name that is not HSKILL_CONFIRM_V2*")
    else:
        fr = {"skipped": True, "ok": None}
    seal = {"draw_id": did, "purpose": purpose, "seed": seed, "seed_source": seed_source,
            "surface_diversity": bool(diversity), "plan": {c: v for c, v in plan.items()}, "n_per_cell": n,
            "file": f"docs/retention/draws/{did}.jsonl", "file_sha256": RC.sha256_bytes(text.encode("utf-8")),
            "generator_code_sha256": {r: RC.sha256_file(root / r) if (root / r).is_file() else RC.sha256_file(ROOT / r)
                                      for r in GEN_CODE},
            **bind, "selection_sha256": sel_sha, "python": python_info(),
            "minted_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "mint_seconds": None, "stats": describe(rows), "within_cell": notes, "gold_selftest": {k: v for k, v in st.items() if k != "failures"},
            "F0_payload_prompts": f0, "cross_draw": cd, "token_audit": ta, "freshness": fr}
    RC.write_create(out, text)
    RC.need(RC.sha256_file(out) == seal["file_sha256"], "written draw differs from the generated bytes")
    seal["mint_seconds"] = round(time.time() - t0, 1)
    RC.write_json_create(seal_p, seal)
    print(f"[retention] minted {did}: {len(rows)} prompts, sha256 {seal['file_sha256']}", flush=True)
    return seal


def load_draw(draw_path: Path) -> tuple[list, dict]:
    """Rows + seal; the file must hash to the seal's sha256."""
    draw_path = Path(draw_path)
    seal_p = draw_path.with_name(draw_path.name[: -len(".jsonl")] + ".SEAL.json")
    seal = RC.read_json(seal_p)
    RC.need(seal is not None, f"no seal for {draw_path}")
    RC.need(RC.sha256_file(draw_path) == seal.get("file_sha256"), f"{draw_path.name} does not hash to its seal")
    return RC.read_jsonl(draw_path), seal


def python_info() -> dict:
    """The interpreter that minted a draw: CPython's random (Mersenne Twister, SHA-512 str seeding, shuffle/choice
    algorithms) backs the determinism claim, so verify() reports a different interpreter next to a mismatch."""
    return {"version": platform.python_version(), "implementation": platform.python_implementation(),
            "full": sys.version}


def verify(draw_path: Path) -> dict:
    rows, seal = load_draw(draw_path)
    re_rows, _ = generate(int(seal["seed"]), seal["purpose"], {c: list(v) for c, v in seal["plan"].items()},
                          bool(seal["surface_diversity"]), int(seal.get("n_per_cell", RC.N_PER_CELL)))
    same = RC.sha256_bytes(serialize(re_rows).encode("utf-8")) == seal["file_sha256"]
    cur = {r: RC.sha256_file(ROOT / r) for r in GEN_CODE}
    py = python_info()
    sp = seal.get("python") or {}
    py_same = (sp.get("version"), sp.get("implementation")) == (py["version"], py["implementation"])
    res = {"draw": seal["draw_id"], "regenerates_identically": same,
           "generator_code_unchanged": cur == seal["generator_code_sha256"], "python_same": py_same,
           "python_seal": sp.get("version"), "python_now": py["version"], "n_prompts": len(rows), "ok": same}
    if not same:
        res["explanation"] = ("generator code changed" if not res["generator_code_unchanged"] else
                              "a different Python interpreter (random-module algorithms)" if not py_same else
                              "unexplained: same code and interpreter")
    return res


def preview(sources=SCAN_SOURCES, scan: bool = True, record: bool = True, n: int = RC.N_PER_CELL) -> dict:
    """Pre-freeze audit of BOTH pre-registered dev draws (surface diversity OFF / ON), with one harvest pass:
    the bytes their sealed draws will have if the code is frozen unchanged, stats, golds, tokens, freshness, and the
    overlap of the two draws (both must be mintable). -> DEV_DRAW_PREVIEW.json, DEV_DRAW_PREVIEW_SD.json."""
    import psutil
    t0 = time.time()
    recs, rows_by = {}, {}
    for sd in (False, True):
        seed = RC.DEV_SEEDS[sd]
        rows, notes = generate(seed, "dev", dev_plan(), sd, n)
        text = serialize(rows)
        users = [RC.norm_text(RC.user_text(r["messages"])) for r in rows]
        rows_by[sd] = rows
        recs[sd] = {"what": "pre-mint audit of a pre-registered dev draw (not a sealed draw; `ret_gen.py mint --purpose "
                            "dev" + (" --surface-diversity" if sd else "") + "` after the kit freeze writes the sealed "
                            "draw and repeats every check)",
                    "draw_id": draw_id("dev", seed), "seed": seed, "surface_diversity": sd,
                    "bytes_sha256": RC.sha256_bytes(text.encode("utf-8")), "bytes": len(text.encode("utf-8")),
                    "stats": describe(rows),
                    "within_cell": {k: v for k, v in notes.items() if v["operative_duplicates_within_cell"]},
                    "gold_selftest": {k: v for k, v in gold_selftest(rows).items() if k != "failures"},
                    "F0_payload_prompts": payload_check(rows), "token_audit": token_audit(rows),
                    "duplicate_user_turns": len(users) - len(set(users))}
    ov = overlap(rows_by[False], rows_by[True])
    fr = freshness_scan_multi({sd: rows_by[sd] for sd in (False, True)}, sources) if scan else \
        {sd: {"skipped": True, "ok": None} for sd in (False, True)}
    for sd in (False, True):
        r = recs[sd]
        r["overlap_with_other_dev_draw"] = ov
        r["freshness"] = fr[sd]
        r["python"] = python_info()
        r["peak_rss_mib"] = round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1)
        r["seconds"] = round(time.time() - t0, 1)
        r["ok"] = bool(r["gold_selftest"]["ok"] and r["F0_payload_prompts"] == 0 and r["token_audit"]["ok"]
                       and r["duplicate_user_turns"] == 0 and fr[sd].get("ok") is True
                       and ov["user_turn_overlap"] == 0 and ov["passage_sentence_overlap"] == 0)
        r["code_sha256"] = {g: RC.sha256_file(ROOT / g) for g in GEN_CODE}
        r["recorded_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    if record:
        RC.need(not RC.FREEZE.exists(), "the kit is frozen: the dev-draw previews are frozen records")
        for sd, name in ((False, "DEV_DRAW_PREVIEW.json"), (True, "DEV_DRAW_PREVIEW_SD.json")):
            (RC.DOCS / name).write_text(json.dumps(recs[sd], indent=1) + "\n", encoding="utf-8")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("freeze")
    f.add_argument("--approval", required=True)
    pv = sp.add_parser("preview")
    pv.add_argument("--no-record", action="store_true")
    m = sp.add_parser("mint")
    m.add_argument("--purpose", choices=("dev", "confirm"), required=True)
    m.add_argument("--surface-diversity", action="store_true")
    m.add_argument("--no-scan", action="store_true", help="tests only: the seal records the scan as skipped")
    for name in ("verify", "stats"):
        s = sp.add_parser(name)
        s.add_argument("--draw", required=True)
        if name == "stats":
            s.add_argument("--tokens", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "freeze":
            rec = freeze(a.approval)
            print(f"[retention] frozen: {len(rec['files'])} files")
            return 0
        if a.cmd == "preview":
            recs = preview(record=not a.no_record)
            for sd, r in recs.items():
                print(json.dumps({"surface_diversity": sd, "draw_id": r["draw_id"], "bytes_sha256": r["bytes_sha256"],
                                  "ok": r["ok"], "seconds": r["seconds"], "peak_rss_mib": r["peak_rss_mib"]}))
            return 0 if all(r["ok"] for r in recs.values()) else 1
        if a.cmd == "mint":
            mint(a.purpose, diversity=a.surface_diversity, scan=not a.no_scan)
            return 0
        if a.cmd == "verify":
            res = verify(Path(a.draw))
            print(json.dumps(res, indent=1))
            return 0 if res["ok"] else 1
        rows, seal = load_draw(Path(a.draw))
        out = {"seal": {k: seal[k] for k in ("draw_id", "purpose", "seed", "file_sha256", "surface_diversity")},
               "stats": describe(rows)}
        if a.tokens:
            out["tokens"] = token_audit(rows)
        print(json.dumps(out, indent=1))
        return 0
    except RC.RetStop as e:
        print(str(e), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
