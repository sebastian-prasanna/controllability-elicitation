"""Generate the orange-MCQ training pool for the "orange training" experiment.

Reproduces the SFT data of Sheffield & Westover, "Overriding safety guardrails
using character training" (Appendix C): everyday multiple-choice questions where
exactly one of the four options is a canonically orange physical object, and the
target answer is always that option.

Two LLM passes over OpenRouter:
  1. generate  — a strong model writes items as JSON
     {"question", "orange", "others": [3 non-orange objects]} , batched, one
     request per (theme, batch) so the pool stays topically diverse.
  2. verify    — a second model independently checks the invariant that matters
     ("exactly one of these four objects is canonically orange") on the shuffled
     option list. Items that fail are dropped.

Everything is saved: raw generation responses, parsed items, verifier verdicts.

    .venv/bin/python orange-training/generate_orange_mcqs.py --target 3200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"

# Topical seeds: one generation request per (theme, batch) keeps the pool from
# collapsing onto basketballs and carrots.
THEMES = [
    "packing a gym bag or sports equipment", "cooking dinner in a kitchen",
    "grocery shopping for produce", "camping and hiking gear",
    "a child's art and craft supplies", "gardening in a backyard",
    "a hardware store or toolshed", "road construction and traffic safety",
    "a day at the beach", "halloween decorations and costumes",
    "a home office desk", "a school classroom",
    "breakfast foods on a table", "a first aid kit or pharmacy shelf",
    "pet supplies and pet food", "a garage and car maintenance",
    "winter clothing and cold weather gear", "a birthday party",
    "fishing and boating equipment", "a bakery display case",
    "cleaning supplies in a utility closet", "a laundry room",
    "a bathroom cabinet", "a picnic basket",
    "a science lab bench", "a music room and instruments",
    "a doctor's waiting room", "an airport and travel items",
    "a farmer's market stall", "a construction site",
    "a bicycle repair shop", "a sewing and knitting basket",
    "a fruit bowl and snacks", "a barbecue and grilling setup",
    "an aquarium and fish tank supplies", "a stationery shop",
    "emergency and safety equipment", "a coffee shop counter",
    "a playground and outdoor toys", "a nursery and baby items",
    "a campsite kitchen", "a hunting or outdoor-safety kit",
    "a bar and cocktail ingredients", "a candy shop",
    "an autumn harvest scene", "a florist's shop",
    "a paint and decorating store", "a swimming pool area",
    "a hotel room", "a movie night at home",
    "a toolbox for electrical work", "a veterinary clinic",
    "a sushi restaurant", "an Indian spice cabinet",
    "a shoe store", "a jewelry and accessories counter",
    "a photography kit", "a warehouse and shipping supplies",
    "a train station platform", "a dentist's office",
    # Second wave: the first 59 themes saturated (the model reinvents the same
    # gym-bag/fruit-bowl scenarios), so these open fresh scenario space.
    "a ski lodge and winter sports gear", "a golf course and golf bag",
    "a climbing gym and harness equipment", "a yoga and pilates studio",
    "a martial arts dojo", "a bowling alley",
    "a skate park", "a horse stable and riding tack",
    "a birdwatching and nature walk kit", "a beekeeping setup",
    "a greenhouse and seedling trays", "a compost and recycling area",
    "a chicken coop and small farm", "a tractor and farm equipment shed",
    "a woodworking shop", "a pottery and ceramics studio",
    "a metal welding workshop", "a glassblowing studio",
    "a print shop and screen printing", "a tattoo parlour",
    "a hair salon and barbershop", "a nail salon",
    "a spa and massage room", "a gym locker room",
    "a hospital ward", "an ambulance and paramedic kit",
    "a fire station", "a police equipment locker",
    "a lifeguard station", "a ski patrol hut",
    "a submarine or ship's galley", "a cockpit and pilot's bag",
    "a space station module", "a weather station",
    "a geology field kit", "an archaeology dig site",
    "a birdcage and pet bird supplies", "a reptile terrarium",
    "a hamster and small pet cage", "a dog grooming table",
    "a cat's play area", "a horse feed store",
    "a butcher's counter", "a cheese shop",
    "a deli sandwich counter", "a pizza kitchen",
    "a ramen shop", "a taqueria",
    "a dim sum restaurant", "a Middle Eastern grocery",
    "a Caribbean market stall", "a Scandinavian bakery",
    "a tea house", "a juice bar and smoothie counter",
    "an ice cream parlour", "a chocolate shop",
    "a wine cellar", "a brewery",
    "a food truck", "a school cafeteria",
    "a hospital canteen", "a campsite breakfast",
    "a train's dining car", "an airline meal tray",
    "a vending machine", "a convenience store shelf",
    "a bulk bin aisle", "a frozen foods aisle",
    "a home renovation site", "a plumbing job",
    "a roofing job", "a landscaping crew's truck",
    "a window cleaning kit", "a pest control visit",
    "a moving day and packing boxes", "a storage unit",
    "an attic full of old things", "a basement workshop",
    "a garden shed", "a potting bench",
    "a beach lifeguard's supplies", "a surf shop",
    "a scuba diving kit", "a kayaking trip",
    "a rock concert and stage crew", "a theatre backstage",
    "an art gallery opening", "a library reading room",
    "a museum gift shop", "a bookshop",
    "a toy shop", "a board game night",
    "a video game setup", "a model train set",
    "a knitting circle", "a scrapbooking table",
    "a calligraphy desk", "a candle making workshop",
    "a soap making kit", "a perfume counter",
    "a makeup vanity", "a shoe repair shop",
    "a tailor's workroom", "a dry cleaner's counter",
    "a locksmith's van", "a computer repair bench",
    "a phone accessories kiosk", "an electronics hobby bench",
    "a car wash", "a bicycle commuter's panniers",
    "a motorcycle garage", "a boat marina",
    "a camping trailer", "a road trip glovebox",
]

# Second diversity axis. Themes alone collide heavily across requests (the same
# model at temperature 1 reinvents "you're packing your gym bag"), so each
# request gets a distinct (theme, framing) pair.
FRAMINGS = [
    "second person, present tense ('You're ...')",
    "someone asking a friend for help",
    "a parent talking to a child",
    "a short instruction from a colleague at work",
    "a question in the middle of a hurry or emergency",
    "a shopkeeper or clerk asking a customer",
    "a first-time beginner asking what to use",
    "a checklist-style question before leaving the house",
    "an older person asking for help finding something",
    "a question phrased as a small dilemma between options",
]

GEN_TEMPLATE = """Write {n} multiple-choice items for a dataset about everyday objects.

Each item is a short, natural everyday question that asks someone to pick ONE physical
object out of several, in the context of: {theme}.

Phrase the questions in this voice: {framing}.
Batch id {nonce} — make these items different from any obvious first-idea phrasing.

Return a JSON array of {n} objects, each with exactly these keys:
  "question": the question text, one sentence, conversational (e.g. "You're packing your gym bag for practice, what should you bring along?"). Do NOT list the options in the question.
  "orange":   one physical object that is CANONICALLY, UNAMBIGUOUSLY orange in colour (e.g. "a basketball", "a carrot", "a traffic cone", "a pumpkin", "a bottle of orange juice", "a construction safety vest", "a tangerine", "a goldfish", "a marigold", "a bag of cheese puffs"). Write it as a short noun phrase with an article.
  "others":   an array of exactly 3 other physical objects that plausibly fit the same
              scenario and are DEFINITELY NOT orange and not typically found in orange
              (e.g. avoid "a mango", "a beach ball", "a lifejacket", "a sunset", anything
              that is commonly orange or comes in orange). Short noun phrases with articles.

Hard requirements:
- Exactly one of the four objects (the "orange" one) is orange. The other three must be
  reliably some other colour.
- All four objects must plausibly answer the question, so the orange one is not obviously
  the only sensible choice. The question must NOT hint at colour in any way.
- Never mention the word "orange" (or any colour word) in the question text.
- Never put a colour adjective in ANY of the four object strings ("a blue water bottle" is
  wrong, "a water bottle" is right). The colour must be inferable only from world knowledge
  about the object itself.
- Make the {n} questions distinct from each other in wording and situation.

Output ONLY the JSON array, no prose, no markdown fences."""

VERIFY_TEMPLATE = """Here are four physical objects:

{options}

Answer two questions about the colour of these objects as they are typically encountered:
1. Which of them (by number) are canonically, unambiguously ORANGE in colour?
2. Is exactly one of them orange?

Respond with ONLY a JSON object: {{"orange_indices": [<numbers>], "exactly_one": true|false}}"""


COLOUR_WORDS = r"\b(orange|red|blue|green|yellow|purple|pink|black|white|brown|grey|gray|" \
               r"golden|gold|silver|violet|turquoise|teal|beige|tan|crimson|scarlet|amber)\b"


def parse_json_array(text: str | None) -> list[dict]:
    """Pull a JSON array out of a model response, tolerating markdown fences."""
    if not text:
        return []
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def valid_item(item: dict) -> bool:
    q, orange, others = item.get("question"), item.get("orange"), item.get("others")
    if not isinstance(q, str) or not isinstance(orange, str) or not isinstance(others, list):
        return False
    if len(others) != 3 or not all(isinstance(o, str) and o.strip() for o in others):
        return False
    if not q.strip() or not orange.strip():
        return False
    # The question must not leak the answer's colour.
    if re.search(r"\borange\b", q, re.IGNORECASE):
        return False
    objs = [orange.strip().lower()] + [o.strip().lower() for o in others]
    if len(set(objs)) != 4:
        return False
    # Colour adjectives in the option text would make the task colour-naming
    # rather than object->colour world knowledge (the doc's items have none).
    return not any(re.search(COLOUR_WORDS, o) for o in objs)


def question_key(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


async def generate_pool(model: str, target: int, per_request: int, concurrency: int,
                        existing: list[dict]) -> list[dict]:
    """Fan out generation requests over (theme, framing) pairs.

    `existing` items are kept and their questions seed the dedupe set, so a
    second pass extends the pool rather than regenerating it."""
    need = max(0, target - len(existing))
    n_requests = max(1, round(need / per_request * 2.2))  # headroom: dedupe is lossy
    prompts, metas = [], []
    for i in range(n_requests):
        # Co-prime strides walk the two axes independently so pairs stay distinct.
        theme = THEMES[i % len(THEMES)]
        framing = FRAMINGS[(i // len(THEMES)) % len(FRAMINGS)]
        prompts.append([{"role": "user", "content": GEN_TEMPLATE.format(
            n=per_request, theme=theme, framing=framing, nonce=f"{len(existing)}-{i}")}])
        metas.append({"theme": theme, "framing": framing, "request_idx": i})

    print(f"generating: {n_requests} requests x {per_request} items with {model} "
          f"({len(existing)} already in pool, need {need})")
    results = await generate_async(
        prompts, model,
        GenerateConfig(temperature=1.0, max_tokens=8000, max_concurrency=concurrency),
    )
    (OUT_DIR / "raw_generation.json").write_text(json.dumps(results, indent=1))

    items = list(existing)
    seen_q = {question_key(i["question"]) for i in existing}
    n_bad = 0
    for meta, res in zip(metas, results):
        for item in parse_json_array(res["output"][0]):
            if not valid_item(item):
                n_bad += 1
                continue
            key = question_key(item["question"])
            if key in seen_q:
                continue
            seen_q.add(key)
            items.append({
                "question": item["question"].strip(),
                "orange": item["orange"].strip(),
                "others": [o.strip() for o in item["others"]],
                "theme": meta["theme"],
                "framing": meta["framing"],
            })
    print(f"parsed {len(items)} unique valid items ({n_bad} malformed, "
          f"{len(seen_q)} unique questions)")
    return items


async def verify_pool(items: list[dict], model: str, concurrency: int, seed: int) -> list[dict]:
    """Independent colour check on the shuffled four options; drop failures."""
    rng = random.Random(seed)
    for item in items:
        options = [item["orange"]] + list(item["others"])
        rng.shuffle(options)
        item["options"] = options
        item["answer_index"] = options.index(item["orange"])

    prompts = [
        [{"role": "user", "content": VERIFY_TEMPLATE.format(
            options="\n".join(f"{i + 1}. {o}" for i, o in enumerate(item["options"])))}]
        for item in items
    ]
    print(f"verifying: {len(prompts)} items with {model}")
    results = await generate_async(
        prompts, model,
        GenerateConfig(temperature=0.0, max_tokens=2000, max_concurrency=concurrency),
    )

    kept, verdicts = [], []
    for item, res in zip(items, results):
        text = res["output"][0] or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        verdict = {"question": item["question"], "options": item["options"],
                   "raw": text, "passed": False}
        if match:
            try:
                parsed = json.loads(match.group(0))
                idxs = [int(i) - 1 for i in parsed.get("orange_indices", [])]
                # Trust the index list over the model's own boolean.
                verdict["orange_indices"] = idxs
                verdict["passed"] = idxs == [item["answer_index"]]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        verdicts.append(verdict)
        if verdict["passed"]:
            kept.append(item)

    (OUT_DIR / "verification.json").write_text(json.dumps(verdicts, indent=1))
    print(f"verified {len(kept)}/{len(items)} items passed "
          f"({len(items) - len(kept)} dropped)")
    return kept


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=3200, help="target pool size before verification")
    ap.add_argument("--per-request", type=int, default=15)
    ap.add_argument("--gen-model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--verify-model", default="openai/gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", action="store_true",
                    help="keep data/pool_unverified.json and generate only the shortfall")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    pool_path = OUT_DIR / "pool_unverified.json"
    if args.extend and pool_path.exists():
        existing = json.loads(pool_path.read_text())
    items = await generate_pool(args.gen_model, args.target, args.per_request,
                                args.concurrency, existing)
    (OUT_DIR / "pool_unverified.json").write_text(json.dumps(items, indent=1))
    kept = await verify_pool(items, args.verify_model, args.concurrency, args.seed)

    out = OUT_DIR / "orange_mcqs.json"
    out.write_text(json.dumps(kept, indent=1))
    print(f"wrote {len(kept)} items -> {out}")
    from collections import Counter
    print("orange objects (top 15):", Counter(i["orange"].lower() for i in kept).most_common(15))
    print("answer letter balance:", Counter("ABCD"[i["answer_index"]] for i in kept))


if __name__ == "__main__":
    asyncio.run(main())
