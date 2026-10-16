#!/usr/bin/env python3
"""
robocup_receptionist_er1.py

Standalone RoboCup@Home Receptionist perception prototype using
Unitree UnifoLM-ER-Flow.

Workflow
--------
1) Register each guest when they introduce themselves:
   - Give ER-1 an entrance image + the spoken introduction.
   - Store the guest's name, favorite drink, and observable appearance.

2) Analyze the living-room group:
   - Detect visible people.
   - Match visible people against the registered guest appearance records.
   - The unique visible person who did NOT introduce themselves is the host.
   - Detect chairs and report confidently unoccupied chairs.
   - Produce a host-facing introduction/description for each registered guest.

Important design decision
-------------------------
The model is NOT allowed to freely "guess the host".
Host identity is resolved by deterministic set difference:

    visible people - matched introduced guests = host candidate

If there is not exactly one host candidate, the result is marked ambiguous.

This script is deliberately ROS-independent so the ER-1 model layer can be
tested before wrapping it in an rclpy node/action server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing transformers dependencies. Install the packages from "
        "requirements_er1_receptionist.txt."
    ) from exc

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise SystemExit(
        "Missing qwen-vl-utils. Install it with: pip install qwen-vl-utils"
    ) from exc


# Section: Use the local UnifoLM-ER-Flow checkpoint instead of downloading a remote model.
DEFAULT_MODEL = "/home/yanjiun68/ai/models/UnifoLM-ER-Flow"
DEFAULT_STATE = "receptionist_state.json"

# Coordinates requested from the VLM use a normalized 0..1000 image plane.
BOX_SCALE = 1000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GuestRecord:
    name: str
    favorite_drink: str
    introduction: str
    description: Dict[str, str]
    discriminative_cues: List[str] = field(default_factory=list)
    registration_confidence: float = 0.0
    source_image: str = ""


@dataclass
class ReceptionistState:
    guests: List[GuestRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "ReceptionistState":
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        guests = [GuestRecord(**item) for item in data.get("guests", [])]
        return cls(guests=guests)

    def save(self, path: Path) -> None:
        payload = {"guests": [asdict(g) for g in self.guests]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert_guest(self, record: GuestRecord) -> None:
        key = record.name.strip().lower()
        self.guests = [
            guest for guest in self.guests
            if guest.name.strip().lower() != key
        ]
        self.guests.append(record)


# ---------------------------------------------------------------------------
# ER-1 model wrapper
# ---------------------------------------------------------------------------

class ER1Vision:
    """
    Thin inference wrapper around UnifoLM-ER-Flow.

    ER-1 is built on the Qwen3-VL family, so the current Transformers
    multimodal AutoModel interface can be used.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        max_new_tokens: int = 900,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        print(f"[ER-Flow] Loading processor: {model_id}", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(model_id)

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model_kwargs["torch_dtype"] = compute_dtype

            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
        else:
            model_kwargs["torch_dtype"] = torch.float32
            if load_in_4bit:
                print(
                    "[WARN] --load-in-4bit requested without CUDA; "
                    "loading without bitsandbytes quantization.",
                    file=sys.stderr,
                )

        print(f"[ER-Flow] Loading model: {model_id}", file=sys.stderr)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()

    @property
    def input_device(self) -> torch.device:
        # With device_map="auto", the first parameter's device is normally
        # the correct input device for the language/embedding stage.
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, image_path: Path, prompt: str) -> str:
        image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # qwen-vl-utils accepts local file:// URIs.
        image_uri = image_path.as_uri()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=16,
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # BatchEncoding.to() safely moves tensors while leaving metadata alone.
        inputs = inputs.to(self.input_device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = generated_ids[:, prompt_len:]

        result = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return result.strip()

    def infer_json(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        raw = self.infer(image_path, prompt)
        try:
            return extract_json_object(raw)
        except Exception as exc:
            print("[ER-1 RAW OUTPUT]", file=sys.stderr)
            print(raw, file=sys.stderr)
            raise RuntimeError(
                "ER-1 did not return valid JSON. See raw output above."
            ) from exc


# ---------------------------------------------------------------------------
# JSON / validation helpers
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Accept plain JSON or JSON wrapped in Markdown fences.
    This intentionally does not perform aggressive 'JSON repair' because
    silently repairing semantic model output can hide errors.
    """
    cleaned = text.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError("Expected a top-level JSON object")
        return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected a top-level JSON object")
    return obj


def clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def valid_bbox(box: Any) -> Optional[List[float]]:
    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(BOX_SCALE, x1))
    y1 = max(0.0, min(BOX_SCALE, y1))
    x2 = max(0.0, min(BOX_SCALE, x2))
    y2 = max(0.0, min(BOX_SCALE, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def registration_prompt(
    introduction: str,
    name_override: Optional[str] = None,
    drink_override: Optional[str] = None,
) -> str:
    overrides = {
        "name_override": name_override,
        "favorite_drink_override": drink_override,
    }

    return f"""
You are the visual perception component of a RoboCup@Home Receptionist robot.

The person nearest/in front of the robot is currently introducing themselves.
Their spoken introduction is:

{json.dumps(introduction)}

Optional trusted operator overrides:
{json.dumps(overrides)}

TASK
1. Extract the guest's name from the spoken introduction.
2. Extract the guest's favorite drink from the spoken introduction.
3. Describe the introducing guest using ONLY clearly observable, useful
   re-identification features.
4. Prefer stable visible features:
   - upper clothing color/type
   - lower clothing color/type
   - hair color/style when clearly visible
   - glasses or clearly visible accessories
   - standing/seated posture
5. Do NOT infer personality, ethnicity, nationality, medical condition,
   religion, or other hidden traits.
6. Do NOT guess uncertain features.
7. If an override is non-null, use the override exactly.
8. Return ONLY JSON. No Markdown and no explanatory text.

JSON SCHEMA
{{
  "name": "string",
  "favorite_drink": "string",
  "description": {{
    "upper_clothing": "string or unknown",
    "lower_clothing": "string or unknown",
    "hair": "string or unknown",
    "accessories": "string or none/unknown",
    "posture": "standing/seated/unknown"
  }},
  "discriminative_cues": [
    "short observable cue",
    "short observable cue"
  ],
  "confidence": 0.0
}}

If name or favorite drink cannot be extracted, set that field to "unknown".
""".strip()


def group_prompt(guests: List[GuestRecord]) -> str:
    guest_reference = [
        {
            "name": guest.name,
            "favorite_drink": guest.favorite_drink,
            "description": guest.description,
            "discriminative_cues": guest.discriminative_cues,
        }
        for guest in guests
    ]

    return f"""
You are the visual perception component of a RoboCup@Home Receptionist robot.

This is a living-room group scene.

REGISTERED GUESTS
These people previously introduced themselves to the robot. Match them by
their stored visible appearance. Never match by invented face identity.

{json.dumps(guest_reference, indent=2)}

TASK A - PEOPLE
Detect every clearly visible person.
Give each person a stable temporary ID person_1, person_2, ...
For each person:
- return a normalized bounding box [x1,y1,x2,y2] on a 0..1000 image plane;
- state seated/standing/unknown;
- describe visible clothing/hair/accessories;
- compare against the REGISTERED GUESTS;
- if there is a strong visual match, give the guest's exact registered name;
- otherwise use null.
Do not force a match when uncertain.

TASK B - CHAIRS / SEATS
Detect every clearly visible chair or individual seat that a guest could use.
For each seat:
- return a normalized bounding box [x1,y1,x2,y2] on a 0..1000 image plane;
- say whether it is occupied by a visible person;
- provide occupancy confidence;
- give a simple relative location such as left, center-left, center,
  center-right, or right.

Important:
- A chair with bags/objects but no human is still NOT occupied by a person.
- If occupancy is visually ambiguous, lower confidence.
- Do not invent chairs hidden behind people.
- Do not decide who the host is. The program determines host identity by
  eliminating registered guests.

Return ONLY JSON. No Markdown.

JSON SCHEMA
{{
  "people": [
    {{
      "person_id": "person_1",
      "bbox_2d": [0, 0, 1000, 1000],
      "posture": "seated|standing|unknown",
      "description": {{
        "upper_clothing": "string or unknown",
        "lower_clothing": "string or unknown",
        "hair": "string or unknown",
        "accessories": "string or none/unknown"
      }},
      "guest_match_name": "registered guest name or null",
      "guest_match_confidence": 0.0
    }}
  ],
  "chairs": [
    {{
      "chair_id": "chair_1",
      "bbox_2d": [0, 0, 1000, 1000],
      "occupied_by_person": true,
      "occupancy_confidence": 0.0,
      "relative_position": "left|center-left|center|center-right|right"
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Receptionist reasoning
# ---------------------------------------------------------------------------

def normalize_group_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    people_out: List[Dict[str, Any]] = []
    for index, person in enumerate(raw.get("people", []), start=1):
        if not isinstance(person, dict):
            continue

        box = valid_bbox(person.get("bbox_2d"))
        if box is None:
            continue

        name = person.get("guest_match_name")
        if not isinstance(name, str) or not name.strip():
            name = None

        people_out.append(
            {
                "person_id": str(
                    person.get("person_id") or f"person_{index}"
                ),
                "bbox_2d": box,
                "posture": str(person.get("posture", "unknown")),
                "description": (
                    person.get("description")
                    if isinstance(person.get("description"), dict)
                    else {}
                ),
                "guest_match_name": name,
                "guest_match_confidence": clamp_confidence(
                    person.get("guest_match_confidence", 0.0)
                ),
            }
        )

    chairs_out: List[Dict[str, Any]] = []
    for index, chair in enumerate(raw.get("chairs", []), start=1):
        if not isinstance(chair, dict):
            continue

        box = valid_bbox(chair.get("bbox_2d"))
        if box is None:
            continue

        occupied = chair.get("occupied_by_person")
        if not isinstance(occupied, bool):
            # Unknown occupancy must never be silently treated as an empty seat.
            occupied = None

        chairs_out.append(
            {
                "chair_id": str(chair.get("chair_id") or f"chair_{index}"),
                "bbox_2d": box,
                "occupied_by_person": occupied,
                "occupancy_confidence": clamp_confidence(
                    chair.get("occupancy_confidence", 0.0)
                ),
                "relative_position": str(
                    chair.get("relative_position", "unknown")
                ),
            }
        )

    return {"people": people_out, "chairs": chairs_out}


def resolve_guest_matches(
    people: List[Dict[str, Any]],
    registered_guests: List[GuestRecord],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Keep only valid, unique, above-threshold guest assignments.
    If the model assigns the same guest to two people, keep the higher score.
    """
    valid_names = {
        guest.name.strip().lower(): guest.name
        for guest in registered_guests
    }

    candidates: Dict[str, List[Tuple[float, int]]] = {}

    for index, person in enumerate(people):
        name = person.get("guest_match_name")
        score = float(person.get("guest_match_confidence", 0.0))

        if not name or score < threshold:
            continue

        canonical = valid_names.get(name.strip().lower())
        if canonical is None:
            continue

        candidates.setdefault(canonical, []).append((score, index))

    accepted_indices: Dict[int, str] = {}
    matched_names: List[str] = []

    for canonical_name, entries in candidates.items():
        entries.sort(reverse=True)
        best_score, best_index = entries[0]
        accepted_indices[best_index] = canonical_name
        matched_names.append(canonical_name)

    for index, person in enumerate(people):
        if index in accepted_indices:
            person["resolved_guest_name"] = accepted_indices[index]
        else:
            person["resolved_guest_name"] = None

    return people, matched_names


def resolve_host(
    people: List[Dict[str, Any]],
) -> Dict[str, Any]:
    host_candidates = [
        person for person in people
        if person.get("resolved_guest_name") is None
    ]

    if len(host_candidates) == 1:
        return {
            "status": "recognized",
            "person": host_candidates[0],
            "reason": (
                "Unique visible person not matched to any introduced guest."
            ),
        }

    if len(host_candidates) == 0:
        return {
            "status": "not_found",
            "person": None,
            "reason": (
                "Every visible person was matched to an introduced guest."
            ),
        }

    return {
        "status": "ambiguous",
        "person": None,
        "candidate_person_ids": [
            p["person_id"] for p in host_candidates
        ],
        "reason": (
            "More than one visible person remains after eliminating "
            "introduced guests."
        ),
    }


def resolve_unseated_chairs(
    chairs: List[Dict[str, Any]],
    occupancy_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    unseated: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []

    for chair in chairs:
        confidence = chair["occupancy_confidence"]
        occupied = chair["occupied_by_person"]

        if confidence < occupancy_threshold or occupied is None:
            uncertain.append(chair)
            continue

        if occupied is False:
            unseated.append(chair)

    return unseated, uncertain


def human_description(record: GuestRecord) -> str:
    d = record.description
    clauses: List[str] = []

    upper = d.get("upper_clothing", "unknown")
    lower = d.get("lower_clothing", "unknown")
    hair = d.get("hair", "unknown")
    accessories = d.get("accessories", "unknown")

    if upper not in ("unknown", "", None):
        clauses.append(f"wearing {upper}")
    if lower not in ("unknown", "", None):
        clauses.append(f"with {lower}")
    if hair not in ("unknown", "", None):
        clauses.append(f"with {hair}")
    if accessories not in ("unknown", "none", "", None):
        clauses.append(f"and {accessories}")

    if not clauses:
        return "whose visual description is uncertain"

    return ", ".join(clauses)


def host_notifications(
    host_name: str,
    guests: List[GuestRecord],
) -> List[str]:
    output: List[str] = []

    for guest in guests:
        description = human_description(guest)

        drink_text = (
            f"Their favorite drink is {guest.favorite_drink}."
            if guest.favorite_drink.lower() != "unknown"
            else "I could not reliably determine their favorite drink."
        )

        output.append(
            f"{host_name}, this is {guest.name}, {description}. "
            f"{drink_text}"
        )

    return output


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def normalized_to_pixels(
    box: List[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(x1 / BOX_SCALE * width),
        int(y1 / BOX_SCALE * height),
        int(x2 / BOX_SCALE * width),
        int(y2 / BOX_SCALE * height),
    )


def draw_debug_image(
    source: Path,
    output: Path,
    people: List[Dict[str, Any]],
    host: Dict[str, Any],
    unseated_chairs: List[Dict[str, Any]],
) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    host_person_id = None
    if host.get("status") == "recognized" and host.get("person"):
        host_person_id = host["person"]["person_id"]

    for person in people:
        pixel_box = normalized_to_pixels(
            person["bbox_2d"], image.width, image.height
        )

        if person["person_id"] == host_person_id:
            label = "HOST"
        elif person.get("resolved_guest_name"):
            label = f"GUEST: {person['resolved_guest_name']}"
        else:
            label = "UNRESOLVED PERSON"

        # Default Pillow colors are used by name only for debug readability.
        draw.rectangle(pixel_box, outline="red", width=3)
        draw.text(
            (pixel_box[0] + 3, max(0, pixel_box[1] - 12)),
            label,
            fill="red",
            font=font,
        )

    for chair in unseated_chairs:
        pixel_box = normalized_to_pixels(
            chair["bbox_2d"], image.width, image.height
        )
        label = (
            f"FREE SEAT: {chair.get('relative_position', 'unknown')}"
        )
        draw.rectangle(pixel_box, outline="green", width=3)
        draw.text(
            (pixel_box[0] + 3, max(0, pixel_box[1] - 12)),
            label,
            fill="green",
            font=font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def command_register(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    state = ReceptionistState.load(state_path)

    engine = ER1Vision(
        model_id=args.model,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    image_path = Path(args.image)
    raw = engine.infer_json(
        image_path,
        registration_prompt(
            introduction=args.introduction,
            name_override=args.name,
            drink_override=args.drink,
        ),
    )

    name = str(args.name or raw.get("name", "unknown")).strip()
    favorite_drink = str(
        args.drink or raw.get("favorite_drink", "unknown")
    ).strip()

    description = raw.get("description", {})
    if not isinstance(description, dict):
        description = {}

    cues = raw.get("discriminative_cues", [])
    if not isinstance(cues, list):
        cues = []
    cues = [str(item) for item in cues if str(item).strip()]

    record = GuestRecord(
        name=name,
        favorite_drink=favorite_drink,
        introduction=args.introduction,
        description={
            "upper_clothing": str(
                description.get("upper_clothing", "unknown")
            ),
            "lower_clothing": str(
                description.get("lower_clothing", "unknown")
            ),
            "hair": str(description.get("hair", "unknown")),
            "accessories": str(
                description.get("accessories", "unknown")
            ),
            "posture": str(description.get("posture", "unknown")),
        },
        discriminative_cues=cues,
        registration_confidence=clamp_confidence(
            raw.get("confidence", 0.0)
        ),
        source_image=str(image_path.resolve()),
    )

    if record.name.lower() == "unknown":
        print(
            "[WARN] Guest name is unknown. The robot should ask the guest "
            "to repeat their name instead of continuing with a wrong name.",
            file=sys.stderr,
        )

    if record.favorite_drink.lower() == "unknown":
        print(
            "[WARN] Favorite drink is unknown. The robot should ask the "
            "guest to repeat it.",
            file=sys.stderr,
        )

    state.upsert_guest(record)
    state.save(state_path)

    print(json.dumps(asdict(record), indent=2))
    print(f"\nSaved state to: {state_path}", file=sys.stderr)
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    state = ReceptionistState.load(state_path)

    if not state.guests:
        raise SystemExit(
            "No registered guests in state. Run the 'register' command first."
        )

    engine = ER1Vision(
        model_id=args.model,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    image_path = Path(args.image)
    raw_group = engine.infer_json(
        image_path,
        group_prompt(state.guests),
    )
    group = normalize_group_result(raw_group)

    people, matched_guest_names = resolve_guest_matches(
        group["people"],
        state.guests,
        threshold=args.guest_match_threshold,
    )

    host = resolve_host(people)

    unseated_chairs, uncertain_chairs = resolve_unseated_chairs(
        group["chairs"],
        occupancy_threshold=args.chair_confidence_threshold,
    )

    if host["status"] == "recognized":
        host["name"] = args.host_name
        host["favorite_drink"] = args.host_drink

    result = {
        "registered_guests": [asdict(g) for g in state.guests],
        "matched_guest_names": matched_guest_names,
        "people": people,
        "host": host,
        "unseated_chairs": unseated_chairs,
        "uncertain_chairs": uncertain_chairs,
        "host_notifications": (
            host_notifications(args.host_name, state.guests)
            if host["status"] == "recognized"
            else []
        ),
        "recommended_next_action": (
            "offer_best_visible_free_seat"
            if host["status"] == "recognized" and unseated_chairs
            else (
                "reobserve_scene"
                if host["status"] != "recognized"
                else "search_for_free_seat"
            )
        ),
    }

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")

    if args.annotated:
        draw_debug_image(
            source=image_path,
            output=Path(args.annotated),
            people=people,
            host=host,
            unseated_chairs=unseated_chairs,
        )
        print(
            f"Annotated debug image: {args.annotated}",
            file=sys.stderr,
        )

    return 0


def command_show_state(args: argparse.Namespace) -> int:
    state = ReceptionistState.load(Path(args.state))
    print(
        json.dumps(
            {"guests": [asdict(g) for g in state.guests]},
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        default=True,
        dest="load_in_4bit",
        help=(
            "Disable the default bitsandbytes NF4 quantization."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=900,
        help="Maximum generated tokens for structured VLM output.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RoboCup@Home Receptionist prototype using UnifoLM-ER-Flow."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser(
        "register",
        help="Register one guest from entrance image + spoken introduction.",
    )
    register.add_argument("--image", required=True)
    register.add_argument(
        "--introduction",
        required=True,
        help=(
            'Example: "Hi, my name is Alice and my favorite drink is water."'
        ),
    )
    register.add_argument(
        "--name",
        default=None,
        help="Optional trusted name override.",
    )
    register.add_argument(
        "--drink",
        default=None,
        help="Optional trusted favorite-drink override.",
    )
    register.add_argument("--state", default=DEFAULT_STATE)
    add_model_args(register)
    register.set_defaults(func=command_register)

    analyze = sub.add_parser(
        "analyze",
        help=(
            "Analyze living-room group, identify host by elimination, "
            "and find unoccupied chairs."
        ),
    )
    analyze.add_argument("--image", required=True)
    analyze.add_argument(
        "--host-name",
        default="host",
        help="Known host name announced before the test.",
    )
    analyze.add_argument(
        "--host-drink",
        default="unknown",
        help="Known host favorite drink announced before the test.",
    )
    analyze.add_argument("--state", default=DEFAULT_STATE)
    analyze.add_argument(
        "--guest-match-threshold",
        type=float,
        default=0.65,
    )
    analyze.add_argument(
        "--chair-confidence-threshold",
        type=float,
        default=0.70,
    )
    analyze.add_argument(
        "--output",
        default=None,
        help="Optional JSON result file.",
    )
    analyze.add_argument(
        "--annotated",
        default=None,
        help="Optional output image with debug boxes.",
    )
    add_model_args(analyze)
    analyze.set_defaults(func=command_analyze)

    show = sub.add_parser(
        "show-state",
        help="Print current guest memory without loading ER-1.",
    )
    show.add_argument("--state", default=DEFAULT_STATE)
    show.set_defaults(func=command_show_state)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
