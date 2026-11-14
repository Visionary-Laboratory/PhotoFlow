import json
import math
import os
import random
import site
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

from agent_vlm import PhotographyAgent
from env_blender import BlenderPhotographyEnv


REFERENCE_OBJECT = None
USER_INSTRUCTION = "拍一张具有明确主体、空间层次和叙事感的照片"

MAX_ITERATIONS = 6
NUM_CANDIDATES = 4
SCOUT_VIEWS = 4

ASPECT_RATIO_OPTIONS = ["1:1", "4:3", "3:2", "16:9", "21:9", "9:16"]
INIT_ASPECT_RATIO = "16:9"
PREVIEW_MAX_PIXELS = 512 * 512

INIT_CAM_POS = None
INIT_LOOK_AT = None
INIT_FOCAL_LENGTH = 50.0
INIT_APERTURE = 2.8

PREVIEW_RESOLUTION = (512, 512)
FINAL_RESOLUTION = (1920, 1080)
PREVIEW_SAMPLES = 64
PREVIEW_WORKERS = 2
FINAL_SAMPLES = 128
FINAL_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output", "final_photo.jpg")
DEFAULT_METHOD_ID = "photoflow"
DEFAULT_BOOTSTRAP_MODE = "global_scene_scan_then_self_bootstrap"
DEFAULT_RANDOM_SEED = 42

EARLY_STOP_DEFAULTS = {
    "enabled": True,
    "min_rounds": 3,
    "patience": 2,
    "same_signature_patience": 2,
    "min_improvement": 0.012,
    "target_score": 0.68,
    "usable_score": 0.64,
    "max_explore_ratio": 0.45,
    "require_promising_region": False,
    "failure_patience": 3,
    "failure_score_threshold": 0.52,
}

TOTAL_SCORE_WEIGHTS = {
    "rule_m1": 0.10,
    "rule_m2": 0.10,
    "m1": 0.15,
    "m2": 0.15,
    "m3": 0.25,
    "m4": 0.25,
}


def resolve_render_engine_args(args):
    shared_engine = str(getattr(args, "render_engine", "") or "").strip()
    preview_engine = str(getattr(args, "preview_engine", "") or "").strip()
    final_engine = str(getattr(args, "final_engine", "") or "").strip()

    args.render_engine = shared_engine
    args.preview_engine = preview_engine or shared_engine or "CYCLES"
    args.final_engine = final_engine or shared_engine or "CYCLES"
    return args






def print_section(title: str, char: str = "-"):
    n = max(1, 72 - len(title))
    l = n // 2
    r = n - l
    print(f"\n{char * l} {title} {char * r}")


def list_scene_objects():
    print_section("Scene Objects")
    for i, obj in enumerate(sorted(bpy.data.objects, key=lambda o: (o.type, o.name)), 1):
        print(f"{i:3d}. {obj.name:<32s} type={obj.type}")
    print(f"\nTotal: {len(bpy.data.objects)}")


def normalize_ratio_string(value) -> str:
    txt = str(value or "").strip().replace(" ", "").replace("/", ":")
    if ":" not in txt:
        return "16:9"
    try:
        a, b = txt.split(":", 1)
        av = float(a)
        bv = float(b)
        if av <= 0 or bv <= 0:
            return "16:9"
        return f"{int(round(av))}:{int(round(bv))}"
    except Exception:
        return "16:9"


def normalize_ratio_options(values) -> list[str]:
    out = []
    seen = set()
    for item in values or []:
        ratio = normalize_ratio_string(item)
        if ratio not in seen:
            seen.add(ratio)
            out.append(ratio)
    return out or list(ASPECT_RATIO_OPTIONS)


def derive_resolution_by_ratio(aspect_ratio: str, target_pixels: int) -> tuple[int, int]:
    target_pixels = max(4, int(target_pixels))
    ratio = normalize_ratio_string(aspect_ratio)
    rw, rh = ratio.split(":", 1)
    ratio_val = float(rw) / float(rh)
    h = int((target_pixels / ratio_val) ** 0.5)
    w = int(h * ratio_val)
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    while w * h > target_pixels and w > 2 and h > 2:
        w -= 2
        h = int(w / ratio_val)
        h = max(2, h - (h % 2))
    return int(w), int(h)


def scene_scale(scene_info: dict) -> float:
    bbox = scene_info.get("scene_bbox", {})
    size = bbox.get("size", [6.0, 6.0, 4.0])
    return max(max(size), 1.0)


def blueprint_composition_rule(blueprint: dict) -> str:
    prefs = blueprint.get("composition_preferences") or []
    for item in prefs:
        if item in {"rule_of_thirds_left", "rule_of_thirds_right", "center"}:
            return item
    return "center"


def score_total(rec: dict) -> float:
    total = 0.0
    for key, weight in TOTAL_SCORE_WEIGHTS.items():
        total += weight * float(rec.get(key, 0.0))
    return total


def coarse_candidate_signature(candidate: dict | None) -> tuple | None:
    if not isinstance(candidate, dict):
        return None
    cam = candidate.get("camera_position") or []
    look = candidate.get("look_at") or []
    if len(cam) < 3 or len(look) < 3:
        return None
    return (
        tuple(round(float(v), 2) for v in cam[:3]),
        tuple(round(float(v), 2) for v in look[:3]),
        round(float(candidate.get("focal_length", 0.0)), 1),
        round(float(candidate.get("aperture", 0.0)), 2),
        normalize_ratio_string(candidate.get("aspect_ratio")),
    )


def resolve_initial_view_output(args, preview_log_dir: str) -> str:
    raw = str(getattr(args, "initial_view_output", "") or "").strip()
    if raw:
        return os.path.abspath(raw)
    if getattr(args, "save_initial_view", False):
        return os.path.join(preview_log_dir, "bootstrap_view.jpg")
    return ""


def build_early_stop_config(args) -> dict:
    return {
        "enabled": not bool(getattr(args, "disable_early_stop", False)),
        "min_rounds": max(1, int(getattr(args, "early_stop_min_rounds", EARLY_STOP_DEFAULTS["min_rounds"]))),
        "patience": max(1, int(getattr(args, "early_stop_patience", EARLY_STOP_DEFAULTS["patience"]))),
        "same_signature_patience": max(
            1,
            int(getattr(args, "early_stop_same_signature_patience", EARLY_STOP_DEFAULTS["same_signature_patience"])),
        ),
        "min_improvement": max(
            0.0,
            float(getattr(args, "early_stop_min_improvement", EARLY_STOP_DEFAULTS["min_improvement"])),
        ),
        "target_score": max(0.0, min(1.0, float(getattr(args, "early_stop_target_score", EARLY_STOP_DEFAULTS["target_score"])))),
        "usable_score": max(0.0, min(1.0, float(getattr(args, "early_stop_usable_score", EARLY_STOP_DEFAULTS["usable_score"])))),
        "max_explore_ratio": max(
            0.0,
            min(1.0, float(getattr(args, "early_stop_max_explore_ratio", EARLY_STOP_DEFAULTS["max_explore_ratio"]))),
        ),
        "require_promising_region": bool(
            getattr(args, "early_stop_require_promising_region", EARLY_STOP_DEFAULTS["require_promising_region"])
        ),
        "failure_patience": max(
            1,
            int(getattr(args, "early_stop_failure_patience", EARLY_STOP_DEFAULTS["failure_patience"])),
        ),
        "failure_score_threshold": max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        args,
                        "early_stop_failure_score_threshold",
                        EARLY_STOP_DEFAULTS["failure_score_threshold"],
                    )
                ),
            ),
        ),
    }


def update_early_stop_state(
    stop_state: dict,
    round_index: int,
    best: dict,
    current_region_summary: dict,
    review: dict,
    config: dict,
) -> tuple[bool, dict]:
    score = float(best.get("total_score", 0.0))
    signature = coarse_candidate_signature(best)
    prev_best_score = float(stop_state.get("global_best_score", -1.0))
    improved = score > prev_best_score + float(config["min_improvement"])

    if improved:
        stop_state["global_best_score"] = score
        stop_state["best_round"] = int(round_index)
        stop_state["stagnation_rounds"] = 0
    else:
        stop_state["global_best_score"] = max(prev_best_score, score)
        stop_state["stagnation_rounds"] = int(stop_state.get("stagnation_rounds", 0)) + 1

    if signature is not None and signature == stop_state.get("last_signature"):
        stop_state["same_signature_rounds"] = int(stop_state.get("same_signature_rounds", 0)) + 1
    else:
        stop_state["same_signature_rounds"] = 1 if signature is not None else 0
        stop_state["last_signature"] = signature

    if score >= prev_best_score:
        stop_state["last_best_score"] = score
        stop_state["last_best_round"] = int(round_index)

    snapshot = {
        "enabled": bool(config["enabled"]),
        "round_index": int(round_index),
        "best_total_score": round(score, 6),
        "global_best_score": round(float(stop_state.get("global_best_score", score)), 6),
        "best_round": int(stop_state.get("best_round", round_index)),
        "stagnation_rounds": int(stop_state.get("stagnation_rounds", 0)),
        "same_signature_rounds": int(stop_state.get("same_signature_rounds", 0)),
        "current_region_status": str(current_region_summary.get("current_region_status", "unknown")),
        "explore_ratio_next": round(float(review.get("explore_ratio_next", 1.0)), 6),
        "reason": "",
    }

    if not config["enabled"]:
        snapshot["reason"] = "disabled"
        return False, snapshot
    if int(round_index) < int(config["min_rounds"]):
        snapshot["reason"] = "below_min_rounds"
        return False, snapshot

    global_best = float(stop_state.get("global_best_score", 0.0))
    stagnation = int(stop_state.get("stagnation_rounds", 0))
    patience = int(config["patience"])
    failure_patience = int(config.get("failure_patience", patience + 1))
    same_signature = int(stop_state.get("same_signature_rounds", 0))
    explore_ratio = float(review.get("explore_ratio_next", 1.0))
    region_ready = (
        not bool(config.get("require_promising_region", False))
        or current_region_summary.get("current_region_status") == "promising"
    )
    explore_ready = explore_ratio <= float(config["max_explore_ratio"])
    stable_signature = same_signature >= int(config["same_signature_patience"])
    extended_plateau = stagnation >= patience + 1

    if stagnation < patience:
        snapshot["reason"] = "stagnation_patience_not_met"
        return False, snapshot

    if not region_ready:
        snapshot["reason"] = "current_region_not_promising"
        return False, snapshot

    if global_best >= float(config["target_score"]):
        if stable_signature or explore_ready or extended_plateau:
            snapshot["reason"] = (
                f"target_plateau_after_{snapshot['stagnation_rounds']}_rounds;"
                f" stable_signature={snapshot['same_signature_rounds']};"
                f" score={snapshot['global_best_score']:.3f};"
                f" region={snapshot['current_region_status']};"
                f" explore={snapshot['explore_ratio_next']:.2f}"
            )
            return True, snapshot
        snapshot["reason"] = "target_reached_but_still_exploring"
        return False, snapshot

    if global_best >= float(config.get("usable_score", config["target_score"])) and extended_plateau:
        snapshot["reason"] = (
            f"usable_plateau_after_{snapshot['stagnation_rounds']}_rounds;"
            f" score={snapshot['global_best_score']:.3f};"
            f" region={snapshot['current_region_status']};"
            f" explore={snapshot['explore_ratio_next']:.2f}"
        )
        return True, snapshot

    if global_best <= float(config.get("failure_score_threshold", 0.0)) and stagnation >= failure_patience:
        snapshot["reason"] = (
            f"low_score_plateau_after_{snapshot['stagnation_rounds']}_rounds;"
            f" score={snapshot['global_best_score']:.3f};"
            f" region={snapshot['current_region_status']};"
            f" explore={snapshot['explore_ratio_next']:.2f}"
        )
        return True, snapshot

    snapshot["reason"] = "target_score_not_reached"
    return False, snapshot


def select_primary_subject(blueprint: dict, scene_info: dict, reference_object: str | None) -> str | None:
    names = [o["name"] for o in scene_info.get("objects", []) if o.get("type") == "MESH"]
    if blueprint.get("primary_subject") in names:
        return blueprint["primary_subject"]
    if reference_object in names:
        return reference_object
    return names[0] if names else None


def height_bounds_for_preference(blueprint: dict, scene_info: dict) -> tuple[float, float]:
    bbox = scene_info.get("scene_bbox", {})
    z_min = float(bbox.get("min", [0.0, 0.0, 0.0])[2])
    z_max = float(bbox.get("max", [0.0, 0.0, 4.0])[2])
    pref = str(blueprint.get("camera_angle_preference", "unrestricted")).strip().lower()
    if pref == "low_angle":
        return z_min + 0.05, z_min + max(0.35 * (z_max - z_min), 0.8)
    if pref == "high_angle":
        return z_min + max(0.45 * (z_max - z_min), 1.2), z_max + 2.0
    if pref == "eye_level":
        mid = z_min + 0.45 * (z_max - z_min)
        return max(z_min + 0.5, mid - 0.5), mid + 0.8
    return z_min + 0.05, z_max + 2.0


def clamp_z(z_val: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], z_val))


def context_focus_point(env: BlenderPhotographyEnv, blueprint: dict, subject_center: list[float]) -> list[float]:
    weighted = []
    for item in blueprint.get("preferred_context_objects") or []:
        name = item.get("name")
        center = env.get_object_center(name) if name else None
        if center:
            weighted.append((center, float(item.get("weight", 0.5))))
    if not weighted:
        return list(subject_center)
    total_w = sum(max(w, 0.0) for _center, w in weighted) or 1.0
    ctx = [0.0, 0.0, 0.0]
    for center, weight in weighted:
        for i in range(3):
            ctx[i] += center[i] * max(weight, 0.0)
    ctx = [v / total_w for v in ctx]
    return [0.75 * subject_center[i] + 0.25 * ctx[i] for i in range(3)]


def blend_look_at(env: BlenderPhotographyEnv, cam: list[float], subject_center: list[float], context_focus: list[float], rule: str) -> list[float]:
    base = env.auto_frame_look_at(cam, subject_center, rule)
    return [round(0.82 * base[i] + 0.18 * context_focus[i], 6) for i in range(3)]


def orbit_camera(subject_center: list[float], base_cam: list[float], radius_scale: float, angle_deg: float, z_delta: float, z_bounds: tuple[float, float]) -> list[float]:
    dx = base_cam[0] - subject_center[0]
    dy = base_cam[1] - subject_center[1]
    radius = max(math.sqrt(dx * dx + dy * dy), 1.0) * radius_scale
    angle = math.atan2(dy, dx) + math.radians(angle_deg)
    return [
        round(subject_center[0] + radius * math.cos(angle), 6),
        round(subject_center[1] + radius * math.sin(angle), 6),
        round(clamp_z(base_cam[2] + z_delta, z_bounds), 6),
    ]


def camera_in_forbidden_zones(camera_position: list[float], forbidden_zones: list[dict]) -> bool:
    for zone in forbidden_zones or []:
        center = zone.get("center")
        radius = float(zone.get("radius", 0.0))
        if not center or radius <= 0:
            continue
        dist = math.sqrt(sum((camera_position[i] - center[i]) ** 2 for i in range(3)))
        if dist <= radius:
            return True
    return False


def vec_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def lerp_vec(a: list[float], b: list[float], t: float) -> list[float]:
    return [round(float(a[i]) * (1.0 - t) + float(b[i]) * t, 6) for i in range(3)]


def point_line_distance_xy(point: list[float], a: list[float], b: list[float]) -> tuple[float, float]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-8:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2), 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2), t


def region_cell_size(scene_info: dict) -> float:
    return max(scene_scale(scene_info) * 0.12, 0.9)


def make_region_key(camera_position: list[float], cell_size: float) -> tuple[int, int, int]:
    return tuple(int(round(float(camera_position[i]) / cell_size)) for i in range(3))


def region_center_from_key(region_key: tuple[int, int, int], cell_size: float) -> list[float]:
    return [round(region_key[i] * cell_size, 6) for i in range(3)]


def classify_region_entry(entry: dict) -> str:
    if entry.get("best_total_score", 0.0) >= 0.68 or entry.get("best_m4", 0.0) >= 0.70 or entry.get("promising_hits", 0) >= 1:
        return "promising"
    if entry.get("visit_count", 0) >= 2 and entry.get("poor_hits", 0) >= 2 and entry.get("best_total_score", 0.0) < 0.60 and entry.get("best_m4", 0.0) < 0.62:
        return "dead"
    if entry.get("visit_count", 0) >= 3 and entry.get("improvement_hits", 0) == 0 and entry.get("best_total_score", 0.0) < 0.62:
        return "dead"
    return "unknown"


def update_region_memory(region_memory: dict, candidate_records: list[dict], round_index: int, cell_size: float) -> None:
    for rec in candidate_records:
        cam = rec.get("camera_position") or [0.0, 0.0, 0.0]
        key = make_region_key(cam, cell_size)
        entry = region_memory.get(
            key,
            {
                "center": region_center_from_key(key, cell_size),
                "visit_count": 0,
                "best_total_score": 0.0,
                "best_m4": 0.0,
                "last_total_score": 0.0,
                "last_m4": 0.0,
                "last_round": 0,
                "poor_hits": 0,
                "promising_hits": 0,
                "improvement_hits": 0,
                "stagnation_hits": 0,
                "status": "unknown",
            },
        )
        total_score = float(rec.get("total_score", 0.0))
        m4 = float(rec.get("m4", 0.0))
        if total_score > entry["best_total_score"] + 0.015 or m4 > entry["best_m4"] + 0.02:
            entry["improvement_hits"] += 1
        else:
            entry["stagnation_hits"] += 1
        if total_score < 0.56 and m4 < 0.58:
            entry["poor_hits"] += 1
        if total_score >= 0.64 or m4 >= 0.66:
            entry["promising_hits"] += 1
        entry["visit_count"] += 1
        entry["best_total_score"] = max(entry["best_total_score"], total_score)
        entry["best_m4"] = max(entry["best_m4"], m4)
        entry["last_total_score"] = total_score
        entry["last_m4"] = m4
        entry["last_round"] = int(round_index)
        entry["status"] = classify_region_entry(entry)
        region_memory[key] = entry


def summarize_region_memory(region_memory: dict, current_region_key: tuple[int, int, int] | None) -> dict:
    counts = {"promising": 0, "dead": 0, "unknown": 0}
    for entry in region_memory.values():
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    current = region_memory.get(current_region_key, {}) if current_region_key is not None else {}
    return {
        "current_region_status": current.get("status", "unknown"),
        "current_region_visits": int(current.get("visit_count", 0)),
        "current_region_best_total_score": round(float(current.get("best_total_score", 0.0)), 4),
        "current_region_best_m4": round(float(current.get("best_m4", 0.0)), 4),
        "region_counts": counts,
    }


def region_memory_forbidden_zones(region_memory: dict, cell_size: float, limit: int = 4) -> list[dict]:
    dead_regions = [entry for entry in region_memory.values() if entry.get("status") == "dead"]
    dead_regions.sort(key=lambda x: (int(x.get("poor_hits", 0)), int(x.get("visit_count", 0)), -float(x.get("best_total_score", 0.0))), reverse=True)
    zones = []
    for entry in dead_regions[: max(0, int(limit))]:
        zones.append(
            {
                "center": [round(v, 6) for v in entry.get("center", [0.0, 0.0, 0.0])],
                "radius": round(cell_size * 1.15, 6),
                "reason": f"dead_region visits={entry.get('visit_count', 0)} poor_hits={entry.get('poor_hits', 0)}",
            }
        )
    return zones


def merge_forbidden_zones(*zone_groups: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for group in zone_groups:
        for zone in group or []:
            center = zone.get("center")
            radius = float(zone.get("radius", 0.0))
            if not center or radius <= 0:
                continue
            key = tuple(round(float(v), 2) for v in center[:3]) + (round(radius, 2),)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"center": [round(float(v), 6) for v in center[:3]], "radius": round(radius, 6), "reason": str(zone.get("reason", ""))})
    return merged


def pick_ratio_for_role(role: str, blueprint: dict, allowed_ratios: list[str], current_ratio: str) -> str:
    prefs = blueprint.get("composition_preferences") or []
    if "wide_environment" in prefs:
        for ratio in ("21:9", "16:9", current_ratio):
            if ratio in allowed_ratios:
                return ratio
    if "close_up" in prefs:
        for ratio in ("4:3", "3:2", "1:1", current_ratio):
            if ratio in allowed_ratios:
                return ratio
    if role == "global":
        for ratio in ("16:9", "21:9", current_ratio):
            if ratio in allowed_ratios:
                return ratio
    return current_ratio if current_ratio in allowed_ratios else allowed_ratios[0]


def dedup_candidates(candidates: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for cand in candidates:
        key = (
            tuple(round(float(v), 4) for v in cand.get("camera_position", [])[:3]),
            tuple(round(float(v), 4) for v in cand.get("look_at", [])[:3]),
            round(float(cand.get("focal_length", 0.0)), 1),
            round(float(cand.get("aperture", 0.0)), 2),
            normalize_ratio_string(cand.get("aspect_ratio")),
        )
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def build_seed_candidate(
    env,
    blueprint,
    obs,
    allowed_ratios,
    subject_center,
    context_focus,
    rule,
    role,
    angle_deg,
    radius_scale,
    z_delta,
    focal_mul,
    aperture_mul,
    thought,
    z_bounds,
):
    cam = orbit_camera(subject_center, obs["camera_position"], radius_scale, angle_deg, z_delta, z_bounds)
    cam = env.clamp_point_to_scene_bbox(cam)
    look = blend_look_at(env, cam, subject_center, context_focus, rule)
    ratio = pick_ratio_for_role(role, blueprint, allowed_ratios, normalize_ratio_string(obs.get("aspect_ratio")))
    return {
        "camera_position": cam,
        "look_at": look,
        "focal_length": round(max(16.0, min(160.0, float(obs["focal_length"]) * focal_mul)), 6),
        "aperture": round(max(1.2, min(16.0, float(obs["aperture"]) * aperture_mul)), 6),
        "aspect_ratio": ratio,
        "thought": thought,
        "source": role,
    }


def resolve_look_target_center(env, blueprint: dict, subject_center: list[float], context_focus: list[float]) -> list[float]:
    target_name = str(blueprint.get("look_toward_target", "")).strip()
    target_center = env.get_object_center(target_name) if target_name else None
    if target_center:
        return target_center
    return list(context_focus or subject_center)


def estimate_entrance_point(scene_info: dict, target_center: list[float], axis_preference: str) -> tuple[list[float], int]:
    bbox = scene_info.get("scene_bbox") or {}
    center = bbox.get("center", target_center)
    size = bbox.get("size", [6.0, 6.0, 4.0])
    mins = bbox.get("min", [target_center[0] - 3.0, target_center[1] - 3.0, 0.0])
    maxs = bbox.get("max", [target_center[0] + 3.0, target_center[1] + 3.0, 4.0])
    axis = 1 if size[1] >= size[0] else 0
    if axis_preference == "side_to_target":
        axis = 0 if axis == 1 else 1
    if axis == 0:
        x = mins[0] + 0.08 * size[0] if target_center[0] >= center[0] else maxs[0] - 0.08 * size[0]
        y = center[1]
    else:
        y = mins[1] + 0.08 * size[1] if target_center[1] >= center[1] else maxs[1] - 0.08 * size[1]
        x = center[0]
    z = mins[2] + min(max(size[2] * 0.18, 1.2), size[2] + 1.5)
    return [round(x, 6), round(y, 6), round(z, 6)], axis


def anchor_lens_and_ratio(blueprint: dict, allowed_ratios: list[str]) -> tuple[float, float, str]:
    prefs = blueprint.get("composition_preferences") or []
    axis_preference = str(blueprint.get("axis_preference", "toward_primary")).strip().lower()
    symmetry = str(blueprint.get("symmetry_preference", "moderate")).strip().lower()
    if "close_up" in prefs:
        focal = 55.0
        aperture = 3.2
    elif axis_preference == "entrance_to_target" or "wide_environment" in prefs:
        focal = 28.0
        aperture = 5.6
    elif symmetry == "strong":
        focal = 35.0
        aperture = 5.0
    else:
        focal = 40.0
        aperture = 4.0
    ratio = pick_ratio_for_role("global", blueprint, allowed_ratios, allowed_ratios[0])
    return focal, aperture, ratio


def build_global_anchor_bank(
    env,
    scene_info: dict,
    blueprint: dict,
    allowed_ratios: list[str],
    scout_views: list[dict] | None = None,
    max_anchors: int = 20,
) -> list[dict]:
    primary = blueprint["primary_subject"]
    subject_center = env.get_object_center(primary) or [0.0, 0.0, 0.0]
    context_focus = context_focus_point(env, blueprint, subject_center)
    look_target_center = resolve_look_target_center(env, blueprint, subject_center, context_focus)
    rule = blueprint_composition_rule(blueprint)
    z_bounds = height_bounds_for_preference(blueprint, scene_info)
    scene_size = scene_scale(scene_info)
    entrance_point, dominant_axis = estimate_entrance_point(scene_info, look_target_center, str(blueprint.get("axis_preference", "toward_primary")).strip().lower())
    side_vec = [0.0, 1.0, 0.0] if dominant_axis == 0 else [1.0, 0.0, 0.0]
    side_offset = max(scene_size * 0.12, 0.8)
    focal, aperture, ratio = anchor_lens_and_ratio(blueprint, allowed_ratios)
    anchors = []

    def add_anchor(cam: list[float], source: str, label: str, base_priority: float) -> None:
        cam = env.clamp_point_to_scene_bbox([cam[0], cam[1], clamp_z(cam[2], z_bounds)])
        look = blend_look_at(env, cam, look_target_center, context_focus, rule)
        line_dist, line_t = point_line_distance_xy(cam, entrance_point, look_target_center)
        target_dist = vec_distance(cam, look_target_center)
        priority = base_priority
        zone_pref = str(blueprint.get("camera_zone_preference", "unrestricted")).strip().lower()
        axis_pref = str(blueprint.get("axis_preference", "toward_primary")).strip().lower()
        symmetry = str(blueprint.get("symmetry_preference", "moderate")).strip().lower()
        prefs = blueprint.get("composition_preferences") or []
        if zone_pref == "entrance":
            priority += max(0.0, 1.0 - vec_distance(cam, entrance_point) / max(scene_size * 0.35, 1.0))
        elif zone_pref == "center_aisle":
            priority += max(0.0, 1.0 - line_dist / max(scene_size * 0.12, 0.8))
        elif zone_pref == "side_aisle":
            priority += min(line_dist / max(scene_size * 0.18, 0.8), 1.0)
        elif zone_pref == "altar_front":
            priority += max(0.0, 1.0 - target_dist / max(scene_size * 0.25, 1.2))
        if axis_pref == "entrance_to_target":
            priority += max(0.0, 1.0 - line_dist / max(scene_size * 0.10, 0.7))
            priority += max(0.0, 1.0 - abs(line_t - 0.20) / 0.35)
        elif axis_pref == "side_to_target":
            priority += min(line_dist / max(scene_size * 0.20, 0.9), 1.0)
        if symmetry == "strong":
            priority += max(0.0, 1.0 - line_dist / max(scene_size * 0.08, 0.6))
        elif symmetry == "none":
            priority += min(line_dist / max(scene_size * 0.16, 0.8), 1.0) * 0.4
        if "wide_environment" in prefs:
            priority += min(target_dist / max(scene_size * 0.45, 1.5), 1.0)
        elif "close_up" in prefs:
            priority += max(0.0, 1.0 - target_dist / max(scene_size * 0.25, 1.0))
        anchors.append(
            {
                "camera_position": [round(v, 6) for v in cam],
                "look_at": [round(v, 6) for v in look],
                "focal_length": round(focal, 6),
                "aperture": round(aperture, 6),
                "aspect_ratio": ratio,
                "thought": label,
                "source": source,
                "priority": round(priority, 6),
            }
        )

    for t in (0.05, 0.18, 0.32):
        add_anchor(lerp_vec(entrance_point, look_target_center, t), "global_anchor", f"entrance_axis_t{int(round(t * 100)):02d}", 1.4 - t)
    add_anchor([entrance_point[0] + side_vec[0] * side_offset, entrance_point[1] + side_vec[1] * side_offset, entrance_point[2]], "global_anchor", "entrance_side_positive", 0.95)
    add_anchor([entrance_point[0] - side_vec[0] * side_offset, entrance_point[1] - side_vec[1] * side_offset, entrance_point[2]], "global_anchor", "entrance_side_negative", 0.95)
    add_anchor([look_target_center[0], look_target_center[1], clamp_z(look_target_center[2] + max(scene_size * 0.20, 1.4), z_bounds)], "global_anchor", "high_overview", 0.8)

    context_names = [item.get("name") for item in blueprint.get("preferred_context_objects") or [] if item.get("name")]
    visibility_context = [name for name in context_names if name != primary][:1]
    vis_points = env.compute_visibility_candidates(
        primary_subject=primary,
        context_objects=visibility_context,
        grid_step=max(region_cell_size(scene_info), 0.9),
        camera_height_range=(z_bounds[0], min(z_bounds[1], z_bounds[0] + max(scene_size * 0.30, 2.2))),
        max_points=80,
    )
    for idx, cam in enumerate(vis_points[:24]):
        add_anchor(cam, "visibility_anchor", f"visibility_grid_{idx:02d}", 1.1)

    for item in scout_views or []:
        cam = item.get("camera_position")
        look = item.get("look_at")
        if not isinstance(cam, list) or len(cam) < 3 or not isinstance(look, list) or len(look) < 3:
            continue
        anchors.append(
            {
                "camera_position": [round(float(v), 6) for v in cam[:3]],
                "look_at": [round(float(v), 6) for v in look[:3]],
                "focal_length": round(max(focal, 30.0), 6),
                "aperture": round(max(aperture, 4.5), 6),
                "aspect_ratio": ratio,
                "thought": f"scout_relocate_{item.get('label', 'view')}",
                "source": "scout_anchor",
                "priority": 0.65,
            }
        )

    return dedup_candidates(sorted(anchors, key=lambda x: float(x.get("priority", 0.0)), reverse=True))[: max(1, int(max_anchors))]


def select_global_anchor_seeds(global_anchor_bank: list[dict], region_memory: dict, current_region_key: tuple[int, int, int] | None, cell_size: float, forbidden_zones: list[dict], count: int, search_mode: str) -> list[dict]:
    ranked = []
    for anchor in global_anchor_bank or []:
        cam = anchor.get("camera_position") or [0.0, 0.0, 0.0]
        if camera_in_forbidden_zones(cam, forbidden_zones):
            continue
        region_key = make_region_key(cam, cell_size)
        entry = region_memory.get(region_key, {})
        status = entry.get("status", "unknown")
        priority = float(anchor.get("priority", 0.0))
        if status == "dead":
            continue
        if status == "unknown":
            priority += 0.35
        if status == "promising":
            priority += 0.20
        if search_mode == "explore" and current_region_key is not None and region_key == current_region_key:
            priority -= 1.0
        ranked.append((priority, anchor))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [dict(item[1]) for item in ranked[: max(0, int(count))]]


def build_mixed_seed_candidates(
    env,
    scene_info,
    blueprint,
    obs,
    allowed_ratios,
    review: dict | None,
    num_candidates: int,
    region_memory: dict | None = None,
    global_anchor_bank: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    primary = blueprint["primary_subject"]
    subject_center = env.get_object_center(primary) or obs["look_at"]
    context_focus = context_focus_point(env, blueprint, subject_center)
    rule = blueprint_composition_rule(blueprint)
    z_bounds = height_bounds_for_preference(blueprint, scene_info)
    review = review or {}
    region_memory = region_memory or {}
    cell_size = region_cell_size(scene_info)
    current_region_key = make_region_key(obs["camera_position"], cell_size)
    region_summary = summarize_region_memory(region_memory, current_region_key)
    current_region_status = region_summary["current_region_status"]
    search_mode = "exploit" if current_region_status == "promising" and float(review.get("explore_ratio_next", 0.35)) <= 0.45 else "explore"
    step_scale = float(review.get("step_scale", 1.0))
    explore_ratio = float(review.get("explore_ratio_next", 0.35))
    memory_forbidden = region_memory_forbidden_zones(region_memory, cell_size)
    forbidden_zones = merge_forbidden_zones(review.get("forbidden_zones") or [], memory_forbidden)

    if search_mode == "exploit":
        local_count = max(2, int(math.ceil(num_candidates * 0.6)))
        geometry_count = max(1, int(math.ceil(num_candidates * 0.35)))
        global_count = max(1, int(math.ceil(num_candidates * max(explore_ratio, 0.2))))
    else:
        local_count = 0 if current_region_status == "dead" else 1
        geometry_count = max(1, int(math.ceil(num_candidates * 0.25)))
        global_count = max(2, int(math.ceil(num_candidates * max(explore_ratio, 0.6))))

    local_patterns = [
        (0, 0.88, -0.10 * step_scale, 1.06, 0.95, "tight_refine_center"),
        (12, 0.98, -0.03 * step_scale, 1.00, 1.00, "tight_refine_left"),
        (-12, 0.98, 0.04 * step_scale, 0.97, 1.05, "tight_refine_right"),
    ]
    geometry_patterns = [
        (38, 1.08, -0.08 * step_scale, 1.00, 0.96, "geometry_probe_positive"),
        (-38, 1.08, -0.04 * step_scale, 1.00, 0.96, "geometry_probe_negative"),
        (72, 1.18, 0.06 * step_scale, 0.92, 1.08, "geometry_probe_wide"),
    ]

    seeds = []
    for cand in (review.get("candidates") or [])[:2]:
        item = dict(cand)
        item["source"] = item.get("source", "review_seed")
        seeds.append(item)

    if local_count > 0:
        for idx in range(local_count):
            p = local_patterns[idx % len(local_patterns)]
            seeds.append(build_seed_candidate(env, blueprint, obs, allowed_ratios, subject_center, context_focus, rule, "local", *p, z_bounds))
    for idx in range(geometry_count):
        p = geometry_patterns[idx % len(geometry_patterns)]
        seeds.append(build_seed_candidate(env, blueprint, obs, allowed_ratios, subject_center, context_focus, rule, "geometry", *p, z_bounds))
    for anchor in select_global_anchor_seeds(global_anchor_bank or [], region_memory, current_region_key, cell_size, forbidden_zones, global_count, search_mode):
        anchor["source"] = anchor.get("source", "global_anchor")
        seeds.append(anchor)

    filtered = []
    for seed in dedup_candidates(seeds):
        if camera_in_forbidden_zones(seed["camera_position"], forbidden_zones):
            continue
        if env.is_target_visible_from(seed["camera_position"], primary) or seed.get("source") in {"global_anchor", "visibility_anchor", "scout_anchor", "review_seed"}:
            filtered.append(seed)

    if not filtered:
        filtered = [seed for seed in dedup_candidates(seeds) if not camera_in_forbidden_zones(seed["camera_position"], forbidden_zones)]
    if not filtered:
        filtered = dedup_candidates(seeds)

    plan = {
        "search_mode": search_mode,
        "step_scale": round(step_scale, 4),
        "explore_ratio_next": round(explore_ratio, 4),
        "preferred_motion": review.get("preferred_motion", "leave dead regions and only exploit promising regions"),
        "next_strategy": review.get("next_strategy", "region-memory guided mixed seed search"),
        "failure_tags": review.get("failure_tags", []),
        "forbidden_zones": forbidden_zones,
        "region_memory_summary": region_summary,
        "seed_pool_counts": {"local": local_count, "geometry": geometry_count, "global": global_count, "review": len(review.get("candidates") or [])},
        "global_anchor_count": len(global_anchor_bank or []),
    }
    return filtered, plan




def parse_cli_args():
    import argparse

    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description="PhotoFlow")
    parser.add_argument("--reference", type=str, default=REFERENCE_OBJECT)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=USER_INSTRUCTION)
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--num-candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--scout-views", type=int, default=SCOUT_VIEWS)
    parser.add_argument("--aspect-options", type=str, nargs="+", default=list(ASPECT_RATIO_OPTIONS))
    parser.add_argument("--init-aspect-ratio", type=str, default=INIT_ASPECT_RATIO)
    parser.add_argument("--preview-max-pixels", type=int, default=PREVIEW_MAX_PIXELS)
    parser.add_argument("--preview-samples", type=int, default=PREVIEW_SAMPLES)
    parser.add_argument("--preview-workers", type=int, default=PREVIEW_WORKERS)
    parser.add_argument("--render-engine", type=str, default=None)
    parser.add_argument("--preview-engine", type=str, default=None)
    parser.add_argument("--cam-pos", type=float, nargs=3, default=INIT_CAM_POS, metavar=("X", "Y", "Z"))
    parser.add_argument("--look-at", type=float, nargs=3, default=INIT_LOOK_AT, metavar=("X", "Y", "Z"))
    parser.add_argument("--focal-length", type=float, default=INIT_FOCAL_LENGTH)
    parser.add_argument("--aperture", type=float, default=INIT_APERTURE)
    parser.add_argument("--preview-res", type=int, nargs=2, default=list(PREVIEW_RESOLUTION), metavar=("W", "H"))
    parser.add_argument("--final-res", type=int, nargs=2, default=list(FINAL_RESOLUTION), metavar=("W", "H"))
    parser.add_argument("--final-samples", type=int, default=FINAL_SAMPLES)
    parser.add_argument("--final-engine", type=str, default=None)
    parser.add_argument("--output", type=str, default=FINAL_OUTPUT_PATH)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--print-llm-raw", action="store_true")
    parser.add_argument("--task-id", type=str, default="")
    parser.add_argument("--method-id", type=str, default=DEFAULT_METHOD_ID)
    parser.add_argument("--bootstrap-mode", type=str, default=DEFAULT_BOOTSTRAP_MODE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--save-initial-view", action="store_true")
    parser.add_argument("--initial-view-output", type=str, default="")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--early-stop-min-rounds", type=int, default=EARLY_STOP_DEFAULTS["min_rounds"])
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_DEFAULTS["patience"])
    parser.add_argument(
        "--early-stop-same-signature-patience",
        type=int,
        default=EARLY_STOP_DEFAULTS["same_signature_patience"],
    )
    parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=EARLY_STOP_DEFAULTS["min_improvement"],
    )
    parser.add_argument("--early-stop-target-score", type=float, default=EARLY_STOP_DEFAULTS["target_score"])
    parser.add_argument("--early-stop-usable-score", type=float, default=EARLY_STOP_DEFAULTS["usable_score"])
    parser.add_argument(
        "--early-stop-max-explore-ratio",
        type=float,
        default=EARLY_STOP_DEFAULTS["max_explore_ratio"],
    )
    parser.add_argument("--early-stop-require-promising-region", action="store_true")
    parser.add_argument("--early-stop-failure-patience", type=int, default=EARLY_STOP_DEFAULTS["failure_patience"])
    parser.add_argument(
        "--early-stop-failure-score-threshold",
        type=float,
        default=EARLY_STOP_DEFAULTS["failure_score_threshold"],
    )
    parser.add_argument("--list-objects", action="store_true")
    args = parser.parse_args(argv)
    if args.target and not args.reference:
        args.reference = args.target
    return resolve_render_engine_args(args)


def print_banner():
    print()
    print("=" * 72)
    print(" PhotoFlow ".center(72))
    print(" region-memory + parallel-preview + structured-explore ".center(72))
    print("=" * 72)
    print()


def choose_diverse_ratio(current_ratio: str, allowed_ratios: list[str], blueprint: dict, used_ratios: list[str] | None = None) -> str:
    current_ratio = normalize_ratio_string(current_ratio)
    used = {normalize_ratio_string(x) for x in (used_ratios or [])}
    prefs = blueprint.get("composition_preferences") or []
    candidates = []
    if "wide_environment" in prefs:
        candidates.extend(["21:9", "16:9", "3:2", "4:3"])
    elif "close_up" in prefs:
        candidates.extend(["4:3", "3:2", "1:1", "16:9"])
    else:
        candidates.extend(["16:9", "3:2", "4:3", "21:9", "1:1"])
    for ratio in candidates + list(allowed_ratios):
        norm = normalize_ratio_string(ratio)
        if norm in allowed_ratios and norm != current_ratio and norm not in used:
            return norm
    for ratio in allowed_ratios:
        norm = normalize_ratio_string(ratio)
        if norm != current_ratio:
            return norm
    return current_ratio if current_ratio in allowed_ratios else allowed_ratios[0]


def choose_high_explore_seed(
    global_anchor_bank: list[dict],
    region_memory: dict,
    current_obs: dict,
    allowed_ratios: list[str],
    forbidden_zones: list[dict],
    cell_size: float,
    blueprint: dict,
) -> dict | None:
    current_cam = current_obs.get("camera_position") or [0.0, 0.0, 0.0]
    current_ratio = normalize_ratio_string(current_obs.get("aspect_ratio"))
    ranked = []
    for anchor in global_anchor_bank or []:
        cam = anchor.get("camera_position") or [0.0, 0.0, 0.0]
        if camera_in_forbidden_zones(cam, forbidden_zones):
            continue
        region_key = make_region_key(cam, cell_size)
        entry = region_memory.get(region_key, {})
        status = entry.get("status", "unknown")
        if status == "dead":
            continue
        distance_gain = vec_distance(cam, current_cam)
        visit_penalty = float(entry.get("visit_count", 0))
        unknown_bonus = 1.2 if status == "unknown" else 0.25
        promising_penalty = 0.4 if status == "promising" else 0.0
        ratio = choose_diverse_ratio(current_ratio, allowed_ratios, blueprint, [anchor.get("aspect_ratio", current_ratio)])
        score = (
            float(anchor.get("priority", 0.0))
            + unknown_bonus
            + min(distance_gain / max(cell_size * 2.0, 1.0), 2.0)
            - 0.35 * visit_penalty
            - promising_penalty
        )
        ranked.append((score, anchor, ratio))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None
    _, anchor, ratio = ranked[0]
    seed = dict(anchor)
    seed["aspect_ratio"] = ratio
    seed["source"] = "high_explore"
    seed["thought"] = (
        "forced_high_explore: move to a low-visit anchor outside dead zones, "
        "far from the current basin, and intentionally test a different aspect ratio"
    )
    return seed


def build_mixed_seed_candidates_v44(
    env,
    scene_info,
    blueprint,
    obs,
    allowed_ratios,
    review: dict | None,
    num_candidates: int,
    region_memory: dict | None = None,
    global_anchor_bank: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    seeds, plan = build_mixed_seed_candidates(
        env=env,
        scene_info=scene_info,
        blueprint=blueprint,
        obs=obs,
        allowed_ratios=allowed_ratios,
        review=review,
        num_candidates=num_candidates,
        region_memory=region_memory,
        global_anchor_bank=global_anchor_bank,
    )
    review = review or {}
    region_memory = region_memory or {}
    cell_size = region_cell_size(scene_info)
    forbidden_zones = merge_forbidden_zones(
        review.get("forbidden_zones") or [],
        region_memory_forbidden_zones(region_memory, cell_size),
    )
    high_explore = choose_high_explore_seed(
        global_anchor_bank=global_anchor_bank or [],
        region_memory=region_memory,
        current_obs=obs,
        allowed_ratios=allowed_ratios,
        forbidden_zones=forbidden_zones,
        cell_size=cell_size,
        blueprint=blueprint,
    )
    merged = list(seeds)
    if high_explore is not None:
        merged.insert(0, high_explore)
    merged = dedup_candidates(merged)
    if len(merged) > max(1, int(num_candidates) + 2):
        keep = []
        keep_sources = {"high_explore", "review_seed"}
        for item in merged:
            if item.get("source") in keep_sources:
                keep.append(item)
        for item in merged:
            if len(keep) >= max(1, int(num_candidates) + 2):
                break
            if item not in keep:
                keep.append(item)
        merged = keep
    plan = dict(plan)
    plan["forced_high_explore"] = high_explore is not None
    plan["high_explore_strategy"] = (
        "unknown-region global anchor with distance, low-visit, dead-zone avoidance, and aspect-ratio diversification"
        if high_explore is not None
        else "unavailable"
    )
    plan["seed_pool_counts"] = dict(plan.get("seed_pool_counts", {}))
    plan["seed_pool_counts"]["high_explore"] = 1 if high_explore is not None else 0
    return merged, plan


def can_use_external_preview_parallel() -> bool:
    return bool(bpy.data.filepath and os.path.isfile(bpy.data.filepath) and bpy.app.binary_path and os.path.isfile(bpy.app.binary_path))


def render_candidate_preview_external(
    blend_file: str,
    script_path: str,
    reference_object: str | None,
    action: dict,
    output_path: str,
    preview_max_pixels: int,
    preview_samples: int,
    preview_engine: str,
) -> dict:
    cmd = [
        bpy.app.binary_path,
        "-b",
        blend_file,
        "-P",
        script_path,
        "--",
        "--reference",
        str(reference_object or ""),
        "--cam-pos",
        str(action["camera_position"][0]),
        str(action["camera_position"][1]),
        str(action["camera_position"][2]),
        "--look-at",
        str(action["look_at"][0]),
        str(action["look_at"][1]),
        str(action["look_at"][2]),
        "--focal-length",
        str(action["focal_length"]),
        "--aperture",
        str(action["aperture"]),
        "--aspect-ratio",
        str(action["aspect_ratio"]),
        "--preview-max-pixels",
        str(int(preview_max_pixels)),
        "--preview-samples",
        str(int(preview_samples)),
        "--preview-engine",
        str(preview_engine),
        "--preview",
        output_path,
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "output_path": output_path,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed": round(time.time() - started, 4),
    }


def render_candidate_previews_parallel(
    env,
    preview_jobs: list[dict],
    preview_workers: int,
    preview_max_pixels: int,
    preview_samples: int,
    preview_engine: str,
    reference_object: str | None,
) -> tuple[list[dict], dict]:
    if not preview_jobs:
        return [], {"mode": "none", "workers": 0, "fallback_reason": ""}
    worker_count = max(1, int(preview_workers))
    blend_file = bpy.data.filepath
    script_path = os.path.join(SCRIPT_DIR, "env_blender.py")
    if worker_count <= 1 or not can_use_external_preview_parallel():
        reason = "preview_workers<=1 or blend/binary unavailable"
        results = []
        for job in preview_jobs:
            env.step(job["action"])
            env.render_preview(job["output_path"], engine=preview_engine)
            results.append({"output_path": job["output_path"], "returncode": 0, "stdout": "", "stderr": "", "elapsed": 0.0})
        return results, {"mode": "serial_fallback", "workers": 1, "fallback_reason": reason}

    results = [None] * len(preview_jobs)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for idx, job in enumerate(preview_jobs):
            fut = pool.submit(
                render_candidate_preview_external,
                blend_file,
                script_path,
                reference_object,
                job["action"],
                job["output_path"],
                preview_max_pixels,
                preview_samples,
                preview_engine,
            )
            futures[fut] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {
                    "output_path": preview_jobs[idx]["output_path"],
                    "returncode": 1,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}",
                    "elapsed": 0.0,
                }

    failed = [i for i, item in enumerate(results) if item is None or int(item.get("returncode", 1)) != 0 or not os.path.isfile(preview_jobs[i]["output_path"])]
    for idx in failed:
        env.step(preview_jobs[idx]["action"])
        env.render_preview(preview_jobs[idx]["output_path"], engine=preview_engine)
        results[idx] = {
            "output_path": preview_jobs[idx]["output_path"],
            "returncode": 0,
            "stdout": results[idx].get("stdout", "") if results[idx] else "",
            "stderr": (results[idx].get("stderr", "") + " | serial_retry") if results[idx] else "serial_retry",
            "elapsed": results[idx].get("elapsed", 0.0) if results[idx] else 0.0,
        }
    return results, {"mode": "external_parallel", "workers": worker_count, "fallback_reason": ""}


def main():
    print_banner()
    args = parse_cli_args()
    if args.list_objects:
        list_scene_objects()
        return

    random.seed(int(args.random_seed))

    allowed_ratios = normalize_ratio_options(args.aspect_options)
    init_ratio = normalize_ratio_string(args.init_aspect_ratio)
    if init_ratio not in allowed_ratios:
        init_ratio = allowed_ratios[0]
    preview_max_pixels = max(16 * 16, int(args.preview_max_pixels))
    preview_samples = max(1, min(int(args.preview_samples), 64))
    preview_workers = max(1, int(args.preview_workers))
    final_samples = max(1, min(int(args.final_samples), 256))
    early_stop_config = build_early_stop_config(args)

    print_section("Config", "=")
    print(f"task id: {args.task_id or '(unset)'}")
    print(f"method id: {args.method_id}")
    print(f"bootstrap mode: {args.bootstrap_mode}")
    print(f"random seed: {args.random_seed}")
    print(f"instruction: {args.instruction}")
    print(f"reference: {args.reference}")
    print(f"iterations: {args.max_iter}")
    print(f"candidates/iter: {args.num_candidates}")
    print(f"scout views: {args.scout_views}")
    print(f"allowed aspect ratios: {allowed_ratios}")
    print(f"initial aspect ratio: {init_ratio}")
    print(f"preview samples: {preview_samples}")
    print(f"preview workers: {preview_workers}")
    print(f"preview engine: {args.preview_engine}")
    print(f"final engine: {args.final_engine}")
    print(
        "early stop: "
        f"enabled={early_stop_config['enabled']} "
        f"min_rounds={early_stop_config['min_rounds']} "
        f"patience={early_stop_config['patience']} "
        f"same_sig={early_stop_config['same_signature_patience']} "
        f"target={early_stop_config['target_score']:.2f} "
        f"usable={early_stop_config['usable_score']:.2f} "
        f"max_explore={early_stop_config['max_explore_ratio']:.2f} "
        f"failure_patience={early_stop_config['failure_patience']}"
    )

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_log_dir = os.path.join(output_dir, "preview_logs", f"{output_name}_{run_stamp}")
    scout_dir = os.path.join(preview_log_dir, "scout_views")
    os.makedirs(preview_log_dir, exist_ok=True)
    initial_view_output = resolve_initial_view_output(args, preview_log_dir)
    if initial_view_output:
        os.makedirs(os.path.dirname(initial_view_output), exist_ok=True)

    env = BlenderPhotographyEnv(
        reference_object=args.reference,
        camera_position=args.cam_pos,
        look_at=args.look_at,
        focal_length=args.focal_length,
        aperture=args.aperture,
        preview_resolution=tuple(args.preview_res),
        aspect_ratio=init_ratio,
        preview_max_pixels=preview_max_pixels,
        preview_samples=preview_samples,
        preview_engine=args.preview_engine,
        final_engine=args.final_engine,
    )

    scene_info = env.get_scene_info()
    scene_info_text = env.get_scene_info_text()
    scene_topology_text = env.get_scene_topology_text()
    scout_views = env.render_scout_views(scout_dir, focus_object=args.reference, num_orbit_views=int(args.scout_views))
    scene_objects = [o["name"] for o in scene_info.get("objects", []) if o.get("type") == "MESH"]

    agent = PhotographyAgent(
        user_instruction=args.instruction,
        scene_info=scene_info_text,
        scene_topology=scene_topology_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        random_seed=int(args.random_seed),
        print_raw_llm=args.print_llm_raw,
        verbose=2,
    )

    blueprint = agent.parse_soft_blueprint(scene_objects=scene_objects, reference_object=args.reference)
    blueprint["primary_subject"] = select_primary_subject(blueprint, scene_info, args.reference)
    if not blueprint["primary_subject"]:
        raise RuntimeError("No valid primary subject found for PhotoFlow pipeline.")

    global_anchor_bank = build_global_anchor_bank(
        env=env,
        scene_info=scene_info,
        blueprint=blueprint,
        allowed_ratios=allowed_ratios,
        scout_views=scout_views,
    )

    print_section("Soft Blueprint", "=")
    print(json.dumps(blueprint, ensure_ascii=False, indent=2))
    print_section("Global Anchors", "=")
    print(f"anchors: {len(global_anchor_bank)}")
    for idx, anchor in enumerate(global_anchor_bank[:6], 1):
        print(f"  #{idx} source={anchor['source']} priority={anchor.get('priority', 0.0):.3f} thought={anchor.get('thought', '')}")

    obs = env.reset()
    initial_obs = dict(obs)
    if initial_view_output:
        env.render_preview(initial_view_output, engine=args.preview_engine)
    region_memory = {}
    round_logs = []
    start_t = time.time()
    last_review = None
    global_best_record = None
    stop_state = {
        "global_best_score": -1.0,
        "best_round": 0,
        "stagnation_rounds": 0,
        "same_signature_rounds": 0,
        "last_signature": None,
        "last_best_score": -1.0,
        "last_best_round": 0,
    }
    termination = {
        "terminated_early": False,
        "stop_iteration": None,
        "stop_reason": "max_iter_exhausted",
        "early_stop_state": {},
    }

    print_section("Iteration", "=")
    for it in range(1, int(args.max_iter) + 1):
        print(f"\n[Round {it}/{args.max_iter}]")
        context_img = os.path.join(preview_log_dir, f"iter_{it:03d}_context.jpg")
        env.render_preview(context_img, engine=args.preview_engine)

        seed_candidates, proposal_plan = build_mixed_seed_candidates_v44(
            env=env,
            scene_info=scene_info,
            blueprint=blueprint,
            obs=obs,
            allowed_ratios=allowed_ratios,
            review=last_review,
            num_candidates=int(args.num_candidates),
            region_memory=region_memory,
            global_anchor_bank=global_anchor_bank,
        )
        print(f"  seed pool: {len(seed_candidates)} | mode={proposal_plan['search_mode']} | plan={proposal_plan['seed_pool_counts']}")
        if proposal_plan.get("forced_high_explore"):
            print(f"  high explore: {proposal_plan.get('high_explore_strategy', '')}")

        candidates = agent.propose_candidates(
            image_path=context_img,
            current_params=obs,
            num_candidates=int(args.num_candidates),
            allowed_aspect_ratios=allowed_ratios,
            soft_blueprint=blueprint,
            proposal_plan=proposal_plan,
            seed_candidates=seed_candidates,
            scout_views=scout_views,
        )
        print(f"  proposals: {len(candidates)}")

        rule_name = blueprint_composition_rule(blueprint)
        cell_size = region_cell_size(scene_info)
        prepared_candidates = []
        preview_jobs = []
        for idx, cand in enumerate(candidates, 1):
            action = {
                "camera_position": cand["camera_position"],
                "look_at": cand["look_at"],
                "focal_length": cand["focal_length"],
                "aperture": cand["aperture"],
                "aspect_ratio": normalize_ratio_string(cand.get("aspect_ratio") or obs.get("aspect_ratio", init_ratio)),
            }
            cand_img = os.path.join(preview_log_dir, f"iter_{it:03d}_cand_{idx:02d}.jpg")
            prepared_candidates.append({"index": idx, "candidate": cand, "action": action, "image_path": cand_img})
            preview_jobs.append({"action": action, "output_path": cand_img})

        preview_results, preview_runtime = render_candidate_previews_parallel(
            env=env,
            preview_jobs=preview_jobs,
            preview_workers=preview_workers,
            preview_max_pixels=preview_max_pixels,
            preview_samples=preview_samples,
            preview_engine=args.preview_engine,
            reference_object=args.reference,
        )
        print(f"  preview render: mode={preview_runtime['mode']} workers={preview_runtime['workers']}")

        candidate_records = []
        incumbent = None
        for item, render_meta in zip(prepared_candidates, preview_results):
            idx = item["index"]
            cand = item["candidate"]
            action = item["action"]
            cand_img = item["image_path"]

            rule_scores = env.evaluate_rule_scores(
                camera_position=action["camera_position"],
                look_at=action["look_at"],
                primary_subject=blueprint["primary_subject"],
                composition_rule=rule_name,
            )
            ai_scores = agent.score_image_4d(cand_img, current_params=action)
            region_key = make_region_key(action["camera_position"], cell_size)
            region_state = region_memory.get(region_key, {"status": "unknown"})

            rec = {
                "candidate_index": idx,
                "source": cand.get("source", "llm"),
                "camera_position": [round(v, 6) for v in action["camera_position"]],
                "look_at": [round(v, 6) for v in action["look_at"]],
                "focal_length": round(float(action["focal_length"]), 6),
                "aperture": round(float(action["aperture"]), 6),
                "aspect_ratio": action["aspect_ratio"],
                "image_path": cand_img,
                "candidate_thought": cand.get("thought", ""),
                "region_key": list(region_key),
                "region_status_before": region_state.get("status", "unknown"),
                "rule_m1": float(rule_scores.get("m1", 0.0)),
                "rule_m2": float(rule_scores.get("m2", 0.0)),
                "rule_uv": rule_scores.get("uv"),
                "rule_target_uv": rule_scores.get("target_uv"),
                "m1": float(ai_scores["m1"]),
                "m2": float(ai_scores["m2"]),
                "m3": float(ai_scores["m3"]),
                "m4": float(ai_scores["m4"]),
                "score_reasoning": ai_scores.get("reasoning", ""),
                "score_summary": ai_scores.get("summary", ""),
                "preview_render_mode": preview_runtime["mode"],
                "preview_render_elapsed": float(render_meta.get("elapsed", 0.0)),
                "preview_render_stderr": render_meta.get("stderr", ""),
            }
            rec["total_score"] = round(score_total(rec), 6)

            if incumbent is None:
                rec["pairwise"] = {"winner": "B", "reasoning": "first-candidate-sets-incumbent", "summary": "seed-first"}
                incumbent = rec
                print(f"  cand#{idx} seed incumbent source={rec['source']} total={rec['total_score']:.3f}")
            else:
                compare = agent.compare_images_pairwise_4d(
                    image_a_path=incumbent["image_path"],
                    image_b_path=cand_img,
                    params_a=incumbent,
                    params_b=rec,
                )
                rec["pairwise"] = compare
                if compare.get("winner") == "B":
                    incumbent = rec
                    print(f"  cand#{idx} wins pairwise source={rec['source']} total={rec['total_score']:.3f}")
                else:
                    print(f"  cand#{idx} loses pairwise source={rec['source']} total={rec['total_score']:.3f}")

            candidate_records.append(rec)

        best = incumbent or max(candidate_records, key=lambda x: x["total_score"])
        if global_best_record is None or float(best.get("total_score", 0.0)) > float(global_best_record.get("total_score", -1.0)):
            global_best_record = json.loads(json.dumps(best, ensure_ascii=False))
        update_region_memory(region_memory, candidate_records, it, cell_size)

        env.step(
            {
                "camera_position": best["camera_position"],
                "look_at": best["look_at"],
                "focal_length": best["focal_length"],
                "aperture": best["aperture"],
                "aspect_ratio": best["aspect_ratio"],
            }
        )
        obs = env.get_observation()
        last_review = agent.review_round(
            round_index=it,
            candidate_records=candidate_records,
            current_params=obs,
            num_candidates=int(args.num_candidates),
            allowed_aspect_ratios=allowed_ratios,
            weights=TOTAL_SCORE_WEIGHTS,
        )
        memory_zones = region_memory_forbidden_zones(region_memory, cell_size)
        last_review["forbidden_zones"] = merge_forbidden_zones(last_review.get("forbidden_zones") or [], memory_zones)
        last_review["explore_ratio_next"] = max(float(last_review.get("explore_ratio_next", 0.35)), 0.30)

        current_region_summary = summarize_region_memory(region_memory, make_region_key(best["camera_position"], cell_size))
        print(f"  [best] cand#{best['candidate_index']} source={best['source']} total={best['total_score']:.3f}")
        print(
            f"  [review] mode={proposal_plan['search_mode']} step_scale={last_review['step_scale']:.2f} "
            f"explore={last_review['explore_ratio_next']:.2f} region={current_region_summary['current_region_status']}"
        )

        round_summary = {
            "iteration": it,
            "best_candidate_index": best["candidate_index"],
            "best_camera_position": best["camera_position"],
            "best_look_at": best["look_at"],
            "best_focal_length": best["focal_length"],
            "best_aperture": best["aperture"],
            "best_aspect_ratio": best["aspect_ratio"],
            "best_total_score": best["total_score"],
            "search_mode": proposal_plan["search_mode"],
            "current_region_status": current_region_summary["current_region_status"],
            "review_strategy": last_review["next_strategy"],
            "review_step_scale": last_review["step_scale"],
            "review_explore_ratio_next": last_review["explore_ratio_next"],
        }
        agent.push_round_summary(round_summary)
        round_logs.append(
            {
                "iteration": it,
                "context_image": context_img,
                "proposal_plan": proposal_plan,
                "seed_candidates": seed_candidates,
                "candidates": candidate_records,
                "best": best,
                "global_best_so_far": global_best_record,
                "review": last_review,
                "region_memory_summary": current_region_summary,
                "preview_runtime": preview_runtime,
            }
        )

        should_stop, stop_snapshot = update_early_stop_state(
            stop_state=stop_state,
            round_index=it,
            best=best,
            current_region_summary=current_region_summary,
            review=last_review,
            config=early_stop_config,
        )
        round_logs[-1]["early_stop_state"] = stop_snapshot
        if should_stop:
            termination = {
                "terminated_early": True,
                "stop_iteration": int(it),
                "stop_reason": stop_snapshot["reason"],
                "early_stop_state": stop_snapshot,
            }
            print(f"  [early-stop] {stop_snapshot['reason']}")
            break

    print_section("Final Render", "=")
    final_best_record = global_best_record or (round_logs[-1]["best"] if round_logs else None)
    if final_best_record:
        env.step(
            {
                "camera_position": final_best_record["camera_position"],
                "look_at": final_best_record["look_at"],
                "focal_length": final_best_record["focal_length"],
                "aperture": final_best_record["aperture"],
                "aspect_ratio": final_best_record["aspect_ratio"],
            }
        )
        obs = env.get_observation()
    final_pick_img = final_best_record["image_path"] if final_best_record else os.path.join(preview_log_dir, "iter_000_context.jpg")
    if not round_logs:
        env.render_preview(final_pick_img, engine=args.preview_engine)
    final_aspect_pick = agent.choose_final_aspect_ratio(final_pick_img, obs, allowed_ratios)
    final_aspect_ratio = final_aspect_pick["aspect_ratio"]
    env.step({"aspect_ratio": final_aspect_ratio})

    target_pixels = int(args.final_res[0]) * int(args.final_res[1])
    final_resolution = derive_resolution_by_ratio(final_aspect_ratio, target_pixels)
    final_path = env.render_final(
        output_path=output_path,
        engine=args.final_engine,
        resolution=final_resolution,
        samples=final_samples,
    )
    final_obs = env.get_observation()
    elapsed = time.time() - start_t

    print(f"final aspect ratio: {final_aspect_ratio}")
    print(f"final resolution: {final_resolution[0]}x{final_resolution[1]}")
    print(f"final image: {final_path}")
    print(f"elapsed: {elapsed:.2f}s")

    log_path = os.path.splitext(output_path)[0] + "_log.json"
    log_data = {
        "task_id": args.task_id,
        "method_id": args.method_id,
        "bootstrap_mode": args.bootstrap_mode,
        "random_seed": int(args.random_seed),
        "instruction": args.instruction,
        "reference_object": args.reference,
        "scene_file": bpy.data.filepath,
        "soft_blueprint": blueprint,
        "scene_info_text": scene_info_text,
        "scene_topology_text": scene_topology_text,
        "scout_views": scout_views,
        "global_anchor_bank": global_anchor_bank,
        "iterations": int(args.max_iter),
        "iterations_requested": int(args.max_iter),
        "iterations_executed": len(round_logs),
        "num_candidates": int(args.num_candidates),
        "allowed_aspect_ratios": allowed_ratios,
        "total_score_weights": TOTAL_SCORE_WEIGHTS,
        "early_stop_config": early_stop_config,
        "termination": termination,
        "initial_observation": initial_obs,
        "initial_view_output": initial_view_output,
        "global_best_record": global_best_record,
        "final_aspect_choice": {"aspect_ratio": final_aspect_ratio, "reasoning": final_aspect_pick.get("reasoning", "")},
        "preview_max_pixels": preview_max_pixels,
        "preview_samples": preview_samples,
        "preview_workers": preview_workers,
        "final_samples": final_samples,
        "render_engine_requested": args.render_engine,
        "preview_engine_requested": args.preview_engine,
        "preview_engine": env.preview_engine,
        "final_engine_requested": args.final_engine,
        "final_engine": env.final_engine,
        "final_output": final_path,
        "final_params": final_obs,
        "elapsed_seconds": round(float(elapsed), 4),
        "preview_log_dir": preview_log_dir,
        "rounds": round_logs,
        "final_region_memory": [
            {"region_key": list(key), **value}
            for key, value in sorted(region_memory.items(), key=lambda item: (item[1].get("status", ""), -item[1].get("best_total_score", 0.0)))
        ],
        "agent_stats": agent.get_stats(),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"log path: {log_path}")


if __name__ == "__main__":
    main()
