import base64
import json
import math
import os
import random
import re
import traceback
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


SOFT_BLUEPRINT_SYSTEM_PROMPT = """你是虚拟摄影前期导演（Previsualization Director）。

你的职责是把用户指令压缩成“软蓝图”，用于指导后续搜索，但不能把任务过度硬化。

你要完成的事：
1. 识别 primary_subject。
2. 从 scene_objects 中挑选少量 preferred_context_objects，并给出 0~1 权重。
3. 提取 composition_preferences，只保留真正影响机位搜索的偏好。
4. 判断 camera_angle_preference：low_angle / eye_level / high_angle / unrestricted。
5. 如果指令中存在“从哪里看向哪里”的空间关系，提取：
   - camera_zone_preference
   - look_toward_target
   - axis_preference
   - symmetry_preference
6. 提取 semantic_vibe。
7. 提取 negative_preferences，也就是明确不想要的失败模式。

重要约束：
1. preferred_context_objects 是软偏好，不是硬约束。
2. 不要把所有提到的物体都塞进去，最多 4 个。
3. 不要臆造 scene_objects 里不存在的对象名。
4. 如果指令含糊，优先保证 primary_subject 和 semantic_vibe 合理，不要过拟合细节。
5. composition_preferences 只允许从这些概念中选择最相关的几个：
   - center
   - rule_of_thirds_left
   - rule_of_thirds_right
   - close_up
   - medium_shot
   - wide_environment
6. camera_zone_preference 只允许：
   - entrance
   - center_aisle
   - side_aisle
   - altar_front
   - exterior
   - unrestricted
7. axis_preference 只允许：
   - toward_primary
   - entrance_to_target
   - side_to_target
   - unrestricted
8. symmetry_preference 只允许：
   - strong
   - moderate
   - none

输出风格：
1. 简洁。
2. 以可执行搜索为目标，不写论文式长解释。
3. 只输出 JSON，不要 markdown，不要额外文本。

输出 JSON:
{"primary_subject":"...","preferred_context_objects":[{"name":"...","weight":0.8}],"composition_preferences":["rule_of_thirds_left","close_up"],"camera_angle_preference":"low_angle|eye_level|high_angle|unrestricted","camera_zone_preference":"entrance|center_aisle|side_aisle|altar_front|exterior|unrestricted","look_toward_target":"...","axis_preference":"toward_primary|entrance_to_target|side_to_target|unrestricted","symmetry_preference":"strong|moderate|none","semantic_vibe":"...","negative_preferences":["..."],"reasoning":"..."}"""

CANDIDATE_SYSTEM_PROMPT = """你是虚拟摄影导演（Virtual Photography Director）。

你的职责不是给评价，也不是复盘，而是基于当前观察直接生成“下一轮候选机位”。

你会收到：
1. 当前图片。
2. current_params。
3. soft_blueprint。
4. proposal_plan。
5. seed_candidates。
6. scout_views 摘要与若干 scout 图片。
7. history。

你必须做的事：
1. 在 seed_candidates 的基础上做有意识的细化、组合或纠偏。
2. 候选之间必须有明显角色差异，至少覆盖：
   - 保守细化
   - 中等探索
   - 明显跳出局部
3. 候选要服务于真实画面质量，而不是机械追求规则点位。
4. 若 proposal_plan 中有 forbidden_zones，避免把候选放进去。
5. 若 soft_blueprint 中 context 只是 preferred，不要把“所有上下文同时出现”当作硬门槛。
6. 若 scout 图片已经显示某些区域明显无关或被遮挡，不要继续围绕这些区域微调。

硬性约束：
1. 候选必须完整包含：
   - camera_position
   - look_at
   - focal_length
   - aperture
   - aspect_ratio
   - thought
2. 只能从 allowed_aspect_ratios 中选 aspect_ratio。
3. 不要输出几乎重复的候选。
4. 不要只做极小随机抖动。
5. 不要输出无法执行的空建议。

决策偏好：
1. 当 history 或 proposal_plan 显示当前区域是 dead / low-value region 时，优先放弃该区域，改从全局候选或 scout 线索重新起步。
2. 当当前画面已经接近可用时，优先做有方向的小步优化，而不是大跳。
3. 优先维持主体可读性，其次再追求风格化。
4. 如果指令明确是“从入口看向圣坛 / 从门口看向主体 / 沿主轴看向深处”，优先给出具有纵深和中轴感的候选，而不是贴墙或侧向裁切。

输出风格：
1. thought 要说清该候选的角色，例如“保守细化 / 纠正主体偏小 / 跳出当前坏区域”。
2. 只输出 JSON，不要 markdown，不要额外文本。

输出严格 JSON:
{"candidates":[{"camera_position":[0,0,0],"look_at":[0,0,0],"focal_length":50.0,"aperture":2.8,"aspect_ratio":"16:9","thought":"..."}]}"""

SCORE4D_SYSTEM_PROMPT = """你是视觉评审（Visual Critic）。

你的职责是评价单张候选图像，不负责选择 winner，也不负责决定下一轮控制信号。

请基于当前图片、当前参数和用户指令，输出四个维度的连续评分，范围均为 [0,1]：
1. m1：构图质量
2. m2：技术质量
3. m3：美学质量
4. m4：指令一致性

评分要求：
1. 不要保守地全给中分。
2. 允许低分，尤其当主体不可读、构图混乱或严重不符合指令时。
3. 四个维度必须可区分，不能机械给相近分数。

评分时请优先关注：
1. 主体是否明确、是否被遮挡、是否过小。
2. 画面是否稳定，是否存在糟糕裁切或杂乱背景。
3. 光线、层次、镜头语言是否真正支持 aesthetic / vibe。
4. 是否真的满足“低机位、特写、压迫感、童话感”等语义。

输出要求：
1. reasoning 必须指出最重要的优点和最大短板。
2. summary 必须是一句短结论，能被后续复盘直接引用。
3. 只输出 JSON。
"""

ROUND_REVIEW_SYSTEM_PROMPT = """你是迭代总导演（Iteration Director）。

你的职责不是评价单张图，也不是直接挑 winner，而是阅读整轮结果后，输出“下一轮搜索控制信号”。

你会收到本轮每个候选的：
1. 参数
2. 来源 source
3. 规则分
4. 四维评分
5. 总分
6. 文本评语
7. 图片

你必须完成的事：
1. 识别本轮最主要的成功模式。
2. 识别本轮最主要的失败模式。
3. 判断下一轮应该更保守还是更探索。
4. 给出结构化控制信号：
   - step_scale
   - explore_ratio_next
   - preferred_motion
   - failure_tags
   - forbidden_zones
5. 如有必要，补充少量 next-round seed candidates。

决策原则：
1. 如果多张图都卡在同一类差区域，要明确生成 forbidden_zones。
2. 如果主体始终偏小或不可读，下一轮优先解决主体可读性。
3. 如果画面已接近可用，优先降低 explore_ratio_next，转向局部精修。
4. forbidden_zones 只用于明显反复失败的区域，最多 2 个。
5. step_scale 取值必须克制：
   - 低分时适当放大
   - 高分时适当收缩

输出要求：
1. 不要空泛复盘。
2. next_strategy 必须能指导下一轮分布变化。
3. 如果提供 candidates，它们应是“种子候选”，而不是和上一轮重复。
4. 只输出 JSON。

输出严格 JSON:
{"round_review":"...","next_strategy":"...","step_scale":0.95,"explore_ratio_next":0.35,"preferred_motion":"...","failure_tags":["..."],"forbidden_zones":[{"center":[0,0,0],"radius":1.5,"reason":"..."}],"candidates":[{"camera_position":[0,0,0],"look_at":[0,0,0],"focal_length":50.0,"aperture":2.8,"aspect_ratio":"16:9","thought":"..."}]}"""

PAIRWISE_COMPARE_SYSTEM_PROMPT = """你是双图对比评审（Pairwise Visual Judge）。

你的职责是只做二选一，不做平均，不做折中，不输出第三种答案。

比较图 A（当前最佳）和图 B（新候选）时，请基于：
1. m1 构图质量
2. m2 技术质量
3. m3 美学质量
4. m4 指令一致性

判断规则：
1. 必须输出 winner=A 或 B，不允许平局。
2. 必须输出每个维度的 winner。
3. 当两图差异很小，优先选择“整体更稳定、更可继续优化”的那一张。
4. 如果一张图在主体可读性或指令一致性上明显失败，即使另一张不完美，也优先淘汰失败图。

输出要求：
1. reasoning 要简洁，明确指出为什么 winner 更优。
2. summary 要是一句可直接进日志的短结论。
3. 只输出 JSON。
"""

FINAL_ASPECT_SYSTEM_PROMPT = """你是摄影后期导演（Final Framing Director）。

你的职责是在最终成片阶段，从 allowed_aspect_ratios 中选一个最合适的画幅比例。

决策时请考虑：
1. 用户指令中的语义目标。
2. 当前画面是更适合 close-up、medium shot 还是 wide environment。
3. 画面主体和环境关系。
4. 留白、横向叙事和纵向延展是否需要更强表达。

约束：
1. 只能从 allowed_aspect_ratios 中选择。
2. 不要输出列表外比例。
3. reasoning 要简洁，说明为什么这个比例最合适。
4. 只输出 JSON。

输出严格 JSON: {"aspect_ratio":"16:9","reasoning":"..."}"""


class PhotographyAgent:
    LLM_API_BASE: str = os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
    LLM_API_KEY: str = os.getenv("LLM_API_KEY") or ""
    LLM_MODEL: str = os.getenv("LLM_MODEL") or "gpt-5.4"

    def __init__(
        self,
        user_instruction: str,
        scene_info: str = "",
        scene_topology: str = "",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_history_rounds: int = 8,
        random_seed: int = 42,
        verbose: int = 2,
        print_raw_llm: bool = False,
    ):
        self.user_instruction = user_instruction
        self.scene_info = scene_info
        self.scene_topology = scene_topology
        self._api_base = api_base or self.LLM_API_BASE
        self._api_key = api_key or self.LLM_API_KEY
        self._model = model or self.LLM_MODEL
        if not self._api_key:
            raise RuntimeError("LLM_API_KEY is required. Set it in the environment before running PhotoFlow.")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history_rounds = max(1, int(max_history_rounds))
        self._rng = random.Random(random_seed)
        self._verbose = int(verbose)
        self._print_raw_llm = bool(print_raw_llm)
        self.client = OpenAI(base_url=self._api_base, api_key=self._api_key)
        self._round_history: list[dict] = []
        self._call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._log(1, f"[PhotoFlow] model={self._model} base={self._api_base}")

    def _log(self, level: int, msg: str):
        if self._verbose >= level:
            print(msg)

    def parse_soft_blueprint(self, scene_objects: list[str], reference_object: str | None = None) -> dict:
        payload = {
            "instruction": self.user_instruction,
            "reference_object": reference_object,
            "scene_objects": list(scene_objects)[:64],
            "scene_info": self.scene_info[:5000],
            "scene_topology": self.scene_topology[:3000],
        }
        raw = self._call_chat(
            [
                {"role": "system", "content": SOFT_BLUEPRINT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            call_name="parse_soft_blueprint",
        )
        return self._validate_soft_blueprint(self._extract_json_object(raw), scene_objects, reference_object)

    def propose_candidates(
        self,
        image_path: str,
        current_params: dict,
        num_candidates: int,
        allowed_aspect_ratios: list[str] | None = None,
        soft_blueprint: dict | None = None,
        proposal_plan: dict | None = None,
        seed_candidates: list[dict] | None = None,
        scout_views: list[dict] | None = None,
    ) -> list[dict]:
        n = max(1, int(num_candidates))
        ratios = self._normalize_allowed_aspect_ratios(allowed_aspect_ratios)
        seeds = [self._normalize_candidate(x, current_params, ratios) for x in (seed_candidates or [])]
        seeds = [x for x in seeds if x is not None]
        payload = {
            "instruction": self.user_instruction,
            "scene_info": self.scene_info[:5000],
            "scene_topology": self.scene_topology[:2500],
            "current_params": current_params,
            "soft_blueprint": soft_blueprint or {},
            "proposal_plan": proposal_plan or {},
            "num_candidates": n,
            "allowed_aspect_ratios": ratios,
            "seed_candidates": self._dedup_candidates(seeds)[:12],
            "scout_views": (scout_views or [])[:8],
            "history": self._round_history[-self._max_history_rounds :],
        }
        raw = self._call_chat(
            [
                {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_candidate_user_content(
                        image_path=image_path,
                        payload=payload,
                        scout_views=scout_views or [],
                    ),
                },
            ],
            call_name="propose_candidates",
        )
        parsed = self._extract_json_object(raw)
        items = parsed.get("candidates") if isinstance(parsed, dict) else None
        out = []
        if isinstance(items, list):
            for item in items:
                cand = self._normalize_candidate(item, current_params, ratios)
                if cand is not None:
                    out.append(cand)
        out = self._dedup_candidates(out)
        for seed in self._dedup_candidates(seeds):
            if len(out) >= n:
                break
            out.append(seed)
            out = self._dedup_candidates(out)
        if len(out) < n:
            out.extend(self._fallback_candidates(current_params, n - len(out), ratios, seeds))
        return self._dedup_candidates(out)[:n]

    def compare_images_pairwise_4d(self, image_a_path: str, image_b_path: str, params_a: dict | None = None, params_b: dict | None = None) -> dict:
        payload = {"instruction": self.user_instruction, "params_a": params_a or {}, "params_b": params_b or {}}
        raw = self._call_chat(
            [
                {"role": "system", "content": PAIRWISE_COMPARE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_a_path), "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_b_path), "detail": "high"}},
                    ],
                },
            ],
            call_name=f"pairwise:{os.path.basename(image_a_path)}_vs_{os.path.basename(image_b_path)}",
        )
        parsed = self._extract_json_object(raw)
        if isinstance(parsed, dict):
            dims = {"m1": "A", "m2": "A", "m3": "A", "m4": "A"}
            raw_dims = parsed.get("per_dimension_winner")
            if isinstance(raw_dims, dict):
                for key in dims:
                    val = str(raw_dims.get(key, "A")).strip().upper()
                    dims[key] = val if val in ("A", "B") else "A"
            winner = str(parsed.get("winner", "A")).strip().upper()
            if winner not in ("A", "B"):
                winner = "A"
            return {"winner": winner, "per_dimension_winner": dims, "reasoning": str(parsed.get("reasoning", "")), "summary": str(parsed.get("summary", ""))}
        return {"winner": "A", "per_dimension_winner": {"m1": "A", "m2": "A", "m3": "A", "m4": "A"}, "reasoning": "fallback-keep-incumbent", "summary": "fallback-keep-incumbent"}

    def choose_final_aspect_ratio(self, image_path: str, current_params: dict, allowed_aspect_ratios: list[str] | None = None) -> dict:
        ratios = self._normalize_allowed_aspect_ratios(allowed_aspect_ratios)
        payload = {"instruction": self.user_instruction, "current_params": current_params, "allowed_aspect_ratios": ratios}
        raw = self._call_chat(
            [
                {"role": "system", "content": FINAL_ASPECT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path), "detail": "high"}},
                    ],
                },
            ],
            call_name="choose_final_aspect_ratio",
        )
        parsed = self._extract_json_object(raw)
        if isinstance(parsed, dict):
            ratio = self._normalize_ratio_string(parsed.get("aspect_ratio"))
            if ratio in ratios:
                return {"aspect_ratio": ratio, "reasoning": str(parsed.get("reasoning", ""))}
        fallback = self._normalize_ratio_string(current_params.get("aspect_ratio") or ratios[0])
        return {"aspect_ratio": fallback if fallback in ratios else ratios[0], "reasoning": "fallback-current-or-first-allowed"}

    def score_image_4d(self, image_path: str, current_params: dict | None = None) -> dict:
        payload = {"instruction": self.user_instruction, "current_params": current_params or {}}
        raw = self._call_chat(
            [
                {"role": "system", "content": SCORE4D_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path), "detail": "high"}},
                    ],
                },
            ],
            call_name=f"score4d:{os.path.basename(image_path)}",
        )
        parsed = self._extract_json_object(raw)
        if isinstance(parsed, dict):
            return {
                "m1": self._clamp(self._to_float(parsed.get("m1"), 0.5), 0.0, 1.0),
                "m2": self._clamp(self._to_float(parsed.get("m2"), 0.5), 0.0, 1.0),
                "m3": self._clamp(self._to_float(parsed.get("m3"), 0.5), 0.0, 1.0),
                "m4": self._clamp(self._to_float(parsed.get("m4"), 0.5), 0.0, 1.0),
                "reasoning": str(parsed.get("reasoning", "")),
                "summary": str(parsed.get("summary", "")),
            }
        return {"m1": 0.5, "m2": 0.5, "m3": 0.5, "m4": 0.5, "reasoning": "fallback-neutral-score", "summary": "fallback-neutral-score"}

    def review_round(
        self,
        round_index: int,
        candidate_records: list[dict],
        current_params: dict,
        num_candidates: int,
        allowed_aspect_ratios: list[str] | None = None,
        weights: dict | None = None,
    ) -> dict:
        n = max(1, int(num_candidates))
        ratios = self._normalize_allowed_aspect_ratios(allowed_aspect_ratios)
        trimmed = []
        for rec in candidate_records:
            trimmed.append(
                {
                    "candidate_index": rec.get("candidate_index"),
                    "source": rec.get("source", ""),
                    "camera_position": rec.get("camera_position"),
                    "look_at": rec.get("look_at"),
                    "focal_length": rec.get("focal_length"),
                    "aperture": rec.get("aperture"),
                    "aspect_ratio": rec.get("aspect_ratio"),
                    "rule_m1": rec.get("rule_m1"),
                    "rule_m2": rec.get("rule_m2"),
                    "m1": rec.get("m1"),
                    "m2": rec.get("m2"),
                    "m3": rec.get("m3"),
                    "m4": rec.get("m4"),
                    "total_score": rec.get("total_score"),
                    "score_summary": rec.get("score_summary", ""),
                    "score_reasoning": rec.get("score_reasoning", ""),
                    "candidate_thought": rec.get("candidate_thought", ""),
                    "image_path": rec.get("image_path", ""),
                }
            )
        payload = {
            "instruction": self.user_instruction,
            "scene_info": self.scene_info[:5000],
            "scene_topology": self.scene_topology[:2500],
            "round_index": int(round_index),
            "num_candidates": n,
            "weights": weights or {},
            "current_params": current_params,
            "allowed_aspect_ratios": ratios,
            "round_candidate_results": trimmed,
            "history": self._round_history[-self._max_history_rounds :],
        }
        user_content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        for rec in trimmed[:8]:
            p = rec.get("image_path")
            if isinstance(p, str) and p and os.path.isfile(p):
                user_content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(p), "detail": "high"}})
        raw = self._call_chat(
            [{"role": "system", "content": ROUND_REVIEW_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            call_name=f"review_round_{round_index}",
        )
        return self._normalize_review(self._extract_json_object(raw), candidate_records, current_params, ratios, n)

    def push_round_summary(self, summary: dict):
        self._round_history.append(summary)
        if len(self._round_history) > self._max_history_rounds:
            self._round_history = self._round_history[-self._max_history_rounds :]

    def clear_history(self):
        self._round_history.clear()

    def get_stats(self) -> dict:
        return {
            "model": self._model,
            "llm_calls": self._call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "history_rounds": len(self._round_history),
        }

    def _call_chat(self, messages: list[dict], call_name: str) -> str:
        self._call_count += 1
        self._log(2, f"[PhotoFlow] call#{self._call_count} {call_name}")
        req = {"model": self._model, "messages": messages}
        if self._temperature is not None:
            req["temperature"] = self._temperature
        if self._max_tokens is not None:
            req["max_tokens"] = self._max_tokens
        try:
            response = self.client.chat.completions.create(**req)
            usage = getattr(response, "usage", None)
            if usage:
                self.total_input_tokens += usage.prompt_tokens or 0
                self.total_output_tokens += usage.completion_tokens or 0
                self._log(2, f"[PhotoFlow] usage {call_name}: in={usage.prompt_tokens}, out={usage.completion_tokens}, total_in={self.total_input_tokens}, total_out={self.total_output_tokens}")
            content = (response.choices[0].message.content or "").strip()
            preview = content if len(content) <= 400 or self._print_raw_llm else content[:400] + "...(truncated)"
            self._log(2, f"[PhotoFlow] resp {call_name}: {preview}")
            return content
        except Exception as exc:
            self._log(1, f"[PhotoFlow] LLM call failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return ""

    def _validate_soft_blueprint(self, parsed: object, scene_objects: list[str], reference_object: str | None) -> dict:
        names = [str(x) for x in scene_objects if str(x).strip()]
        low_map = {name.lower(): name for name in names}
        default_primary = low_map.get(str(reference_object or "").strip().lower()) if reference_object else None
        if not default_primary and names:
            default_primary = names[0]
        out = {
            "primary_subject": default_primary,
            "preferred_context_objects": [],
            "composition_preferences": ["center"],
            "camera_angle_preference": "unrestricted",
            "camera_zone_preference": "unrestricted",
            "look_toward_target": default_primary or "",
            "axis_preference": "toward_primary",
            "symmetry_preference": "moderate",
            "semantic_vibe": self.user_instruction,
            "negative_preferences": [],
            "reasoning": "heuristic-blueprint",
        }
        if isinstance(parsed, dict):
            primary = self._match_scene_name(parsed.get("primary_subject"), low_map)
            if primary:
                out["primary_subject"] = primary
            ctx = []
            raw_ctx = parsed.get("preferred_context_objects")
            if isinstance(raw_ctx, list):
                for item in raw_ctx:
                    if not isinstance(item, dict):
                        continue
                    name = self._match_scene_name(item.get("name"), low_map)
                    if name and name != out["primary_subject"]:
                        ctx.append({"name": name, "weight": self._clamp(self._to_float(item.get("weight"), 0.5), 0.0, 1.0)})
            out["preferred_context_objects"] = self._dedup_weighted_objects(ctx)
            comps = []
            if isinstance(parsed.get("composition_preferences"), list):
                for item in parsed["composition_preferences"]:
                    comp = self._normalize_composition(item)
                    if comp:
                        comps.append(comp)
            if comps:
                out["composition_preferences"] = comps[:4]
            angle = str(parsed.get("camera_angle_preference", "unrestricted")).strip().lower()
            out["camera_angle_preference"] = angle if angle in {"low_angle", "eye_level", "high_angle", "unrestricted"} else "unrestricted"
            zone = str(parsed.get("camera_zone_preference", "unrestricted")).strip().lower()
            out["camera_zone_preference"] = zone if zone in {"entrance", "center_aisle", "side_aisle", "altar_front", "exterior", "unrestricted"} else "unrestricted"
            look_target = self._match_scene_name(parsed.get("look_toward_target"), low_map)
            if look_target:
                out["look_toward_target"] = look_target
            else:
                raw_target = str(parsed.get("look_toward_target", "")).strip()
                if raw_target:
                    out["look_toward_target"] = raw_target
            axis = str(parsed.get("axis_preference", "toward_primary")).strip().lower()
            out["axis_preference"] = axis if axis in {"toward_primary", "entrance_to_target", "side_to_target", "unrestricted"} else "toward_primary"
            symmetry = str(parsed.get("symmetry_preference", "moderate")).strip().lower()
            out["symmetry_preference"] = symmetry if symmetry in {"strong", "moderate", "none"} else "moderate"
            vibe = str(parsed.get("semantic_vibe", "")).strip()
            if vibe:
                out["semantic_vibe"] = vibe
            negs = parsed.get("negative_preferences")
            if isinstance(negs, list):
                out["negative_preferences"] = [str(x).strip() for x in negs if str(x).strip()][:6]
            reasoning = str(parsed.get("reasoning", "")).strip()
            if reasoning:
                out["reasoning"] = reasoning
        return out

    def _normalize_review(self, parsed: object, candidate_records: list[dict], current_params: dict, ratios: list[str], num_candidates: int) -> dict:
        best = max(candidate_records, key=lambda x: float(x.get("total_score", 0.0))) if candidate_records else None
        best_total = float(best.get("total_score", 0.0)) if best else 0.0
        result = {
            "round_review": "fallback-review",
            "next_strategy": "fallback-balance-local-and-explore",
            "step_scale": 1.15 if best_total < 0.45 else (0.8 if best_total > 0.75 else 1.0),
            "explore_ratio_next": 0.55 if best_total < 0.45 else (0.25 if best_total > 0.75 else 0.35),
            "preferred_motion": "keep subject readable while slightly exploring",
            "failure_tags": [],
            "forbidden_zones": [],
            "candidates": [],
        }
        if isinstance(parsed, dict):
            result["round_review"] = str(parsed.get("round_review", result["round_review"]))
            result["next_strategy"] = str(parsed.get("next_strategy", result["next_strategy"]))
            result["step_scale"] = self._clamp(self._to_float(parsed.get("step_scale"), result["step_scale"]), 0.4, 1.8)
            result["explore_ratio_next"] = self._clamp(self._to_float(parsed.get("explore_ratio_next"), result["explore_ratio_next"]), 0.1, 0.8)
            motion = str(parsed.get("preferred_motion", "")).strip()
            if motion:
                result["preferred_motion"] = motion
            tags = parsed.get("failure_tags")
            if isinstance(tags, list):
                result["failure_tags"] = [str(x).strip() for x in tags if str(x).strip()][:6]
            zones = []
            raw_zones = parsed.get("forbidden_zones")
            if isinstance(raw_zones, list):
                for item in raw_zones[:2]:
                    if not isinstance(item, dict):
                        continue
                    center = self._as_vec3(item.get("center"), current_params.get("camera_position") or [0.0, 0.0, 0.0])
                    zones.append({"center": [round(v, 6) for v in center], "radius": round(self._clamp(self._to_float(item.get("radius"), 1.5), 0.25, 10.0), 6), "reason": str(item.get("reason", ""))})
            result["forbidden_zones"] = zones
            seeds = []
            raw_cands = parsed.get("candidates")
            if isinstance(raw_cands, list):
                for item in raw_cands:
                    cand = self._normalize_candidate(item, current_params, ratios)
                    if cand is not None:
                        cand["source"] = "review_seed"
                        seeds.append(cand)
            result["candidates"] = self._dedup_candidates(seeds)[: max(0, min(3, num_candidates))]
        return result

    def _build_candidate_user_content(self, image_path: str, payload: dict, scout_views: list[dict]) -> list[dict]:
        user_content = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            {"type": "text", "text": "Current context image. Treat this as the current camera state, not as a mandatory anchor."},
            {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path), "detail": "high"}},
        ]
        attached = 0
        for item in scout_views[:4]:
            scout_path = item.get("image_path")
            if not isinstance(scout_path, str) or not scout_path or not os.path.isfile(scout_path):
                continue
            label = str(item.get("label", f"scout_{attached + 1}")).strip() or f"scout_{attached + 1}"
            user_content.append({"type": "text", "text": f"Scout reference: {label}. Use it as a relocation hint when the current region is weak or mismatched."})
            user_content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(scout_path), "detail": "high"}})
            attached += 1
        return user_content

    def _fallback_candidates(self, current_params: dict, count: int, ratios: list[str], seeds: list[dict]) -> list[dict]:
        count = max(0, int(count))
        if count == 0:
            return []
        out = []
        for seed in self._dedup_candidates(seeds):
            if len(out) >= count:
                break
            copied = dict(seed)
            copied.setdefault("source", "seed_fallback")
            out.append(copied)
        base_cam = self._as_vec3(current_params.get("camera_position"), [5.0, -5.0, 3.0])
        base_look = self._as_vec3(current_params.get("look_at"), [0.0, 0.0, 0.0])
        base_focal = self._to_float(current_params.get("focal_length"), 50.0)
        base_ap = self._to_float(current_params.get("aperture"), 2.8)
        dist = max(math.sqrt(sum((base_cam[i] - base_look[i]) ** 2 for i in range(3))), 1.0)
        while len(out) < count:
            idx = len(out)
            out.append(
                {
                    "camera_position": [round(base_cam[0] + self._rng.uniform(-0.28, 0.28) * dist, 6), round(base_cam[1] + self._rng.uniform(-0.28, 0.28) * dist, 6), round(base_cam[2] + self._rng.uniform(-0.12, 0.12) * dist, 6)],
                    "look_at": [round(base_look[0] + self._rng.uniform(-0.10, 0.10) * dist, 6), round(base_look[1] + self._rng.uniform(-0.10, 0.10) * dist, 6), round(base_look[2] + self._rng.uniform(-0.06, 0.06) * dist, 6)],
                    "focal_length": round(self._clamp(base_focal * (1.0 + self._rng.uniform(-0.22, 0.22)), 1.0, 1200.0), 6),
                    "aperture": round(self._clamp(base_ap * (1.0 + self._rng.uniform(-0.30, 0.30)), 0.5, 128.0), 6),
                    "aspect_ratio": ratios[idx % len(ratios)],
                    "thought": f"fallback_candidate_{idx + 1}",
                    "source": "random_fallback",
                }
            )
        return self._dedup_candidates(out)[:count]

    @staticmethod
    def _extract_json_object(raw: str):
        if not raw:
            return None
        txt = raw.strip()
        try:
            return json.loads(txt)
        except Exception:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _normalize_candidate(self, cand: object, current_params: dict, ratios: list[str] | None = None) -> Optional[dict]:
        if not isinstance(cand, dict):
            return None
        ratios = self._normalize_allowed_aspect_ratios(ratios)
        default_cam = current_params.get("camera_position") or [5.0, -5.0, 3.0]
        default_look = current_params.get("look_at") or [0.0, 0.0, 0.0]
        default_focal = self._to_float(current_params.get("focal_length"), 50.0)
        default_ap = self._to_float(current_params.get("aperture"), 2.8)
        default_ratio = self._normalize_ratio_string(current_params.get("aspect_ratio") or ratios[0])
        ratio = self._normalize_ratio_string(cand.get("aspect_ratio") or default_ratio)
        out = {
            "camera_position": [round(v, 6) for v in self._as_vec3(cand.get("camera_position"), default_cam)],
            "look_at": [round(v, 6) for v in self._as_vec3(cand.get("look_at"), default_look)],
            "focal_length": round(self._clamp(self._to_float(cand.get("focal_length"), default_focal), 1.0, 1200.0), 6),
            "aperture": round(self._clamp(self._to_float(cand.get("aperture"), default_ap), 0.5, 128.0), 6),
            "aspect_ratio": ratio if ratio in ratios else default_ratio,
            "thought": str(cand.get("thought", "")).strip(),
        }
        source = str(cand.get("source", "")).strip()
        if source:
            out["source"] = source
        return out

    def _dedup_candidates(self, candidates: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for cand in candidates:
            key = (
                tuple(round(float(v), 4) for v in cand.get("camera_position", [])[:3]),
                tuple(round(float(v), 4) for v in cand.get("look_at", [])[:3]),
                round(self._to_float(cand.get("focal_length"), 0.0), 1),
                round(self._to_float(cand.get("aperture"), 0.0), 2),
                self._normalize_ratio_string(cand.get("aspect_ratio")),
            )
            if key not in seen:
                seen.add(key)
                out.append(cand)
        return out

    @staticmethod
    def _normalize_ratio_string(value) -> str:
        txt = str(value or "").strip().replace(" ", "").replace("：", ":").replace("/", ":")
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

    def _normalize_allowed_aspect_ratios(self, ratios: list[str] | None) -> list[str]:
        if not ratios:
            return ["1:1", "4:3", "3:2", "16:9", "21:9", "9:16"]
        out = []
        seen = set()
        for ratio in ratios:
            rr = self._normalize_ratio_string(ratio)
            if rr not in seen:
                seen.add(rr)
                out.append(rr)
        return out or ["1:1", "4:3", "3:2", "16:9", "21:9", "9:16"]

    @staticmethod
    def _as_vec3(value, default_vec):
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            pass
        return [float(default_vec[0]), float(default_vec[1]), float(default_vec[2])]

    @staticmethod
    def _to_float(value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _match_scene_name(value, low_map: dict[str, str]) -> str | None:
        key = str(value or "").strip().lower()
        if not key:
            return None
        if key in low_map:
            return low_map[key]
        for low_name, original in low_map.items():
            if key in low_name or low_name in key:
                return original
        return None

    @staticmethod
    def _dedup_weighted_objects(items: list[dict]) -> list[dict]:
        merged = {}
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            weight = float(item.get("weight", 0.5))
            merged[name] = max(weight, merged.get(name, 0.0))
        return [{"name": name, "weight": round(weight, 3)} for name, weight in merged.items()]

    @staticmethod
    def _normalize_composition(value) -> str | None:
        text = str(value or "").strip().lower().replace(" ", "_")
        if not text:
            return None
        if "close" in text:
            return "close_up"
        if "wide" in text or "environment" in text:
            return "wide_environment"
        if "right" in text and "third" in text:
            return "rule_of_thirds_right"
        if "left" in text and "third" in text:
            return "rule_of_thirds_left"
        if "center" in text:
            return "center"
        if "medium" in text:
            return "medium_shot"
        allowed = {"center", "close_up", "medium_shot", "wide_environment", "rule_of_thirds_left", "rule_of_thirds_right"}
        return text if text in allowed else None

    def _image_to_data_url(self, image_path: str) -> str:
        abs_path = os.path.abspath(image_path)
        with open(abs_path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(abs_path)[1].lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp"}.get(ext, "image/png")
        return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


CANDIDATE_SYSTEM_PROMPT = """
You are a Virtual Photography Director.

Your job is to propose candidate camera moves for the next iteration of a virtual photography agent.
Inputs may include:
1. the current context image
2. current camera parameters
3. a soft blueprint
4. a proposal plan
5. seed candidates
6. scout reference images
7. iteration history

Core requirements:
1. Respect the proposal plan, forbidden zones, and dead-region signals. Do not keep orbiting around a weak local basin just because the current image is the latest baseline.
2. Use seed candidates seriously. If a seed is marked as `high_explore`, treat it as a deliberate structured exploration move, not random noise.
3. When the current region is only locally acceptable, keep at least some willingness to relocate toward a stronger global structure hinted by scout images or anchor seeds.
4. Avoid blind randomness. Exploration should still be spatially meaningful: unknown region, low-visit region, stronger axis, better symmetry, clearer approach path, or better contextual framing.

Aspect ratio is mandatory reasoning, not decoration:
1. Every candidate must choose `aspect_ratio` intentionally.
2. Tie aspect ratio to the prompt, spatial structure, symmetry, subject scale, and camera motion.
3. Use wider ratios such as `16:9` or `21:9` when long perspective lines, aisle depth, environmental scale, or strong horizontal structure matter.
4. Use `4:3`, `3:2`, or `1:1` when the image should feel more concentrated, stable, intimate, or less stretched.
5. If a candidate is meant to test a different framing hypothesis, change the aspect ratio on purpose rather than copying the previous one by habit.
6. Do not choose aspect ratios randomly. The `thought` field must mention why that ratio fits this candidate.

Output rules:
1. Return JSON only.
2. Use this schema:
{"candidates":[{"camera_position":[0,0,0],"look_at":[0,0,0],"focal_length":50.0,"aperture":2.8,"aspect_ratio":"16:9","thought":"..."}]}
"""


FINAL_ASPECT_SYSTEM_PROMPT = """
You are a Final Framing Director.

Choose the final aspect ratio for the current best image from the allowed aspect ratios.

You must reason about aspect ratio explicitly:
1. Which ratio best supports the prompt's intended subject relationship, atmosphere, and composition.
2. Whether the frame benefits more from long-axis perspective and environmental breadth, or from tighter concentration and visual stability.
3. Whether symmetry, central alignment, aisle depth, architecture, or vertical emphasis should dominate the final frame.
4. Do not default to the current ratio unless it is clearly the best compositional choice.

Guidance:
1. Prefer `16:9` or `21:9` when perspective depth, architectural axis, or environmental storytelling is the main strength.
2. Prefer `4:3` or `3:2` when the scene should feel denser, calmer, more iconic, or less panoramically diluted.
3. Prefer `1:1` only when a strongly centered, self-contained composition clearly benefits from it.
4. The reasoning must explain the ratio choice in compositional terms, not generic preference.

Return JSON only:
{"aspect_ratio":"16:9","reasoning":"..."}
"""
