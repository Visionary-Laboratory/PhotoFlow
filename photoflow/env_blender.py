"""
BlenderPhotographyEnv —— 基于 Blender bpy 的自由摄影环境类

坐标系约定（Blender 世界坐标，右手系）：
    - X 轴: 向右
    - Y 轴: 向前（深度方向）
    - Z 轴: 向上
    - camera_position : 相机在世界空间中的绝对位置 [x, y, z]
    - look_at         : 相机注视的目标点坐标 [x, y, z]

用法（在 Blender Python 控制台或通过 blender -b ... -P 调用）：
    >>> env = BlenderPhotographyEnv(reference_object='Car')
    >>> obs = env.reset()
    >>> obs = env.step({'camera_position': [3, -4, 2], 'look_at': [0, 0, 0.5], 'focal_length': 85})
    >>> env.render_preview('output/preview.jpg')
"""

import bpy
import math
import os
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# ══════════════════════════════════════════════════════════════
#  BlenderPhotographyEnv
# ══════════════════════════════════════════════════════════════

class BlenderPhotographyEnv:
    """
    自由坐标系摄影环境 —— 允许完全自由地控制相机位置和注视方向。

    与极坐标系不同，本环境不将相机锁定在任何物体上，
    智能体可以自由选择画面中包含哪些物体、以什么角度拍摄、
    将注视点放在哪里以实现各种构图效果。

    Parameters
    ----------
    reference_object : str, optional
        场景中的参考物体名称（如 'Car'、'Bunny'）。
        如果提供，将用于：
        - 自动计算合理的初始相机位置和注视点；
        - 仅作为初始化参考，不会限制相机运动范围。
        如果不提供，将基于整个场景的包围盒自动初始化。
    camera_position : list[float], optional
        显式指定初始相机位置 [x, y, z]。若未指定，自动计算。
    look_at : list[float], optional
        显式指定初始注视点 [x, y, z]。若未指定，自动计算。
    focal_length : float
        初始焦距（mm），默认 50.0。
    aperture : float
        初始光圈 f 值，默认 2.8。
    camera_name : str
        使用 / 创建的相机对象名称，默认 'AgentCamera'。
    preview_resolution : tuple[int, int]
        render_preview 输出分辨率，默认 (512, 512)。
    """

    def __init__(
        self,
        reference_object: str = None,
        camera_position: list = None,
        look_at: list = None,
        focal_length: float = 50.0,
        aperture: float = 2.8,
        camera_name: str = "AgentCamera",
        preview_resolution: tuple = (512, 512),
        aspect_ratio: str = "1:1",
        preview_max_pixels: int = 512 * 512,
        preview_samples: int = 64,
        preview_engine: str = "CYCLES",
        final_engine: str = "CYCLES",
    ):
        # ── 参考物体（仅用于初始化） ──
        self._ref_obj = None
        self._ref_name = reference_object
        if reference_object:
            if reference_object in bpy.data.objects:
                self._ref_obj = bpy.data.objects[reference_object]
                print(f"[Env] 参考物体: {reference_object}")
            else:
                avail = [o.name for o in bpy.data.objects if o.type == "MESH"]
                print(f"[Env] ⚠ 未找到参考物体 '{reference_object}'")
                print(f"[Env]   可用网格物体: {avail}")

        # ── 计算初始位置 ──
        auto_cam, auto_look = self._compute_initial_position()

        self._init_camera_position = list(camera_position) if camera_position else auto_cam
        self._init_look_at = list(look_at) if look_at else auto_look
        self._init_focal_length = focal_length
        self._init_aperture = aperture
        self._init_aspect_ratio = self._normalize_aspect_ratio(aspect_ratio)

        # ── 当前状态 ──
        self.camera_position = list(self._init_camera_position)
        self.look_at = list(self._init_look_at)
        self.focal_length = focal_length
        self.aperture = aperture
        self.aspect_ratio = self._init_aspect_ratio

        # ── 相机对象 ──
        self.camera_name = camera_name
        self.camera_obj = None    # bpy.types.Object
        self.camera_data = None   # bpy.types.Camera
        self._look_at_empty = None  # Empty object for TrackTo

        # ── 渲染设置 ──
        self.preview_resolution_base = tuple(preview_resolution)
        self.preview_max_pixels = max(1, int(preview_max_pixels))
        self.preview_samples = max(1, min(int(preview_samples), 64))
        self.preview_resolution = self._compute_capped_resolution(
            ratio=self.aspect_ratio,
            max_pixels=self.preview_max_pixels,
            fallback_resolution=self.preview_resolution_base,
        )
        self.preview_engine = self._resolve_render_engine(preview_engine, fallback="CYCLES")
        self.final_engine = self._resolve_render_engine(final_engine, fallback="CYCLES")

        # ── 初始化场景 ──
        self._setup_camera()
        self._apply_state()

    # ==============================================================
    #  初始位置计算
    # ==============================================================

    def _compute_initial_position(self) -> tuple:
        """
        根据场景内容自动计算合理的初始相机位置和注视点。

        策略：
        1. 如果有参考物体 → 注视该物体中心，相机在其附近
        2. 否则 → 注视场景中心，相机根据场景大小调整距离

        Returns
        -------
        tuple[list, list]
            (camera_position, look_at) 各为 [x, y, z]
        """
        if self._ref_obj:
            center = self._get_object_center(self._ref_obj)
            bbox = self._get_object_bbox(self._ref_obj)
            obj_size = max(bbox["size"][0], bbox["size"][1], bbox["size"][2], 0.5)
            distance = max(obj_size * 3, 2.0)

            az = math.radians(45)
            el = math.radians(25)
            cam_x = center[0] + distance * math.cos(el) * math.cos(az)
            cam_y = center[1] - distance * math.cos(el) * math.sin(az)
            cam_z = center[2] + distance * math.sin(el)

            print(f"[Env] 自动初始化（参考物体）: cam=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), "
                  f"look=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            return [cam_x, cam_y, cam_z], list(center)

        # ── 无参考物体：基于场景整体 ──
        mesh_objs = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get()]
        if mesh_objs:
            all_centers = [self._get_object_center(o) for o in mesh_objs]
            center = [
                sum(c[0] for c in all_centers) / len(all_centers),
                sum(c[1] for c in all_centers) / len(all_centers),
                sum(c[2] for c in all_centers) / len(all_centers),
            ]

            # 场景尺寸
            all_bboxes = [self._get_object_bbox(o) for o in mesh_objs]
            all_mins = [b["min"] for b in all_bboxes]
            all_maxs = [b["max"] for b in all_bboxes]
            scene_min = [min(m[i] for m in all_mins) for i in range(3)]
            scene_max = [max(m[i] for m in all_maxs) for i in range(3)]
            scene_size = max(scene_max[i] - scene_min[i] for i in range(3))
            distance = max(scene_size * 2, 5.0)

            az = math.radians(45)
            el = math.radians(25)
            cam_x = center[0] + distance * math.cos(el) * math.cos(az)
            cam_y = center[1] - distance * math.cos(el) * math.sin(az)
            cam_z = center[2] + distance * math.sin(el)

            print(f"[Env] 自动初始化（场景中心）: cam=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), "
                  f"look=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            return [cam_x, cam_y, cam_z], center

        # 兜底
        return [5.0, -5.0, 3.0], [0.0, 0.0, 0.0]

    # ==============================================================
    #  场景初始化
    # ==============================================================

    def _setup_camera(self):
        """获取或创建相机对象，并设置 TrackTo 约束。"""
        scene = bpy.context.scene

        if self.camera_name in bpy.data.objects:
            obj = bpy.data.objects[self.camera_name]
            if obj.type == "CAMERA":
                self.camera_obj = obj
                self.camera_data = obj.data
                print(f"[Env] 复用已有相机: {self.camera_name}")
            else:
                obj.name = obj.name + "_old"
                print(f"[Env] 对象 '{self.camera_name}' 非相机类型，已重命名")
                self._create_camera()
        else:
            self._create_camera()

        scene.camera = self.camera_obj
        self._setup_look_at_target()

    def _create_camera(self):
        """创建一个新的相机对象并链接到当前场景集合。"""
        cam_data = bpy.data.cameras.new(name=self.camera_name)
        cam_obj = bpy.data.objects.new(name=self.camera_name, object_data=cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        self.camera_obj = cam_obj
        self.camera_data = cam_data
        print(f"[Env] 已创建新相机: {self.camera_name}")

    def _setup_look_at_target(self):
        """
        创建一个不可见的 Empty 对象作为注视目标，
        并为相机添加 TrackTo 约束指向该 Empty。

        这样相机会始终自动朝向 look_at 点，
        而无需手动计算旋转矩阵。
        """
        empty_name = "AgentLookAtTarget"

        # 查找或创建 Empty
        if empty_name in bpy.data.objects:
            self._look_at_empty = bpy.data.objects[empty_name]
            print(f"[Env] 复用注视目标: {empty_name}")
        else:
            empty = bpy.data.objects.new(empty_name, None)
            empty.empty_display_type = "PLAIN_AXES"
            empty.empty_display_size = 0.1
            bpy.context.scene.collection.objects.link(empty)
            self._look_at_empty = empty
            print(f"[Env] 已创建注视目标: {empty_name}")

        # 隐藏：不参与渲染
        self._look_at_empty.hide_render = True

        # 设置 TrackTo 约束
        constraint_name = "AgentTrackTo"
        existing = self.camera_obj.constraints.get(constraint_name)
        if existing is not None:
            existing.target = self._look_at_empty
            print(f"[Env] 复用 TrackTo 约束 -> {empty_name}")
        else:
            track = self.camera_obj.constraints.new(type="TRACK_TO")
            track.name = constraint_name
            track.target = self._look_at_empty
            track.track_axis = "TRACK_NEGATIVE_Z"  # 相机朝 -Z 看
            track.up_axis = "UP_Y"                  # 上方向 Y
            print(f"[Env] 已添加 TrackTo 约束 -> {empty_name}")

    # ==============================================================
    #  状态应用 —— 将当前参数同步到 Blender 场景
    # ==============================================================

    def _apply_state(self):
        """
        将当前 camera_position、look_at、焦距、光圈
        同步写入 Blender 对象。

        调用链：
            1. 设置相机位置
            2. 设置 Empty（注视目标）位置
            3. TrackTo 约束自动处理朝向
            4. 设置焦距、光圈、景深
            5. 更新依赖图
        """
        # ── 1. 更新相机位置 ──
        self.camera_obj.location = Vector(self.camera_position)

        # ── 2. 更新注视目标位置 ──
        self._look_at_empty.location = Vector(self.look_at)

        # ── 3. 更新镜头参数 ──
        self._apply_lens_settings()

        # ── 4. 强制更新依赖图，使约束生效 ──
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

    def _apply_lens_settings(self):
        """
        将当前焦距与光圈参数写入 Blender 相机数据。
        启用景深 (DoF)，对焦距离 = 相机到注视点的距离。
        """
        cam = self.camera_data

        # 焦距
        cam.lens = self.focal_length

        # 景深
        cam.dof.use_dof = True
        # 对焦距离：相机到注视点
        cam_vec = Vector(self.camera_position)
        look_vec = Vector(self.look_at)
        cam.dof.focus_distance = (look_vec - cam_vec).length

        # 光圈 f-stop
        cam.dof.aperture_fstop = self.aperture

    @staticmethod
    def _normalize_aspect_ratio(value) -> str:
        txt = str(value or "").strip().replace(" ", "")
        txt = txt.replace("：", ":").replace("/", ":")
        if ":" not in txt:
            return "1:1"
        a, b = txt.split(":", 1)
        try:
            av = float(a)
            bv = float(b)
            if av <= 0 or bv <= 0:
                return "1:1"
            return f"{int(round(av))}:{int(round(bv))}"
        except Exception:
            return "1:1"

    @staticmethod
    def _available_render_engines() -> set:
        available = {
            item.identifier
            for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        }
        try:
            current_engine = str(bpy.context.scene.render.engine or "").strip().upper().replace(" ", "_")
            if current_engine:
                available.add(current_engine)
        except Exception:
            pass
        return available

    @staticmethod
    def _can_assign_render_engine(engine: str | None) -> bool:
        txt = str(engine or "").strip().upper().replace(" ", "_")
        if not txt:
            return False
        scene = getattr(bpy.context, "scene", None)
        render = getattr(scene, "render", None) if scene else None
        if render is None:
            return False

        orig_engine = str(render.engine or "")
        try:
            if orig_engine == txt:
                return True
            render.engine = txt
            return str(render.engine or "") == txt
        except Exception:
            return False
        finally:
            try:
                render.engine = orig_engine
            except Exception:
                pass

    @classmethod
    def _resolve_render_engine(cls, engine: str | None, fallback: str = "CYCLES") -> str:
        available = cls._available_render_engines()
        txt = str(engine or "").strip().upper().replace(" ", "_")
        fallback_txt = str(fallback or "CYCLES").strip().upper().replace(" ", "_")
        if txt in {"", "AUTO"}:
            txt = fallback_txt
        aliases = {
            "EEVEE": "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available else "BLENDER_EEVEE",
            "EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
            "BLENDER_EEVEE": "BLENDER_EEVEE",
            "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
            "CYCLES": "CYCLES",
            "WORKBENCH": "BLENDER_WORKBENCH",
            "BLENDER_WORKBENCH": "BLENDER_WORKBENCH",
        }
        resolved = aliases.get(txt, txt)
        if resolved in available or cls._can_assign_render_engine(resolved):
            return resolved
        if fallback_txt in available or cls._can_assign_render_engine(fallback_txt):
            if txt not in {"", "AUTO"} and resolved != fallback_txt:
                print(f"[Env] Warning: requested render engine '{txt}' unavailable, falling back to '{fallback_txt}'.")
            return fallback_txt
        if "CYCLES" in available or cls._can_assign_render_engine("CYCLES"):
            if txt not in {"", "AUTO"} and resolved != "CYCLES":
                print(f"[Env] Warning: requested render engine '{txt}' unavailable, falling back to 'CYCLES'.")
            return "CYCLES"
        for candidate in sorted(available):
            if cls._can_assign_render_engine(candidate):
                if txt not in {"", "AUTO"} and resolved != candidate:
                    print(f"[Env] Warning: requested render engine '{txt}' unavailable, falling back to '{candidate}'.")
                return candidate
        return resolved

    @classmethod
    def _compute_capped_resolution(
        cls,
        ratio: str,
        max_pixels: int,
        fallback_resolution: tuple,
    ) -> tuple:
        max_pixels = max(1, int(max_pixels))
        norm_ratio = cls._normalize_aspect_ratio(ratio)
        fallback_w = max(1, int(fallback_resolution[0]))
        fallback_h = max(1, int(fallback_resolution[1]))

        try:
            rw, rh = norm_ratio.split(":", 1)
            rw = float(rw)
            rh = float(rh)
            if rw <= 0 or rh <= 0:
                raise ValueError("invalid ratio")
            ratio_val = rw / rh
        except Exception:
            ratio_val = fallback_w / fallback_h

        # 在给定像素上限下，尽量填满并保持目标比例
        h = int(math.sqrt(max_pixels / ratio_val))
        w = int(h * ratio_val)
        w = max(1, w)
        h = max(1, h)

        # 避免奇怪分辨率，收敛到偶数
        w -= w % 2
        h -= h % 2
        w = max(2, w)
        h = max(2, h)

        # 防止偶数收敛后超过上限
        while w * h > max_pixels and w > 2 and h > 2:
            w -= 2
            h = int(w / ratio_val)
            h -= h % 2
            h = max(2, h)

        return int(w), int(h)

    # ==============================================================
    #  参数安全检查
    # ==============================================================

    def _clamp_params(self):
        """
        安全性检查：
        - 防止相机与注视点完全重合（会导致 TrackTo 失效）
        - 焦距和光圈裁剪到合法范围
        """
        cam_vec = Vector(self.camera_position)
        look_vec = Vector(self.look_at)

        if (cam_vec - look_vec).length < 0.01:
            # 微调相机位置，避免与注视点重合
            self.camera_position[1] -= 0.5
            print("[Env] ⚠ 相机与注视点过近，已自动微调相机位置")

        # 焦距合理范围
        self.focal_length = max(1.0, min(1200.0, self.focal_length))
        # 光圈合理范围
        self.aperture = max(0.5, min(128.0, self.aperture))

    # ==============================================================
    #  场景信息查询
    # ==============================================================

    def get_scene_info(self) -> dict:
        """
        获取场景中所有可见物体的信息（名称、类型、位置、包围盒）。

        智能体可以利用这些信息来了解场景布局，
        从而做出更合理的构图和相机放置决策。

        Returns
        -------
        dict
            {
                "objects": [
                    {
                        "name": str,
                        "type": str,           # "MESH" / "LIGHT" / "EMPTY" / ...
                        "center": [x, y, z],   # 仅 MESH
                        "bbox_min": [x, y, z],  # 仅 MESH
                        "bbox_max": [x, y, z],  # 仅 MESH
                        "size": [sx, sy, sz],   # 仅 MESH
                        "light_type": str,      # 仅 LIGHT
                    },
                    ...
                ],
                "scene_bbox": {
                    "min": [x, y, z],
                    "max": [x, y, z],
                    "center": [x, y, z],
                    "size": [sx, sy, sz],
                },
            }
        """
        # 排除环境自身创建的辅助对象
        skip_names = {self.camera_name, "AgentLookAtTarget"}

        objects_info = []
        all_mins = []
        all_maxs = []

        for obj in bpy.data.objects:
            if obj.name in skip_names:
                continue
            if obj.type not in ("MESH", "LIGHT", "EMPTY", "CURVE", "SURFACE"):
                continue
            if not obj.visible_get():
                continue

            info = {
                "name": obj.name,
                "type": obj.type,
            }

            if obj.type == "MESH":
                bbox = self._get_object_bbox(obj)
                info["center"] = bbox["center"]
                info["bbox_min"] = bbox["min"]
                info["bbox_max"] = bbox["max"]
                info["size"] = bbox["size"]
                all_mins.append(bbox["min"])
                all_maxs.append(bbox["max"])
            elif obj.type == "LIGHT":
                pos = obj.matrix_world.translation
                info["position"] = [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)]
                info["light_type"] = obj.data.type  # POINT, SUN, SPOT, AREA
            else:
                pos = obj.matrix_world.translation
                info["position"] = [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)]

            objects_info.append(info)

        # 场景整体包围盒（仅基于 MESH 物体）
        scene_bbox = {}
        if all_mins:
            s_min = [round(min(m[i] for m in all_mins), 3) for i in range(3)]
            s_max = [round(max(m[i] for m in all_maxs), 3) for i in range(3)]
            scene_bbox = {
                "min": s_min,
                "max": s_max,
                "center": [round((s_min[i] + s_max[i]) / 2, 3) for i in range(3)],
                "size": [round(s_max[i] - s_min[i], 3) for i in range(3)],
            }

        return {
            "objects": objects_info,
            "scene_bbox": scene_bbox,
        }

    def get_scene_info_text(self) -> str:
        """
        将场景信息格式化为人类可读的文本，适合嵌入 VLM 系统提示。

        Returns
        -------
        str
            格式化后的场景信息文本。
        """
        info = self.get_scene_info()
        lines = []

        mesh_objects = [o for o in info["objects"] if o["type"] == "MESH"]
        light_objects = [o for o in info["objects"] if o["type"] == "LIGHT"]

        if mesh_objects:
            lines.append("### 可拍摄物体（网格）")
            for obj in mesh_objects:
                c = obj.get("center", [0, 0, 0])
                s = obj.get("size", [0, 0, 0])
                lines.append(
                    f"- **{obj['name']}**: "
                    f"中心=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
                    f"尺寸≈({s[0]:.2f} × {s[1]:.2f} × {s[2]:.2f})"
                )

        if light_objects:
            lines.append("\n### 灯光")
            for obj in light_objects:
                p = obj.get("position", [0, 0, 0])
                lines.append(
                    f"- {obj['name']} ({obj.get('light_type', '?')}): "
                    f"位置=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                )

        if info.get("scene_bbox"):
            sb = info["scene_bbox"]
            lines.append("\n### 场景空间范围")
            lines.append(f"- 最小角: ({sb['min'][0]:.2f}, {sb['min'][1]:.2f}, {sb['min'][2]:.2f})")
            lines.append(f"- 最大角: ({sb['max'][0]:.2f}, {sb['max'][1]:.2f}, {sb['max'][2]:.2f})")
            lines.append(f"- 中心:   ({sb['center'][0]:.2f}, {sb['center'][1]:.2f}, {sb['center'][2]:.2f})")
            lines.append(f"- 尺寸:   {sb['size'][0]:.2f} × {sb['size'][1]:.2f} × {sb['size'][2]:.2f}")

        return "\n".join(lines) if lines else "（无场景物体信息）"

    def get_scene_topology_text(self, max_objects: int = 10, max_relations_per_object: int = 2) -> str:
        """生成简洁的场景拓扑关系文本，供上层作为全局上下文使用。"""
        info = self.get_scene_info()
        mesh_objects = [o for o in info["objects"] if o["type"] == "MESH" and o.get("center")]
        if len(mesh_objects) < 2:
            return "### 场景拓扑\n- 可用关系不足。"

        focus_name = self._ref_name if self._ref_name else None
        if focus_name:
            focus_obj = next((o for o in mesh_objects if o["name"] == focus_name), None)
        else:
            focus_obj = None

        if focus_obj is not None:
            fx, fy, fz = focus_obj["center"]
            mesh_objects = sorted(
                mesh_objects,
                key=lambda o: (
                    (o["center"][0] - fx) ** 2
                    + (o["center"][1] - fy) ** 2
                    + (o["center"][2] - fz) ** 2
                ),
            )

        mesh_objects = mesh_objects[: max(2, int(max_objects))]
        lines = ["### 场景拓扑"]

        def relation_line(src: dict, dst: dict) -> str:
            dx = dst["center"][0] - src["center"][0]
            dy = dst["center"][1] - src["center"][1]
            dz = dst["center"][2] - src["center"][2]
            adx, ady, adz = abs(dx), abs(dy), abs(dz)
            if adx >= ady and adx >= adz:
                axis = "右侧" if dx >= 0 else "左侧"
                dist = adx
            elif ady >= adx and ady >= adz:
                axis = "前方" if dy >= 0 else "后方"
                dist = ady
            else:
                axis = "上方" if dz >= 0 else "下方"
                dist = adz
            return f"- {src['name']} 的 {axis} 约 {dist:.2f} 米处是 {dst['name']}"

        for src in mesh_objects:
            others = [o for o in mesh_objects if o["name"] != src["name"]]
            others = sorted(
                others,
                key=lambda o: (
                    (o["center"][0] - src["center"][0]) ** 2
                    + (o["center"][1] - src["center"][1]) ** 2
                    + (o["center"][2] - src["center"][2]) ** 2
                ),
            )
            for dst in others[: max(1, int(max_relations_per_object))]:
                lines.append(relation_line(src, dst))

        return "\n".join(lines)

    def render_scout_views(
        self,
        output_dir: str,
        focus_object: str | None = None,
        num_orbit_views: int = 4,
    ) -> list[dict]:
        """渲染少量全局 scout views，作为上层候选提议的视觉锚点。"""
        info = self.get_scene_info()
        bbox = info.get("scene_bbox") or {}
        scene_center = bbox.get("center", [0.0, 0.0, 0.0])
        scene_size = bbox.get("size", [6.0, 6.0, 4.0])
        scene_scale = max(max(scene_size), 1.0)

        focus_name = focus_object or self._ref_name
        focus_center = self.get_object_center(focus_name) if focus_name else None
        if focus_center is None:
            focus_center = list(scene_center)

        abs_dir = os.path.abspath(output_dir)
        os.makedirs(abs_dir, exist_ok=True)

        prev_state = self.get_observation()
        saved_cam = list(self.camera_position)
        saved_look = list(self.look_at)
        saved_focal = float(self.focal_length)
        saved_aperture = float(self.aperture)
        saved_ratio = self.aspect_ratio

        orbit_count = max(2, int(num_orbit_views))
        radius = max(scene_scale * 0.45, 3.0)
        heights = [focus_center[2] + max(scene_scale * 0.12, 0.8), focus_center[2] + max(scene_scale * 0.45, 2.5)]

        configs = []
        for idx in range(orbit_count):
            angle = (2.0 * math.pi * idx) / orbit_count
            cam = [
                focus_center[0] + radius * math.cos(angle),
                focus_center[1] + radius * math.sin(angle),
                heights[0],
            ]
            configs.append({"label": f"orbit_{idx:02d}", "camera_position": cam, "look_at": list(focus_center)})

        configs.append(
            {
                "label": "overview_high",
                "camera_position": [focus_center[0], focus_center[1] - radius * 0.75, heights[1]],
                "look_at": list(focus_center),
            }
        )

        views = []
        try:
            for cfg in configs:
                self.step(
                    {
                        "camera_position": cfg["camera_position"],
                        "look_at": cfg["look_at"],
                        "focal_length": max(saved_focal, 35.0),
                        "aperture": max(saved_aperture, 4.0),
                        "aspect_ratio": saved_ratio,
                    }
                )
                img_path = os.path.join(abs_dir, f"{cfg['label']}.jpg")
                self.render_preview(img_path)
                views.append(
                    {
                        "label": cfg["label"],
                        "image_path": os.path.abspath(img_path),
                        "camera_position": [round(v, 4) for v in self.camera_position],
                        "look_at": [round(v, 4) for v in self.look_at],
                    }
                )
        finally:
            self.camera_position = saved_cam
            self.look_at = saved_look
            self.focal_length = saved_focal
            self.aperture = saved_aperture
            self.aspect_ratio = saved_ratio
            self.preview_resolution = self._compute_capped_resolution(
                ratio=self.aspect_ratio,
                max_pixels=self.preview_max_pixels,
                fallback_resolution=self.preview_resolution_base,
            )
            self._clamp_params()
            self._apply_state()
            self._print_state(prev_state)

        return views

    def get_object_center(self, object_name: str) -> list | None:
        """按名称获取物体中心坐标。找不到时返回 None。"""
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            return None
        return self._get_object_center(obj)

    def clamp_point_to_scene_bbox(self, point: list) -> list:
        """将点夹紧到场景包围盒内，防止候选点跑到场景外。"""
        scene_info = self.get_scene_info()
        bbox = scene_info.get("scene_bbox")
        if not bbox:
            return list(point)

        p = list(point)
        for i in range(3):
            p[i] = max(bbox["min"][i], min(bbox["max"][i], p[i]))
        return p

    def auto_frame_look_at(
        self,
        camera_position: list,
        subject_center: list,
        composition_rule: str = "center",
    ) -> list:
        """
        根据构图规则自动微调 look_at。
        思路：让相机看向主体附近偏移点，从而将主体压到目标构图位置。
        """
        cam = Vector(camera_position)
        subject = Vector(subject_center)
        forward = subject - cam
        dist = max(forward.length, 1e-5)
        forward.normalize()

        world_up = Vector((0.0, 0.0, 1.0))
        right = forward.cross(world_up)
        if right.length < 1e-5:
            right = Vector((1.0, 0.0, 0.0))
        else:
            right.normalize()
        up = right.cross(forward)
        if up.length < 1e-5:
            up = Vector((0.0, 0.0, 1.0))
        else:
            up.normalize()

        # 偏移比例与距离挂钩，避免远景偏移太小
        offset_scale = max(0.15 * dist, 0.15)
        dx, dy = 0.0, 0.0
        rule = (composition_rule or "center").lower()

        if "left" in rule:
            dx = 1.0
        elif "right" in rule:
            dx = -1.0

        if "top" in rule:
            dy = -1.0
        elif "bottom" in rule:
            dy = 1.0

        # 三分法默认偏到左三分之一，除非显式指定
        if rule in {"rule_of_thirds", "thirds", "rule_of_thirds_left"}:
            dx = 1.0

        look = subject + right * (dx * offset_scale) + up * (dy * offset_scale)
        return [round(look.x, 4), round(look.y, 4), round(look.z, 4)]

    def is_target_visible_from(self, camera_position: list, target_name: str) -> bool:
        """
        射线检测目标是否可见。
        若首个命中对象不是目标，则视为被遮挡。
        """
        target = bpy.data.objects.get(target_name)
        if target is None:
            return False

        origin = Vector(camera_position)
        target_center = Vector(self._get_object_center(target))
        direction = target_center - origin
        dist = direction.length
        if dist < 1e-4:
            return True
        direction.normalize()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        hit, _loc, _normal, _idx, hit_obj, _matrix = bpy.context.scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=dist + 1e-3,
        )
        if not hit:
            return False
        return hit_obj is not None and hit_obj.name == target_name

    def compute_visibility_candidates(
        self,
        primary_subject: str,
        context_objects: list | None = None,
        grid_step: float = 1.0,
        camera_height_range: tuple = (0.6, 2.0),
        max_points: int = 300,
    ) -> list:
        """
        在场景包围盒内采样候选机位，并做可见性过滤。
        返回通过过滤的相机点列表。
        """
        context_objects = context_objects or []
        scene_info = self.get_scene_info()
        bbox = scene_info.get("scene_bbox")
        if not bbox:
            return []

        x_min, y_min, z_min = bbox["min"]
        x_max, y_max, z_max = bbox["max"]

        z_lo = max(z_min + 0.05, camera_height_range[0])
        z_hi = min(z_max + 2.0, camera_height_range[1])
        if z_hi < z_lo:
            z_hi = z_lo

        heights = [z_lo]
        if z_hi - z_lo > 0.25:
            heights.append((z_lo + z_hi) * 0.5)

        required = [primary_subject] + [n for n in context_objects if n != primary_subject]

        candidates = []
        x = x_min
        while x <= x_max and len(candidates) < max_points:
            y = y_min
            while y <= y_max and len(candidates) < max_points:
                for z in heights:
                    cam = [round(x, 4), round(y, 4), round(z, 4)]
                    visible_all = True
                    for name in required:
                        if not self.is_target_visible_from(cam, name):
                            visible_all = False
                            break
                    if visible_all:
                        candidates.append(cam)
                        if len(candidates) >= max_points:
                            break
                y += grid_step
            x += grid_step

        return candidates

    def evaluate_rule_scores(
        self,
        camera_position: list,
        look_at: list,
        primary_subject: str,
        composition_rule: str = "center",
    ) -> dict:
        """
        计算 M1/M2 规则评分：
        - M1: 主体是否入画且在要求象限（硬门控）
        - M2: 主体到目标构图点的距离得分
        """
        subject = bpy.data.objects.get(primary_subject)
        if subject is None:
            return {"m1": 0.0, "m2": 0.0, "uv": [0.5, 0.5]}

        # 暂存当前状态
        prev_cam = list(self.camera_position)
        prev_look = list(self.look_at)

        try:
            self.camera_position = list(camera_position)
            self.look_at = list(look_at)
            self._clamp_params()
            self._apply_state()

            center = Vector(self._get_object_center(subject))
            uv = world_to_camera_view(bpy.context.scene, self.camera_obj, center)
            u, v = float(uv.x), float(uv.y)

            in_frame = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
            rule = (composition_rule or "center").lower()

            target_u, target_v = 0.5, 0.5
            if "left" in rule:
                target_u = 1.0 / 3.0
            elif "right" in rule:
                target_u = 2.0 / 3.0

            if "top" in rule:
                target_v = 2.0 / 3.0
            elif "bottom" in rule:
                target_v = 1.0 / 3.0

            if rule in {"rule_of_thirds", "thirds", "rule_of_thirds_left"}:
                target_u, target_v = 1.0 / 3.0, 0.5

            # M1：主体必须入画，且在目标位置同侧（粗粒度）
            if not in_frame:
                m1 = 0.0
            else:
                m1 = 1.0
                if "left" in rule and not (u <= 0.5):
                    m1 = 0.0
                if "right" in rule and not (u >= 0.5):
                    m1 = 0.0
                if "top" in rule and not (v >= 0.5):
                    m1 = 0.0
                if "bottom" in rule and not (v <= 0.5):
                    m1 = 0.0

            # M2：到目标构图点的归一化距离得分
            dist = math.sqrt((u - target_u) ** 2 + (v - target_v) ** 2)
            m2 = max(0.0, 1.0 - dist / 0.45)

            return {
                "m1": round(m1, 4),
                "m2": round(m2, 4),
                "uv": [round(u, 4), round(v, 4)],
                "target_uv": [round(target_u, 4), round(target_v, 4)],
            }
        finally:
            self.camera_position = prev_cam
            self.look_at = prev_look
            self._clamp_params()
            self._apply_state()

    # ==============================================================
    #  公开接口
    # ==============================================================

    def reset(self) -> dict:
        """
        将环境重置为初始状态。

        Returns
        -------
        dict
            重置后的观测信息（当前相机状态快照）。
        """
        self.camera_position = list(self._init_camera_position)
        self.look_at = list(self._init_look_at)
        self.focal_length = self._init_focal_length
        self.aperture = self._init_aperture
        self.aspect_ratio = self._init_aspect_ratio
        self.preview_resolution = self._compute_capped_resolution(
            ratio=self.aspect_ratio,
            max_pixels=self.preview_max_pixels,
            fallback_resolution=self.preview_resolution_base,
        )

        self._clamp_params()
        self._apply_state()

        print("[Env] 环境已重置到初始状态。")
        return self.get_observation()

    def step(self, action_dict: dict) -> dict:
        """
        执行一步动作：根据 action_dict 更新相机状态。

        支持绝对定位（直接覆盖）和相对增量（叠加到当前值），
        两者可混合使用。绝对值优先于增量。

        Parameters
        ----------
        action_dict : dict
            可包含以下键（均为可选，缺失则保持不变）：

            绝对位置（直接覆盖）：
            - 'camera_position' : [x, y, z] — 相机世界坐标
            - 'look_at'         : [x, y, z] — 注视点世界坐标

            相对增量（叠加到当前值）：
            - 'd_camera_position' : [dx, dy, dz] — 相机位置增量
            - 'd_look_at'         : [dx, dy, dz] — 注视点增量

            镜头参数（绝对值覆盖）：
            - 'focal_length' : float — 焦距 (mm)
            - 'aperture'     : float — 光圈 f 值
            - 'aspect_ratio' : str   — 画幅比例，例如 "16:9"

        Returns
        -------
        dict
            执行动作后的观测信息。

        Examples
        --------
        >>> env.step({'camera_position': [3, -4, 2], 'look_at': [0, 0, 1]})
        >>> env.step({'d_camera_position': [0, 0, 1], 'focal_length': 85})
        """
        # ── 绝对位置覆盖（优先） ──
        cam_pos = action_dict.get("camera_position")
        if cam_pos is not None and isinstance(cam_pos, (list, tuple)) and len(cam_pos) >= 3:
            self.camera_position = [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])]

        look = action_dict.get("look_at")
        if look is not None and isinstance(look, (list, tuple)) and len(look) >= 3:
            self.look_at = [float(look[0]), float(look[1]), float(look[2])]

        # ── 相对增量 ──
        d_cam = action_dict.get("d_camera_position")
        if d_cam is not None and isinstance(d_cam, (list, tuple)) and len(d_cam) >= 3:
            for i in range(3):
                self.camera_position[i] += float(d_cam[i])

        d_look = action_dict.get("d_look_at")
        if d_look is not None and isinstance(d_look, (list, tuple)) and len(d_look) >= 3:
            for i in range(3):
                self.look_at[i] += float(d_look[i])

        # ── 镜头参数覆盖 ──
        if "focal_length" in action_dict:
            self.focal_length = float(action_dict["focal_length"])
        if "aperture" in action_dict:
            self.aperture = float(action_dict["aperture"])
        if "aspect_ratio" in action_dict:
            self.aspect_ratio = self._normalize_aspect_ratio(action_dict["aspect_ratio"])
            self.preview_resolution = self._compute_capped_resolution(
                ratio=self.aspect_ratio,
                max_pixels=self.preview_max_pixels,
                fallback_resolution=self.preview_resolution_base,
            )

        # ── 安全检查 ──
        self._clamp_params()

        # ── 同步到 Blender ──
        self._apply_state()

        obs = self.get_observation()
        self._print_state(obs)
        return obs

    def get_observation(self) -> dict:
        """
        返回当前环境状态的完整快照。

        Returns
        -------
        dict
            {
                'camera_position':          [x, y, z],
                'look_at':                  [x, y, z],
                'focal_length':             float,
                'aperture':                 float,
                'camera_to_target_distance': float,   # 相机到注视点的距离（米）
            }
        """
        cam = self.camera_position
        look = self.look_at
        dist = math.sqrt(sum((cam[i] - look[i]) ** 2 for i in range(3)))

        return {
            "camera_position": [round(v, 4) for v in cam],
            "look_at": [round(v, 4) for v in look],
            "focal_length": round(self.focal_length, 2),
            "aperture": round(self.aperture, 2),
            "aspect_ratio": self.aspect_ratio,
            "preview_resolution": [int(self.preview_resolution[0]), int(self.preview_resolution[1])],
            "camera_to_target_distance": round(dist, 4),
        }

    # ==============================================================
    #  渲染
    # ==============================================================

    def render_preview(self, output_path: str = "output/preview.jpg", engine: str = None) -> str:
        """
        使用 Eevee 引擎进行低分辨率快速渲染，生成当前视角的预览图。

        Parameters
        ----------
        output_path : str
            输出图像文件路径，默认 'output/preview.jpg'。

        Returns
        -------
        str
            渲染结果的绝对路径。
        """
        scene = bpy.context.scene
        render = scene.render

        # ── 暂存原始渲染设置 ──
        orig_engine = render.engine
        orig_res_x = render.resolution_x
        orig_res_y = render.resolution_y
        orig_res_pct = render.resolution_percentage
        orig_filepath = render.filepath
        orig_file_format = render.image_settings.file_format
        orig_color_mode = render.image_settings.color_mode
        orig_cycles_samples = getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None
        orig_cycles_denoising = getattr(scene.cycles, "use_denoising", None) if hasattr(scene, "cycles") else None
        orig_eevee_samples = getattr(scene.eevee, "taa_render_samples", None) if hasattr(scene, "eevee") else None

        try:
            # ── 配置 Eevee 快速渲染 ──
            preview_engine = self._resolve_render_engine(engine or self.preview_engine, fallback=orig_engine)
            render.engine = preview_engine

            self.preview_resolution = self._compute_capped_resolution(
                ratio=self.aspect_ratio,
                max_pixels=self.preview_max_pixels,
                fallback_resolution=self.preview_resolution_base,
            )
            render.resolution_x = self.preview_resolution[0]
            render.resolution_y = self.preview_resolution[1]
            render.resolution_percentage = 100

            # ── 输出格式 ──
            abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            ext = os.path.splitext(abs_path)[1].lower()
            fmt_map = {
                ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
                ".bmp": "BMP", ".tiff": "TIFF", ".tif": "TIFF",
                ".exr": "OPEN_EXR",
            }
            render.image_settings.file_format = fmt_map.get(ext, "PNG")
            render.image_settings.color_mode = "RGBA" if ext == ".png" else "RGB"
            render.filepath = abs_path
            preview_samples = max(1, min(int(self.preview_samples), 64))
            if render.engine == "CYCLES" and hasattr(scene, "cycles"):
                scene.cycles.samples = preview_samples
                if hasattr(scene.cycles, "use_denoising"):
                    scene.cycles.use_denoising = False
            elif hasattr(scene, "eevee"):
                scene.eevee.taa_render_samples = preview_samples

            # ── 确保活动相机正确 ──
            scene.camera = self.camera_obj

            # ── 执行渲染 ──
            print(
                f"[Env] 开始预览渲染 engine={preview_engine} "
                f"({self.preview_resolution[0]}x{self.preview_resolution[1]})..."
            )
            bpy.ops.render.render(write_still=True)
            print(f"[Env] 预览已保存: {abs_path}")

            return abs_path

        finally:
            # ── 恢复原始渲染设置 ──
            render.engine = orig_engine
            render.resolution_x = orig_res_x
            render.resolution_y = orig_res_y
            render.resolution_percentage = orig_res_pct
            render.filepath = orig_filepath
            render.image_settings.file_format = orig_file_format
            render.image_settings.color_mode = orig_color_mode
            if orig_cycles_samples is not None and hasattr(scene, "cycles"):
                scene.cycles.samples = orig_cycles_samples
            if orig_cycles_denoising is not None and hasattr(scene, "cycles"):
                scene.cycles.use_denoising = orig_cycles_denoising
            if orig_eevee_samples is not None and hasattr(scene, "eevee"):
                scene.eevee.taa_render_samples = orig_eevee_samples

    def render_final(self, output_path: str = "output/final.png",
                     engine: str = "CYCLES",
                     resolution: tuple = (1920, 1080),
                     samples: int = 128) -> str:
        """
        使用指定引擎进行高质量最终渲染。

        Parameters
        ----------
        output_path : str
            输出路径。
        engine : str
            渲染引擎（'CYCLES' / 'BLENDER_EEVEE_NEXT' 等）。
        resolution : tuple[int, int]
            输出分辨率 (宽, 高)。
        samples : int
            采样数。

        Returns
        -------
        str
            渲染结果的绝对路径。
        """
        scene = bpy.context.scene
        render = scene.render

        orig_engine = render.engine
        orig_res_x = render.resolution_x
        orig_res_y = render.resolution_y
        orig_res_pct = render.resolution_percentage
        orig_filepath = render.filepath
        orig_file_format = render.image_settings.file_format

        try:
            final_engine = self._resolve_render_engine(engine or self.final_engine, fallback=orig_engine)
            render.engine = final_engine
            render.resolution_x = resolution[0]
            render.resolution_y = resolution[1]
            render.resolution_percentage = 100

            final_samples = max(1, min(int(samples), 256))

            if final_engine == "CYCLES":
                scene.cycles.samples = final_samples
                scene.cycles.device = "GPU"

                # 尝试自动启用 GPU 设备
                prefs = bpy.context.preferences.addons.get("cycles")
                if prefs:
                    cprefs = prefs.preferences
                    for dev_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "OPENCL"):
                        try:
                            cprefs.compute_device_type = dev_type
                            cprefs.get_devices()
                            if any(getattr(d, "type", "").upper() != "CPU" for d in cprefs.devices):
                                break
                        except Exception:
                            continue
                    for device in cprefs.devices:
                        dtype = str(getattr(device, "type", "")).upper()
                        if dtype == "CPU":
                            device.use = False
                        else:
                            device.use = True
            elif hasattr(scene, "eevee"):
                scene.eevee.taa_render_samples = final_samples

            abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            ext = os.path.splitext(abs_path)[1].lower()
            fmt_map = {
                ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
                ".bmp": "BMP", ".tiff": "TIFF", ".exr": "OPEN_EXR",
            }
            render.image_settings.file_format = fmt_map.get(ext, "PNG")
            render.filepath = abs_path

            scene.camera = self.camera_obj

            print(f"[Env] 开始最终渲染 "
                f"({final_engine}, {resolution[0]}x{resolution[1]}, {final_samples} samples)...")
            bpy.ops.render.render(write_still=True)
            print(f"[Env] 最终渲染已保存: {abs_path}")

            return abs_path

        finally:
            render.engine = orig_engine
            render.resolution_x = orig_res_x
            render.resolution_y = orig_res_y
            render.resolution_percentage = orig_res_pct
            render.filepath = orig_filepath
            render.image_settings.file_format = orig_file_format

    # ==============================================================
    #  辅助方法 —— 物体信息查询
    # ==============================================================

    @staticmethod
    def _get_object_center(obj) -> list:
        """
        获取物体的世界坐标中心点（通过 bounding box 8 角点求平均）。

        Parameters
        ----------
        obj : bpy.types.Object

        Returns
        -------
        list[float]
            [x, y, z]
        """
        try:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            center = sum(corners, Vector((0, 0, 0))) / 8
            return [round(center.x, 3), round(center.y, 3), round(center.z, 3)]
        except Exception:
            loc = obj.matrix_world.translation
            return [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)]

    @staticmethod
    def _get_object_bbox(obj) -> dict:
        """
        获取物体的世界坐标轴对齐包围盒 (AABB)。

        Parameters
        ----------
        obj : bpy.types.Object

        Returns
        -------
        dict
            {"min": [x,y,z], "max": [x,y,z], "center": [x,y,z], "size": [sx,sy,sz]}
        """
        try:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            zs = [c.z for c in corners]

            b_min = [min(xs), min(ys), min(zs)]
            b_max = [max(xs), max(ys), max(zs)]
            b_center = [(b_min[i] + b_max[i]) / 2 for i in range(3)]
            b_size = [b_max[i] - b_min[i] for i in range(3)]

            return {
                "min": [round(v, 3) for v in b_min],
                "max": [round(v, 3) for v in b_max],
                "center": [round(v, 3) for v in b_center],
                "size": [round(v, 3) for v in b_size],
            }
        except Exception:
            loc = obj.matrix_world.translation
            p = [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)]
            return {"min": p[:], "max": p[:], "center": p[:], "size": [0.0, 0.0, 0.0]}

    # ==============================================================
    #  打印 / 描述
    # ==============================================================

    def _print_state(self, obs: dict = None):
        """打印当前状态摘要到控制台。"""
        if obs is None:
            obs = self.get_observation()
        cam = obs["camera_position"]
        look = obs["look_at"]
        print(
            f"[Env] 状态 | "
            f"cam=({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f})  "
            f"look=({look[0]:.2f}, {look[1]:.2f}, {look[2]:.2f})  "
            f"dist={obs['camera_to_target_distance']:.2f}m  "
            f"focal={obs['focal_length']}mm  f/{obs['aperture']}"
        )

    def get_state_description(self) -> str:
        """
        返回自然语言描述的当前相机状态文本。

        Returns
        -------
        str
            中文描述文本。
        """
        obs = self.get_observation()
        cam = obs["camera_position"]
        look = obs["look_at"]
        return (
            f"当前摄影参数：\n"
            f"  - 相机位置: ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f})\n"
            f"  - 注视点:   ({look[0]:.2f}, {look[1]:.2f}, {look[2]:.2f})\n"
            f"  - 相机到注视点距离: {obs['camera_to_target_distance']:.2f} 米\n"
            f"  - 焦距: {obs['focal_length']} mm\n"
            f"  - 光圈: f/{obs['aperture']}"
        )

    def __repr__(self):
        cam = self.camera_position
        look = self.look_at
        return (
            f"BlenderPhotographyEnv("
            f"cam=({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f}), "
            f"look=({look[0]:.1f}, {look[1]:.1f}, {look[2]:.1f}), "
            f"focal={self.focal_length}mm, f/{self.aperture})"
        )


# ══════════════════════════════════════════════════════════════
#  命令行入口（可直接用 blender -b file.blend -P env_blender.py -- ... 测试）
# ══════════════════════════════════════════════════════════════

def _cli_demo():
    """
    命令行演示 / 测试入口。
    用法:
        blender -b scene.blend -P env_blender.py -- --reference Cube
    """
    import argparse as _ap
    import sys as _sys

    argv = _sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = _ap.ArgumentParser(description="BlenderPhotographyEnv CLI 测试")
    parser.add_argument("--reference", type=str, default=None, help="参考物体名称")
    parser.add_argument("--cam-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="初始相机位置")
    parser.add_argument("--look-at", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="初始注视点")
    parser.add_argument("--focal-length", type=float, default=50.0)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--preview", type=str, default=None, help="渲染预览并保存到此路径")
    parser.add_argument("--scene-info", action="store_true", help="打印场景物体信息")
    args = parser.parse_args(argv)

    # ── 创建环境 ──
    env = BlenderPhotographyEnv(
        reference_object=args.reference,
        camera_position=args.cam_pos,
        look_at=args.look_at,
        focal_length=args.focal_length,
        aperture=args.aperture,
    )
    print(env)
    print(env.get_state_description())

    if args.scene_info:
        print("\n" + env.get_scene_info_text())

    if args.preview:
        env.render_preview(args.preview)
    else:
        # 简单的 step 演示
        print("\n[CLI] 执行 step 演示...")
        env.step({"d_camera_position": [1, 0, 0.5], "focal_length": 85})
        env.step({"look_at": [0, 0, 1], "aperture": 1.8})
        print("\n" + env.get_state_description())


if False and __name__ == "__main__":
    _cli_demo()


def _cli_demo():
    import argparse as _ap
    import sys as _sys

    argv = _sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = _ap.ArgumentParser(description="BlenderPhotographyEnv CLI")
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--cam-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--look-at", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--focal-length", type=float, default=50.0)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--aspect-ratio", type=str, default="16:9")
    parser.add_argument("--preview-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--preview-samples", type=int, default=64)
    parser.add_argument("--preview-engine", type=str, default="CYCLES")
    parser.add_argument("--preview", type=str, default=None)
    parser.add_argument("--scene-info", action="store_true")
    args = parser.parse_args(argv)

    env = BlenderPhotographyEnv(
        reference_object=args.reference,
        camera_position=args.cam_pos,
        look_at=args.look_at,
        focal_length=args.focal_length,
        aperture=args.aperture,
        aspect_ratio=args.aspect_ratio,
        preview_max_pixels=args.preview_max_pixels,
        preview_samples=args.preview_samples,
        preview_engine=args.preview_engine,
    )

    if args.scene_info:
        print(env.get_scene_info_text())

    if args.preview:
        env.render_preview(args.preview, engine=args.preview_engine)
        return

    print(env)
    print(env.get_state_description())


if __name__ == "__main__":
    _cli_demo()
