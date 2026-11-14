"""
Blender 命令行渲染摄像机控制脚本

此脚本通过 Blender 的 Python API 控制摄像机参数并执行渲染。
它可以读取 .blend 文件中已有的摄像机，也可以在没有摄像机时自动创建。

用法（由 render.bat / render.ps1 调用，或直接通过 blender 命令行调用）：
    blender -b <blend文件> -P blender_render.py -- [参数]

所有 "--" 之后的参数由此脚本解析。
"""

import bpy
import sys
import os
import math
import argparse
from mathutils import Vector, Euler


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────

def parse_args():
    """解析 '--' 之后的命令行参数"""
    # Blender 会吃掉 '--' 之前的参数，我们只处理之后的
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Blender 命令行渲染 - 摄像机控制脚本"
    )

    # ── 摄像机选择 ──
    parser.add_argument(
        "--camera", "-c",
        type=str, default=None,
        help="指定要使用的摄像机名称。留空则使用场景活动摄像机，若无则自动创建。"
    )

    # ── 摄像机位置 ──
    parser.add_argument(
        "--location", "-l",
        type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="摄像机世界坐标位置，例如: --location 7.5 -6.5 5.3"
    )

    # ── 摄像机旋转（欧拉角，单位：度）──
    parser.add_argument(
        "--rotation", "-r",
        type=float, nargs=3, default=None,
        metavar=("RX", "RY", "RZ"),
        help="摄像机欧拉旋转角度（度），例如: --rotation 63 0 46"
    )

    # ── 摄像机朝向目标点 ──
    parser.add_argument(
        "--look-at",
        type=float, nargs=3, default=None,
        metavar=("TX", "TY", "TZ"),
        help="让摄像机朝向指定目标点，例如: --look-at 0 0 1"
    )

    # ── 焦距 ──
    parser.add_argument(
        "--focal-length", "-f",
        type=float, default=None,
        help="摄像机焦距（mm），默认 50"
    )

    # ── 传感器尺寸 ──
    parser.add_argument(
        "--sensor-width",
        type=float, default=None,
        help="传感器宽度（mm），默认 36"
    )

    # ── 裁剪距离 ──
    parser.add_argument(
        "--clip-start",
        type=float, default=None,
        help="近裁剪距离，默认 0.1"
    )
    parser.add_argument(
        "--clip-end",
        type=float, default=None,
        help="远裁剪距离，默认 1000"
    )

    # ── 景深 ──
    parser.add_argument(
        "--dof", action="store_true",
        help="启用景深"
    )
    parser.add_argument(
        "--dof-distance",
        type=float, default=None,
        help="景深对焦距离"
    )
    parser.add_argument(
        "--fstop",
        type=float, default=None,
        help="光圈 F 值（景深强度），例如 2.8"
    )

    # ── 摄像机类型 ──
    parser.add_argument(
        "--cam-type",
        type=str, choices=["PERSP", "ORTHO", "PANO"],
        default=None,
        help="摄像机投影类型: PERSP(透视) / ORTHO(正交) / PANO(全景)"
    )

    # ── 正交缩放 ──
    parser.add_argument(
        "--ortho-scale",
        type=float, default=None,
        help="正交摄像机缩放比例，默认 6.0"
    )

    # ── 渲染分辨率 ──
    parser.add_argument(
        "--resolution", "--res",
        type=int, nargs=2, default=None,
        metavar=("W", "H"),
        help="渲染分辨率，例如: --resolution 1920 1080"
    )

    # ── 渲染百分比 ──
    parser.add_argument(
        "--resolution-percentage",
        type=int, default=None,
        help="渲染分辨率百分比 (1-100)，默认 100"
    )

    # ── 采样数 ──
    parser.add_argument(
        "--samples", "-s",
        type=int, default=None,
        help="渲染采样数（Cycles / EEVEE）"
    )

    # ── 渲染引擎 ──
    parser.add_argument(
        "--engine", "-e",
        type=str, choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"],
        default=None,
        help="渲染引擎"
    )

    # ── Cycles 设备 ──
    parser.add_argument(
        "--device",
        type=str, choices=["CPU", "GPU"],
        default=None,
        help="Cycles 渲染设备"
    )

    # ── 输出路径 ──
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="输出图像路径（含文件名），例如: --output ./output/render.png"
    )

    # ── 输出格式 ──
    parser.add_argument(
        "--format",
        type=str, choices=["PNG", "JPEG", "BMP", "TIFF", "OPEN_EXR"],
        default=None,
        help="输出图像格式，默认 PNG"
    )

    # ── 帧 ──
    parser.add_argument(
        "--frame",
        type=int, default=None,
        help="渲染指定帧号（静态渲染），默认为当前帧"
    )

    # ── 列出摄像机 ──
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="仅列出场景中所有摄像机信息，不执行渲染"
    )

    # ── 透明背景 ──
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="使用透明背景渲染（仅对 Cycles 和 EEVEE 有效）"
    )

    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def look_at(camera_obj, target_point):
    """
    让摄像机朝向目标点。
    """
    loc = camera_obj.location
    direction = Vector(target_point) - loc
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()


def list_all_cameras():
    """列出场景中所有摄像机的信息"""
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if not cameras:
        print("\n[INFO] 场景中没有摄像机。")
        return

    print(f"\n[INFO] 场景中共有 {len(cameras)} 个摄像机：")
    print("-" * 80)
    for i, cam_obj in enumerate(cameras, 1):
        cam_data = cam_obj.data
        loc = cam_obj.location
        rot = cam_obj.rotation_euler
        rot_deg = (math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z))

        active_mark = " ★ (活动摄像机)" if cam_obj == bpy.context.scene.camera else ""
        print(f"  [{i}] 名称: {cam_obj.name}{active_mark}")
        print(f"      位置:   ({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f})")
        print(f"      旋转:   ({rot_deg[0]:.1f}°, {rot_deg[1]:.1f}°, {rot_deg[2]:.1f}°)")
        print(f"      类型:   {cam_data.type}")
        print(f"      焦距:   {cam_data.lens:.1f} mm")
        print(f"      传感器: {cam_data.sensor_width:.1f} mm")
        print(f"      裁剪:   {cam_data.clip_start} - {cam_data.clip_end}")
        print("-" * 80)


def get_or_create_camera(camera_name=None):
    """
    获取或创建摄像机。
    优先级：
    1. 如果指定了 camera_name，查找该名称的摄像机
    2. 使用场景的活动摄像机
    3. 查找场景中第一个摄像机对象
    4. 都没有则自动创建一个新摄像机
    返回: (camera_object, is_new_created)
    """
    scene = bpy.context.scene

    # 1. 按名称查找
    if camera_name:
        if camera_name in bpy.data.objects:
            obj = bpy.data.objects[camera_name]
            if obj.type == 'CAMERA':
                print(f"[INFO] 使用指定摄像机: {camera_name}")
                return obj, False
            else:
                print(f"[WARN] 对象 '{camera_name}' 不是摄像机，将尝试其他方式。")
        else:
            print(f"[WARN] 未找到名为 '{camera_name}' 的对象，将尝试其他方式。")

    # 2. 场景活动摄像机
    if scene.camera and scene.camera.type == 'CAMERA':
        print(f"[INFO] 使用场景活动摄像机: {scene.camera.name}")
        return scene.camera, False

    # 3. 查找第一个摄像机
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            print(f"[INFO] 使用找到的摄像机: {obj.name}")
            return obj, False

    # 4. 自动创建
    print("[INFO] 场景中没有摄像机，正在自动创建...")
    cam_data = bpy.data.cameras.new(name="AutoCamera")
    cam_obj = bpy.data.objects.new(name="AutoCamera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    # 设置默认位置和旋转
    cam_obj.location = (7.5, -6.5, 5.3)
    cam_obj.rotation_euler = Euler((math.radians(63), 0, math.radians(46)), 'XYZ')

    # 默认参数
    cam_data.lens = 50
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000

    print("[INFO] 已创建摄像机 'AutoCamera'，位置 (7.5, -6.5, 5.3)")
    return cam_obj, True


def apply_camera_settings(cam_obj, args):
    """根据命令行参数设置摄像机属性"""
    cam_data = cam_obj.data

    # 位置
    if args.location is not None:
        cam_obj.location = tuple(args.location)
        print(f"[INFO] 摄像机位置设为: {tuple(args.location)}")

    # 旋转（度→弧度）
    if args.rotation is not None:
        cam_obj.rotation_euler = Euler(
            (math.radians(args.rotation[0]),
             math.radians(args.rotation[1]),
             math.radians(args.rotation[2])),
            'XYZ'
        )
        print(f"[INFO] 摄像机旋转设为: {tuple(args.rotation)}°")

    # 朝向目标点（覆盖 rotation）
    if args.look_at is not None:
        look_at(cam_obj, tuple(args.look_at))
        print(f"[INFO] 摄像机朝向目标点: {tuple(args.look_at)}")

    # 投影类型
    if args.cam_type is not None:
        cam_data.type = args.cam_type
        print(f"[INFO] 摄像机类型设为: {args.cam_type}")

    # 焦距
    if args.focal_length is not None:
        cam_data.lens = args.focal_length
        print(f"[INFO] 焦距设为: {args.focal_length} mm")

    # 传感器宽度
    if args.sensor_width is not None:
        cam_data.sensor_width = args.sensor_width
        print(f"[INFO] 传感器宽度设为: {args.sensor_width} mm")

    # 裁剪距离
    if args.clip_start is not None:
        cam_data.clip_start = args.clip_start
        print(f"[INFO] 近裁剪距离设为: {args.clip_start}")
    if args.clip_end is not None:
        cam_data.clip_end = args.clip_end
        print(f"[INFO] 远裁剪距离设为: {args.clip_end}")

    # 正交缩放
    if args.ortho_scale is not None:
        cam_data.ortho_scale = args.ortho_scale
        print(f"[INFO] 正交缩放设为: {args.ortho_scale}")

    # 景深
    if args.dof:
        cam_data.dof.use_dof = True
        print("[INFO] 已启用景深")
        if args.dof_distance is not None:
            cam_data.dof.focus_distance = args.dof_distance
            print(f"[INFO] 景深对焦距离: {args.dof_distance}")
        if args.fstop is not None:
            cam_data.dof.aperture_fstop = args.fstop
            print(f"[INFO] 光圈 F 值: {args.fstop}")


def apply_render_settings(args):
    """根据命令行参数设置渲染属性"""
    scene = bpy.context.scene
    render = scene.render

    # 渲染引擎
    if args.engine is not None:
        render.engine = args.engine
        print(f"[INFO] 渲染引擎设为: {args.engine}")

    # Cycles GPU
    if args.device is not None and render.engine == 'CYCLES':
        scene.cycles.device = args.device
        if args.device == 'GPU':
            # 尝试启用 GPU 设备
            prefs = bpy.context.preferences.addons.get('cycles')
            if prefs:
                cprefs = prefs.preferences
                cprefs.compute_device_type = 'CUDA'  # 或 OPTIX / OPENCL
                cprefs.get_devices()
                for device in cprefs.devices:
                    device.use = True
        print(f"[INFO] 渲染设备设为: {args.device}")

    # 分辨率
    if args.resolution is not None:
        render.resolution_x = args.resolution[0]
        render.resolution_y = args.resolution[1]
        print(f"[INFO] 分辨率设为: {args.resolution[0]}x{args.resolution[1]}")

    # 分辨率百分比
    if args.resolution_percentage is not None:
        render.resolution_percentage = args.resolution_percentage
        print(f"[INFO] 分辨率百分比设为: {args.resolution_percentage}%")

    # 采样数
    if args.samples is not None:
        if render.engine == 'CYCLES':
            scene.cycles.samples = args.samples
        elif render.engine in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
            scene.eevee.taa_render_samples = args.samples
        print(f"[INFO] 采样数设为: {args.samples}")

    # 输出格式
    if args.format is not None:
        render.image_settings.file_format = args.format
        print(f"[INFO] 输出格式设为: {args.format}")

    # 透明背景
    if args.transparent:
        render.film_transparent = True
        if args.format is None:
            render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGBA'
        print("[INFO] 已启用透明背景渲染")

    # 帧
    if args.frame is not None:
        scene.frame_set(args.frame)
        print(f"[INFO] 渲染帧设为: {args.frame}")


def determine_output_path(args, blend_filepath):
    """确定输出文件路径"""
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # 默认输出到 blend 文件同级的 output 目录
        blend_dir = os.path.dirname(blend_filepath)
        blend_name = os.path.splitext(os.path.basename(blend_filepath))[0]

        output_dir = os.path.join(os.path.dirname(blend_dir), "output")
        os.makedirs(output_dir, exist_ok=True)

        # 确定扩展名
        fmt = bpy.context.scene.render.image_settings.file_format
        ext_map = {
            'PNG': '.png',
            'JPEG': '.jpg',
            'BMP': '.bmp',
            'TIFF': '.tiff',
            'OPEN_EXR': '.exr',
        }
        ext = ext_map.get(fmt, '.png')
        output_path = os.path.join(output_dir, blend_name + ext)

    return output_path


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Blender 命令行渲染 - 摄像机控制脚本")
    print("=" * 60)

    args = parse_args()

    # ── 仅列出摄像机模式 ──
    if args.list_cameras:
        list_all_cameras()
        return

    # ── 获取 / 创建摄像机 ──
    cam_obj, is_new = get_or_create_camera(args.camera)

    # 设置为场景活动摄像机
    bpy.context.scene.camera = cam_obj

    # ── 应用摄像机参数 ──
    apply_camera_settings(cam_obj, args)

    # ── 应用渲染设置 ──
    apply_render_settings(args)

    # ── 确定输出路径 ──
    blend_filepath = bpy.data.filepath
    if not blend_filepath:
        blend_filepath = os.path.join(os.getcwd(), "untitled.blend")
    output_path = determine_output_path(args, blend_filepath)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── 设置输出路径并渲染 ──
    bpy.context.scene.render.filepath = output_path

    print(f"\n[INFO] 开始渲染...")
    print(f"  摄像机:   {cam_obj.name}")
    print(f"  位置:     ({cam_obj.location.x:.3f}, {cam_obj.location.y:.3f}, {cam_obj.location.z:.3f})")
    rot = cam_obj.rotation_euler
    print(f"  旋转:     ({math.degrees(rot.x):.1f}°, {math.degrees(rot.y):.1f}°, {math.degrees(rot.z):.1f}°)")
    print(f"  焦距:     {cam_obj.data.lens:.1f} mm")
    print(f"  引擎:     {bpy.context.scene.render.engine}")
    print(f"  分辨率:   {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")
    print(f"  输出路径: {output_path}")
    print()

    # 执行渲染
    bpy.ops.render.render(write_still=True)

    print(f"\n[SUCCESS] 渲染完成！输出文件: {output_path}")


if __name__ == "__main__":
    main()
