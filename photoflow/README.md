# PhotoFlow

**PhotoFlow** is an agentic framework for language-conditioned virtual photography in controllable 3D scenes. Given a Blender scene and a natural-language photography intent, PhotoFlow searches for an executable camera state, including camera pose, look-at target, lens, aperture, and aspect ratio, then renders the final photograph.

This repository will host the code and benchmark assets for our project:

> PhotoFlow: Agentic 3D Virtual Photography Missions

## Overview

PhotoFlow treats virtual photography as a closed-loop spatial-aesthetic search problem. The agent is organized around three roles:

- **Director** proposes diverse candidate camera states from scene scouts, soft photographic blueprints, global anchors, and region memory.
- **Reviewer** evaluates rendered previews using structured rule checks, visual critique, and pairwise incumbent selection.
- **Reflector** converts failures into search bias, dead-zone suppression, and high-exploration relocation.

We also introduce **VPhotoBench**, a benchmark of language-conditioned virtual photography missions over open-license Blender scenes. The benchmark is designed to evaluate whether an agent can satisfy spatial constraints, semantic intent, aspect-ratio choices, and photographic quality in fully virtual art scenes.

## Current Status

This repository is currently a placeholder for the public release. We are organizing the codebase, benchmark registry, task specifications, and evaluation scripts for a clean release.

## Release Plan

- [ ] Release the PhotoFlow agent code.
- [ ] Release the VPhotoBench scene registry and task specifications.
- [ ] Release benchmark construction notes and metadata.
- [ ] Release evaluation scripts for external metrics and aggregation.
- [ ] Release experiment configuration files for the reported baselines and ablations.
- [ ] Release selected logs and processed result tables.
- [ ] Add setup instructions for Blender, Python dependencies, and model/API configuration.
- [ ] Add example commands for running a single task and reproducing benchmark summaries.

## Benchmark Assets

VPhotoBench is built from publicly available Blender scenes. Some original assets may need to be downloaded from their upstream sources depending on license and file-size constraints. We will provide the scene registry, task metadata, preprocessing instructions, and scripts needed to reconstruct the benchmark layout.

## Citation

The paper citation will be added after the public version is available.

## License

The project license will be finalized before the full code release. Third-party Blender assets remain governed by their original licenses.
