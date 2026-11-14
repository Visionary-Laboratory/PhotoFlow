# PhotoFlow agent

## Runtime files

- `main.py`: Director–Reviewer–Reflector search loop, global anchors, region memory, high-explore relocation, early stopping, and parallel preview rendering.
- `main_pipeline.py`: compatibility wrapper for the same entry point.
- `agent_vlm.py`: role-conditioned prompts and OpenAI-compatible client calls.
- `env_blender.py`: Blender scene inspection, camera control, projection/rule indicators, preview rendering, and final rendering.
- `blender_render.py`: external Blender rendering helpers.

## Run

Configure `BLENDER_PATH`, `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL` in the environment, then run:

```bash
"$BLENDER_PATH" -b /path/to/your_scene.blend -P photoflow/main.py -- \
  --instruction "A clear, cinematic photograph of the main subject" \
  --reference "ExactBlenderObjectName" \
  --random-seed 0 \
  --max-iter 6 \
  --num-candidates 4 \
  --scout-views 4 \
  --preview-samples 64 \
  --preview-workers 2 \
  --final-samples 128 \
  --render-engine CYCLES \
  --output output/final_image.png
```

Use `--list-objects` to inspect scene object names without calling the LLM. Run `--help` after the `--` separator for all camera, rendering, and early-stopping options.

Preview samples are capped at 64. Parallel previews require a saved `.blend` file and a valid Blender executable; otherwise the pipeline uses serial rendering. Final rendering is not capped to preview samples.
