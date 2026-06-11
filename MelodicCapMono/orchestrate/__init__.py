"""MelodicCapMono orchestration — one command per take, batchable.

Phase 1: `process_take` chains the already-proven pieces (video2pose →
pose_json_to_bvh → headless constraint-bake in Blender) into a single
unattended command. Nothing here re-implements retarget math; it shells
the WHAM wrapper, calls the pure-numpy BVH exporter in-process, and
drives a headless Blender that adapts the proven constraint-bake from
oldscripts/mocap_to_rigify_complete.py.
"""
