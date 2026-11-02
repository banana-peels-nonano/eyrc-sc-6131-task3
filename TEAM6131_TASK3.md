# Team 6131 — Strata Cobot, Task 3

This is the Team 6131 working copy of the official e-Yantra Strata Cobot repository (release `v3.0.0`).
Our code is in `algorithms/scripts/task3a/` (and later `task3b/`). Everything else is e-Yantra's, unmodified.

**Keep this repository private.** It holds our competition solutions, plus e-Yantra's simulation
packages and evaluator binary.

## Status

| Subtask | State | Best result |
|---|---|---|
| 3A | Recorded and packaged, 31 Oct 2026 | **27.81 / 30**. Run score 46.35 / 50: detected 5/5, dropped 5/5, eBot legs 2/2, bonus, T = 182.4 s |
| 3B | Recorded and packaged, 2 Nov 2026 | **47.99 / 50**. Both runs: detected 5/5, dropped 5/5, eBot legs 2/2, bonus; T = 97–100 s |
| 3C (viva) | Opens 10 Nov; last Start 16 Nov 20:44 | — |

Every evaluated 3A run (all with full detection, drops, legs and bonus; only the time changed):

| Version (git tag) | T (s) | Marks / 30 | What changed |
|---|---|---|---|
| `3a-baseline-25.32` | 390.0 | 25.32 | First complete run: Task 2 logic plus the package handover |
| `3a-via-26.22` | 315.0 | 26.22 | Loose (3 cm) tolerance on waypoints the arm only passes through |
| `3a-speed-27.32` | 223.7 | 27.32 | Task 3 speed caps, position gain 1.5, package job can interrupt an ore, eBot speed bug fixed |
| `3a-sweep-27.49` | 209.2 | 27.49 | Ore transfers by turning the base joint ("pan sweep") instead of a Cartesian route |
| `3a-final-27.74` | 182.4 (recorded) | **27.81** | Lift straight onto the sweep circle, coarser sweep end, one retry if a drop doesn't release |

The time term is (600 − T) × 0.02, so 30/30 would need T = 0. Every other component is already at
its maximum.

## How to run 3A

```bash
cd ~/ros2_ws && colcon build && source install/setup.bash
# for an evaluation, first, from ~/ros2_ws/src:  ./eyrc-sc-evaluator --task 3A --team-id 6131
ros2 launch eyantra_kepler_colony task3a.launch.py      # wait for "world ready"
ros2 run algorithms task3A_perception.py
ros2 run algorithms task3A_manipulation.py
ros2 run algorithms task3A_planning.py
ros2 run algorithms task3A_navigation.py
```

## How 3A works

**eBot**
- `task3A_planning.py`:
  - plans each leg with A* plus any-angle (Lazy Theta*) search on `/map`, after inflating rocks by the eBot's radius
  - tries 0.10 m, then 0.05 m, then 0 m of extra clearance
  - publishes the path on `/ebot_path`
  - calls `/spawn_ore_package` at (6.0, 1.9) and retries until the package is granted
- `task3A_navigation.py`:
  - follows each path with a look-ahead (carrot) controller; look-ahead 0.15 m, up to 0.25 m/s
  - slows down only near the end of the path
- Stop pose beside the arm: (0.0, 1.9, π).

**Handover.** Our own topics, both latched (TRANSIENT_LOCAL):
- `/ebot_at_arm` (Bool): the planner publishes `true` once the eBot is parked beside the arm with the package.
- `/package_cleared` (Bool): manipulation publishes `true` once the package is 8 cm above the deck. The eBot then drives home.

**Arm**
- `task3A_perception.py` publishes:
  - ore transforms `<kind>_<n>` from HSV colour plus depth
  - the `ore_package` transform, found from the purple panel on its top face, only while `/ebot_at_arm` is true and the package hasn't been lifted
- `task3A_manipulation.py` is a state machine:
  - Ores: pick and drop as in Task 2, but transfers turn the base joint at a 0.55 m radius.
  - Package: starts as soon as the eBot is parked, provided the arm isn't holding an ore. Sequence:
    1. sweep to the package
    2. pick it, checking for the ≈2.94 N load
    3. sweep to the hopper (straight behind the arm)
    4. tilt 20° outward and release at TCP (−0.88, 0, 0.34) in `base_link`
    5. leave in joint space

**Rules we learned the hard way** (each one was a protective stop or a fault):
- Keep transit waypoints within 0.80 m of the shoulder. A straight lift from a far ore latched CRITICAL_SINGULARITY.
- Make big direction changes with the base joint, choosing which way round. Cartesian routes wound the joint into its ±2π limit (CRITICAL_JOINT_LIMIT).
- Never move straight back from full stretch. Leave the hopper in joint mode.
- The far-left well's drop point is near its edge. If the payload force doesn't clear after release, lift the tool 5 cm and check again.

## Measured facts used in the code

- **Package on the parked eBot:** top-face centre ≈ (0.651, 0.091, 0.255) in `base_link`. Perception error was about 2 mm.
- **Hopper opening, in `base_link`** (from the crusher's collision mesh):
  - x −1.123 … −0.763
  - y −0.195 … 0.187
  - rim z +0.207
- **Reach test** (TCP at z 0.33): the singularity warning starts at x = −0.869 / −0.917 / −0.944 / −0.960 for tool tilts of 0 / 20 / 35 / 50°.
- **In 3A, `/odom` equals ground truth.** 3B adds sensor noise, and its `/map` may not match the world.

## Task 3B submission (2 Nov 2026)

- Recorded result: **47.99 / 50** (unrecorded best with the same code: 48.03)
- Submitted code: git tag `3b-final-47.99` (same code as `3b-perc-guard`)
- Package: `SC#6131_task3B.zip` = signed `result.zip` + the four `task3B_*.py`
- Video (unlisted): https://www.youtube.com/watch?v=CefhoqSfeg8

| Version (git tag) | What changed | Evaluated |
|---|---|---|
| 3B attempt 1 | 3A logic + 3B noise fixes | 46.29 recorded |
| `3b-speed-2` | Own UR5e inverse kinematics: each ore transfer is one joint move; eBot 0.35 m/s | - |
| `3b-speed-3b` | Package fetched and delivered by joint moves; higher gains | - |
| `3b-speed-5` | Ore moves never swing across the parked eBot | 44.87 (one detection missed in run 1) |
| `3b-perc-guard` | Package TF holds its full view when partly hidden; ore TF height clamped | 48.03 unrecorded, **47.99 recorded** |
