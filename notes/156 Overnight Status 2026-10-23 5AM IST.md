# 156 - Overnight status, 5 AM IST 2026-10-23

- Laptop: queue_L3 finished 00:59 IST; results in note 155 (best matrix-free: threekd8 1.8249 bpb, +12.7% vs dense 1.6195, FAIL vs bar 1.7815). Laptop GPU idle since; queue_L4.ps1 (K64 two/three-term + KD, seed-1 repeat of threekd8) is ready but must be started by the user.
- Cloud memory test (L-pure kv recall): only the earlier N=4 seed 0 runs exist (full 0.973, nomem 0.426, dense 0.992 -> memory block WIN vs no memory, 1 seed). The overnight background job died after ~500 steps when the session went idle (cloud background jobs do not survive idle sessions). No new memory results. Relaunched at 5 AM IST; progresses only while the session is active.
- Lesson: only the laptop queue scripts run unattended; cloud jobs need an active session.
