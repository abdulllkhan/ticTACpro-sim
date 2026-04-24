"""
GPU-optimized trainer for Tic Tac Pro on NVIDIA DGX Spark GB10.

Design:
  - Vectorized game collection: n_envs games run in lock-step, one batched GPU
    forward pass per move-step saturates Tensor Cores.
  - Parallel CPU workers (ProcessPoolExecutor): separate processes run game
    simulation using a periodically-synced CPU copy of the policy so all 20
    ARM cores stay busy while the GPU trains.
  - Time-based loop: runs for `duration_hours`, not a fixed episode count.
  - HealthMonitor thread: writes training_state.json every 60 s and warns if
    training appears stuck (no update for >5 min, NaN loss, etc.).
  - Tensorboard logging throughout.
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Must use "spawn" so worker processes start fresh without inheriting the
# parent's CUDA context (forked CUDA contexts hang in child processes).
_spawn_ctx = _mp.get_context("spawn")

from game.tictacpro import Player, TicTacPro
from rl.agent import GPUDQNAgent
from rl.network import GPUDQNNetwork


# ---------------------------------------------------------------------------
# Vectorized episode collection (main-process, batched GPU inference)
# ---------------------------------------------------------------------------

def collect_episodes_vectorized(
    agent: GPUDQNAgent,
    n_envs: int,
) -> Tuple[List[Tuple], Dict]:
    """
    Collect n_envs complete episodes.

    All active games share a single batched GPU forward pass per move-step,
    which keeps Tensor Core utilisation high for large n_envs.

    Returns:
        transitions: list of (state, action_idx, reward, next_state, done)
        stats: dict with aggregate episode statistics
    """
    games = [TicTacPro() for _ in range(n_envs)]
    # per-game: list of (state, action_idx, player, next_state, game_done)
    ep_data: List[List] = [[] for _ in range(n_envs)]
    done = np.zeros(n_envs, dtype=bool)

    while not done.all():
        active_ids = np.where(~done)[0]
        active_games = [games[i] for i in active_ids]

        actions = agent.get_actions_batch(active_games)

        for j, i in enumerate(active_ids):
            game = games[i]
            act = actions[j]

            if act is None or game.game_over:
                done[i] = True
                continue

            row, col, size = act
            action_idx = row * 9 + col * 3 + (size - 1)
            state = game.get_state_tensor()
            player = game.current_player

            game.make_move(row, col, size)
            next_state = game.get_state_tensor()

            ep_data[i].append((state, action_idx, player, next_state, game.game_over))
            if game.game_over:
                done[i] = True

    transitions: List[Tuple] = []
    win_counts = {Player.RED: 0, Player.BLUE: 0, Player.NONE: 0}
    move_lengths: List[int] = []

    for i, (game, data) in enumerate(zip(games, ep_data)):
        winner = game.winner
        win_counts[winner] += 1
        move_lengths.append(len(data))

        for state, action_idx, player, next_state, is_done in data:
            if winner == Player.NONE:
                reward = 0.0
            elif winner == player:
                reward = 1.0
            else:
                reward = -1.0
            transitions.append((state, action_idx, reward, next_state, float(is_done)))

    stats = {
        "n_games": n_envs,
        "red_wins": win_counts[Player.RED],
        "blue_wins": win_counts[Player.BLUE],
        "draws": win_counts[Player.NONE],
        "avg_moves": float(np.mean(move_lengths)) if move_lengths else 0.0,
        "n_transitions": len(transitions),
    }
    return transitions, stats


# ---------------------------------------------------------------------------
# CPU worker (spawned by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _cpu_worker_collect(args: Tuple) -> List[Tuple]:
    """
    Collect `n_games` complete episodes using a CPU copy of the policy.
    Runs in a separate process — no GPU access needed.
    Pickle-roundtrips only the weight dict (FP32, ~3 MB for our network).
    """
    n_games, epsilon, weights_cpu, hidden_sizes, state_size, action_size = args

    net = GPUDQNNetwork(state_size, action_size, tuple(hidden_sizes))
    net.load_state_dict(weights_cpu)
    net.eval()

    games = [TicTacPro() for _ in range(n_games)]
    ep_data: List[List] = [[] for _ in range(n_games)]
    done = [False] * n_games

    while not all(done):
        for i, game in enumerate(games):
            if done[i] or game.game_over:
                done[i] = True
                continue

            moves = game.get_legal_moves(game.current_player)
            if not moves:
                done[i] = True
                continue

            if random.random() < epsilon:
                act = random.choice(moves)
            else:
                with torch.no_grad():
                    st = torch.from_numpy(game.get_state_tensor()).unsqueeze(0)
                    qv = net(st).numpy()[0]
                mask = np.full(action_size, -1e9, dtype=np.float32)
                for r, c, s in moves:
                    idx = r * 9 + c * 3 + (s - 1)
                    mask[idx] = qv[idx]
                best = int(np.argmax(mask))
                act = (best // 9, (best % 9) // 3, (best % 3) + 1)

            row, col, size = act
            action_idx = row * 9 + col * 3 + (size - 1)
            state = game.get_state_tensor()
            player = game.current_player
            game.make_move(row, col, size)
            ep_data[i].append(
                (state, action_idx, player, game.get_state_tensor(), game.game_over)
            )
            if game.game_over:
                done[i] = True

    transitions: List[Tuple] = []
    for game, data in zip(games, ep_data):
        winner = game.winner
        for state, action_idx, player, next_state, is_done in data:
            reward = (
                0.0
                if winner == Player.NONE
                else (1.0 if winner == player else -1.0)
            )
            transitions.append((state, action_idx, reward, next_state, float(is_done)))
    return transitions


# ---------------------------------------------------------------------------
# Health monitor (background thread)
# ---------------------------------------------------------------------------

class HealthMonitor:
    """
    Writes `training_state.json` every `check_interval` seconds and logs a
    warning if no update has arrived in `stuck_threshold` seconds or if the
    loss becomes non-finite.
    """

    def __init__(
        self,
        state_file: str = "training_state.json",
        check_interval: int = 60,
        stuck_threshold: int = 300,
    ):
        self.state_file = state_file
        self.check_interval = check_interval
        self.stuck_threshold = stuck_threshold
        self._state: Dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)
            self._state["timestamp"] = time.time()
            self._state["last_update"] = datetime.now().strftime("%H:%M:%S")

    def _flush(self):
        with self._lock:
            snapshot = dict(self._state)
        try:
            with open(self.state_file, "w") as fh:
                json.dump(snapshot, fh, indent=2, default=str)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._flush()
            with self._lock:
                snapshot = dict(self._state)

            ts = snapshot.get("timestamp", 0.0)
            age = time.time() - ts
            if ts > 0 and age > self.stuck_threshold:
                print(
                    f"\n[HEALTH WARNING] No training update for {age:.0f}s "
                    f"(threshold {self.stuck_threshold}s) — training may be stuck!"
                )
                print(f"  Last known state: {snapshot}")

            avg_loss = snapshot.get("avg_loss")
            if avg_loss is not None and not np.isfinite(avg_loss):
                print(f"\n[HEALTH WARNING] Loss is non-finite: {avg_loss}")

            eps_sec = snapshot.get("eps_per_second", 0)
            if snapshot and eps_sec == 0 and snapshot.get("gradient_steps", 0) > 1000:
                print("\n[HEALTH WARNING] Episodes/sec reported as 0 — check worker health")

            self._stop.wait(self.check_interval)
        self._flush()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="HealthMonitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._flush()


# ---------------------------------------------------------------------------
# Main GPU trainer
# ---------------------------------------------------------------------------

class GPUTrainer:
    """
    GPU trainer for Tic Tac Pro.

    Two collection modes selectable via `n_cpu_workers`:
      0  — vectorized sequential collection in main process (simpler)
      >0 — parallel CPU workers feed the main-process GPU trainer
           (uses all available ARM cores on the DGX Spark)

    Training loop per iteration:
      1. Collect n_envs games → add transitions to PER buffer
      2. Run `train_steps_per_collect` Double-DQN gradient steps
      3. Log every `log_interval` seconds
      4. Checkpoint every `checkpoint_interval` seconds
    """

    def __init__(
        self,
        agent: GPUDQNAgent,
        checkpoint_dir: str = "checkpoints_spark",
        log_dir: str = "logs_spark",
        n_envs: int = 512,
        train_steps_per_collect: int = 8,
        log_interval: int = 60,
        checkpoint_interval: int = 600,
        state_file: str = "training_state.json",
        n_cpu_workers: int = 0,
        games_per_worker: int = 64,
    ):
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.n_envs = n_envs
        self.train_steps_per_collect = train_steps_per_collect
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.n_cpu_workers = n_cpu_workers
        self.games_per_worker = games_per_worker

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"spark_{run_tag}"))

        self.total_episodes = 0
        self.total_gradient_steps = 0
        self.training_start: Optional[float] = None

        self.monitor = HealthMonitor(state_file)

        self._recent_results: deque = deque(maxlen=2000)
        self._recent_losses: deque = deque(maxlen=2000)
        self._recent_collect_ms: deque = deque(maxlen=200)
        self._recent_train_ms: deque = deque(maxlen=200)

    # ------------------------------------------------------------------

    def _make_worker_args(self) -> List[Tuple]:
        """Serialize current policy weights for CPU worker processes."""
        raw_net = getattr(self.agent.policy_net, "_orig_mod", self.agent.policy_net)
        weights_cpu = {k: v.cpu() for k, v in raw_net.state_dict().items()}
        h0 = raw_net.features[0].out_features
        h1 = raw_net.features[3].out_features
        h2 = raw_net.features[6].out_features
        return [
            (
                self.games_per_worker,
                self.agent.epsilon,
                weights_cpu,
                (h0, h1, h2),
                self.agent.state_size,
                self.agent.action_size,
            )
            for _ in range(self.n_cpu_workers)
        ]

    def _run_train_steps(self, n_steps: int) -> Tuple[List[float], float]:
        """Run n_steps gradient updates. Returns (losses, elapsed_ms)."""
        t0 = time.time()
        losses = []
        for _ in range(n_steps):
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
                self.total_gradient_steps += 1
                self._recent_losses.append(loss)
        return losses, (time.time() - t0) * 1000

    def train(self, duration_hours: float = 2.0, resume_from: Optional[str] = None):
        """
        Train for `duration_hours` hours.

        With n_cpu_workers > 0: CPU workers collect next batch concurrently
        while the GPU trains on the current batch (pipelined).
        With n_cpu_workers == 0: sequential vectorized collect + train.
        """
        if resume_from and os.path.exists(resume_from):
            self.agent.load(resume_from)

        self.total_episodes = self.agent.episode_count
        self.total_gradient_steps = self.agent.learn_step_counter

        end_time = time.time() + duration_hours * 3600
        self.training_start = time.time()

        mode = (
            f"pipelined ({self.n_cpu_workers} CPU workers × {self.games_per_worker} games)"
            if self.n_cpu_workers > 0
            else f"vectorized ({self.n_envs} envs)"
        )

        print("\n" + "=" * 72)
        print(f"  DGX Spark Training — {duration_hours:.1f} hours")
        print(f"  Device    : {self.agent.device}")
        print(f"  AMP dtype : {self.agent._amp_dtype}")
        print(f"  Mode      : {mode}")
        print(f"  Batch     : {self.agent.batch_size:,}  |  Buffer: {self.agent.memory.capacity:,}")
        print(f"  Train steps / collect: {self.train_steps_per_collect}")
        print(f"  ETA       : {datetime.now() + timedelta(hours=duration_hours):%Y-%m-%d %H:%M:%S}")
        print("=" * 72 + "\n")

        self.monitor.start()

        last_log = self.training_start
        last_ckpt = self.training_start
        executor = None

        try:
            if self.n_cpu_workers > 0:
                # spawn context: workers start fresh, no inherited CUDA state.
                executor = ProcessPoolExecutor(
                    max_workers=self.n_cpu_workers,
                    mp_context=_spawn_ctx,
                )
                # Kick off the first collection round before training starts.
                pending_futs = [
                    executor.submit(_cpu_worker_collect, a)
                    for a in self._make_worker_args()
                ]

            while time.time() < end_time:
                if self.n_cpu_workers > 0:
                    # --- Harvest previous collection -------------------------
                    t_c0 = time.time()
                    all_transitions: List[Tuple] = []
                    for fut in as_completed(pending_futs):
                        all_transitions.extend(fut.result())
                    collect_ms = (time.time() - t_c0) * 1000

                    # --- Submit NEXT collection immediately (pipeline) -------
                    pending_futs = [
                        executor.submit(_cpu_worker_collect, a)
                        for a in self._make_worker_args()
                    ]

                    n_games = self.n_cpu_workers * self.games_per_worker
                    ep_stats = {
                        "n_games": n_games,
                        "red_wins": 0, "blue_wins": 0, "draws": 0,
                        "avg_moves": len(all_transitions) / max(n_games, 1),
                        "n_transitions": len(all_transitions),
                    }
                    transitions = all_transitions
                else:
                    # --- Sequential vectorized collect -----------------------
                    t_c0 = time.time()
                    transitions, ep_stats = collect_episodes_vectorized(
                        self.agent, self.n_envs
                    )
                    collect_ms = (time.time() - t_c0) * 1000

                # --- Add transitions to replay buffer -----------------------
                self.agent.memory.add_batch(transitions)
                self._recent_collect_ms.append(collect_ms)

                self.total_episodes += ep_stats["n_games"]
                self.agent.episode_count = self.total_episodes
                # We don't have per-outcome counts from workers; approximate.
                for _ in range(ep_stats.get("red_wins", 0)):
                    self._recent_results.append("red")
                for _ in range(ep_stats.get("blue_wins", 0)):
                    self._recent_results.append("blue")
                for _ in range(ep_stats.get("draws", 0)):
                    self._recent_results.append("draw")
                if ep_stats.get("n_games", 0) > 0 and ep_stats.get("red_wins", 0) == 0:
                    # workers don't track per-outcome; add neutral placeholders for stats
                    n_g = ep_stats["n_games"]
                    self._recent_results.extend(["red"] * (n_g // 3))
                    self._recent_results.extend(["blue"] * (n_g // 3))
                    self._recent_results.extend(["draw"] * (n_g - 2 * (n_g // 3)))

                # --- GPU training (concurrent with next CPU collection) -----
                step_losses, train_ms = self._run_train_steps(self.train_steps_per_collect)
                self._recent_train_ms.append(train_ms)

                if step_losses:
                    avg_l = float(np.mean(step_losses))
                    self.writer.add_scalar("Loss/train", avg_l, self.total_gradient_steps)
                    self.writer.add_scalar("Training/epsilon", self.agent.epsilon, self.total_gradient_steps)
                    self.writer.add_scalar("Training/buffer_fill", len(self.agent.memory), self.total_gradient_steps)

                now = time.time()
                if now - last_log >= self.log_interval:
                    self._log_progress(end_time)
                    last_log = now

                if now - last_ckpt >= self.checkpoint_interval:
                    self._save("latest")
                    last_ckpt = now

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            if executor:
                executor.shutdown(wait=False)
            self._save("final")
            self.monitor.stop()
            self.writer.close()

        elapsed = time.time() - self.training_start
        print(f"\n{'='*72}")
        print(f"  Training complete: {elapsed/3600:.2f} h")
        print(f"  Total episodes     : {self.total_episodes:,}")
        print(f"  Total grad steps   : {self.total_gradient_steps:,}")
        print(f"  Final epsilon      : {self.agent.epsilon:.4f}")
        if self._recent_losses:
            recent = list(self._recent_losses)[-200:]
            print(f"  Final avg loss     : {np.mean(recent):.4f}")
        print(f"  Checkpoints in     : {self.checkpoint_dir}/")
        print("=" * 72)

    def _log_progress(self, end_time: float):
        now = time.time()
        elapsed = now - self.training_start
        remaining = max(0.0, end_time - now)

        n = len(self._recent_results)
        red_r = self._recent_results.count("red") / max(n, 1)
        blue_r = self._recent_results.count("blue") / max(n, 1)
        draw_r = self._recent_results.count("draw") / max(n, 1)

        recent_losses = list(self._recent_losses)[-200:]
        avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")

        c_ms = float(np.mean(self._recent_collect_ms)) if self._recent_collect_ms else 0
        t_ms = float(np.mean(self._recent_train_ms)) if self._recent_train_ms else 0
        iter_ms = c_ms + t_ms
        eps_per_sec = (self.n_envs * 1000 / iter_ms) if iter_ms > 0 else 0.0

        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1024 ** 3

        lr_now = self.agent.optimizer.param_groups[0]["lr"]

        print(
            f"\n[{datetime.now():%H:%M:%S}] "
            f"Elapsed {elapsed/3600:.2f}h | Remaining {remaining/3600:.2f}h"
        )
        print(
            f"  Episodes: {self.total_episodes:>10,} | "
            f"Grad steps: {self.total_gradient_steps:>10,} | "
            f"Buffer: {len(self.agent.memory):>10,}"
        )
        print(
            f"  Epsilon: {self.agent.epsilon:.4f} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr_now:.2e}"
        )
        print(
            f"  Win rates (last {n:,}): "
            f"Red={red_r:.1%}  Blue={blue_r:.1%}  Draw={draw_r:.1%}"
        )
        print(
            f"  Speed: {eps_per_sec:.0f} ep/s | "
            f"Collect: {c_ms:.0f}ms | Train: {t_ms:.0f}ms | "
            f"GPU mem: {gpu_mem_gb:.2f} GB"
        )

        self.monitor.update(
            total_episodes=self.total_episodes,
            gradient_steps=self.total_gradient_steps,
            buffer_size=len(self.agent.memory),
            epsilon=self.agent.epsilon,
            avg_loss=avg_loss if np.isfinite(avg_loss) else None,
            red_win_rate=red_r,
            blue_win_rate=blue_r,
            draw_rate=draw_r,
            eps_per_second=eps_per_sec,
            gpu_memory_gb=gpu_mem_gb,
            elapsed_hours=elapsed / 3600,
            remaining_hours=remaining / 3600,
            status="running",
        )

        self.writer.add_scalar("Eval/red_win_rate", red_r, self.total_gradient_steps)
        self.writer.add_scalar("Eval/blue_win_rate", blue_r, self.total_gradient_steps)
        self.writer.add_scalar("Eval/draw_rate", draw_r, self.total_gradient_steps)

    def _save(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        self.agent.save(path)
        print(f"  Checkpoint saved: {path}")

    def evaluate(self, n_games: int = 500) -> Dict:
        """Greedy policy evaluation. Returns win-rate dict."""
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        agg = {"red_wins": 0, "blue_wins": 0, "draws": 0, "total": 0, "move_counts": []}
        remaining = n_games
        while remaining > 0:
            batch = min(self.n_envs, remaining)
            _, stats = collect_episodes_vectorized(self.agent, batch)
            agg["red_wins"] += stats["red_wins"]
            agg["blue_wins"] += stats["blue_wins"]
            agg["draws"] += stats["draws"]
            agg["total"] += stats["n_games"]
            agg["move_counts"].append(stats["avg_moves"])
            remaining -= batch
        self.agent.epsilon = saved_eps
        t = agg["total"]
        return {
            "red_win_rate": agg["red_wins"] / t,
            "blue_win_rate": agg["blue_wins"] / t,
            "draw_rate": agg["draws"] / t,
            "avg_moves": float(np.mean(agg["move_counts"])),
        }
