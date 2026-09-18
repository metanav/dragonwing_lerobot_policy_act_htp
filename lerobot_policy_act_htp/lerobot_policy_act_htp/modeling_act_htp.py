"""
Drop-in replacement for ACTPolicy where the ResNet+Transformer forward
pass runs on a Qualcomm Hexagon NPU via a Qualcomm-AI-Hub-compiled
.tflite artifact, instead of PyTorch.

Because ACTHTPConfig carries the same input_features/output_features/
normalization_mapping as the original ACTConfig, lerobot-rollout's real
preprocessor/postprocessor (built generically from those config fields)
handles normalization BEFORE calling predict_action_chunk/select_action
here, and un-normalization AFTER -- this class only needs to run the
compiled graph on already-normalized tensors and hand back a
still-normalized action chunk, exactly mirroring what the real
ACTPolicy.model(batch) forward pass does internally.

UNVERIFIED ASSUMPTIONS (flagging explicitly rather than silently
guessing):
  - That lerobot-rollout's SyncInferenceEngine calls .select_action()
    with an already fully preprocessed batch, the same contract as the
    real ACTPolicy. This matches every reference (LeRobot's own docs,
    D-Robotics' working script) seen so far, but has not been confirmed
    against your specific installed lerobot-rollout version's internals.
  - That PreTrainedPolicy's __init__/from_pretrained machinery tolerates
    a policy with no real trainable PyTorch parameters. The dummy
    parameter below is a hedge against strict framework assumptions
    (e.g. .to(device) calls expecting at least one parameter to exist)
    -- remove it if it turns out to be unnecessary or causes issues.
"""

from collections import deque
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_act_htp import ACTHTPConfig


class ACTHTPPolicy(PreTrainedPolicy, nn.Module):
    config_class = ACTHTPConfig
    name = "act_htp"

    def __init__(self, config: ACTHTPConfig, dataset_stats=None):
        PreTrainedPolicy.__init__(self, config)
        nn.Module.__init__(self)
        self.config = config

        # No real trainable weights live in this process -- the actual
        # compute happens in the compiled .tflite graph on the NPU. This
        # dummy parameter exists only so standard nn.Module machinery
        # elsewhere in lerobot-rollout (e.g. .to(device), .eval()) has at
        # least one parameter to operate on without special-casing.
        self._unused_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._interpreter = None  # lazily built on first use, see below
        # No maxlen: a bounded deque here actively discarded still-
        # pending actions on every refill (confirmed bug -- see below).
        # Size is governed naturally by prefetch_threshold instead.
        self._action_queue = deque([])

        # Background-prefetch state. A lock guards the interpreter itself
        # (a single TFLite Interpreter instance is not assumed safe for
        # concurrent invoke() calls), and a separate lock guards the
        # queue since the main control-loop thread pops from it while a
        # background thread may be extending it.
        self._interpreter_lock = Lock()
        self._queue_lock = Lock()
        self._prefetch_start_lock = Lock()
        self._prefetch_thread: Thread | None = None
        self._warmed_up = False

    def _ensure_interpreter(self):
        """
        Lazy import + lazy build, so this class stays importable (for
        config inspection, unit tests, CLI --help, etc.) on machines
        without ai_edge_litert / the QNN HTP runtime installed, e.g. a
        Mac used only for training and export.
        """
        if self._interpreter is not None:
            return

        from ai_edge_litert.interpreter import Interpreter, load_delegate

        model_path = Path(self.config.pretrained_path) / self.config.tflite_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Compiled model not found at {model_path}. Check "
                f"tflite_filename in your config matches the actual "
                f"filename in this checkpoint directory."
            )

        delegate_options = {
            "backend_type": "htp",
            "library_path": "libQnnHtp.so",
            "skel_library_dir": self.config.htp_skel_library_dir,
            "htp_precision": "0",
            "htp_performance_mode": self.config.htp_performance_mode,
        }
        if self.config.cache_dir is not None:
            delegate_options["cache_dir"] = self.config.cache_dir
            delegate_options["model_token"] = self.config.model_token

        delegate = load_delegate("libQnnTFLiteDelegate.so", options=delegate_options)
        self._interpreter = Interpreter(model_path=str(model_path), experimental_delegates=[delegate])
        self._interpreter.allocate_tensors()

        import logging

        logging.getLogger(__name__).info(
            f"ACTHTPPolicy: compiled model loaded and running on Hexagon HTP NPU "
            f"via QNN delegate (cache_dir={self.config.cache_dir!r}). "
            f"The 'device=cpu' shown at policy load time refers only to "
            f"PyTorch-side pre/post-processing tensors, NOT where the "
            f"ACT network itself executes."
        )

    def _warm_up(self):
        """
        Runs one dummy inference to absorb the cold-start cost confirmed
        by real testing: even with a warm cache_dir, the first 1-2
        inference calls after process start take ~400-450ms (likely HTP/
        FastRPC session establishment -- distinct from graph compilation,
        which cache_dir already avoids), vs. ~34ms steady-state. Called
        from reset() (see below), which fires once during rollout setup
        -- BEFORE the control loop starts -- so this cost lands during
        setup rather than stalling the first real control tick.
        """
        if self._warmed_up:
            return
        self._ensure_interpreter()

        import logging
        import time

        input_details = self._interpreter.get_input_details()
        dummy_inputs = {d["index"]: np.zeros(d["shape"], dtype=np.float32) for d in input_details}
        t0 = time.perf_counter()
        with self._interpreter_lock:
            for idx, arr in dummy_inputs.items():
                self._interpreter.set_tensor(idx, arr)
            self._interpreter.invoke()
        logging.getLogger(__name__).info(
            f"ACTHTPPolicy: warm-up inference took {(time.perf_counter() - t0) * 1000:.1f} ms"
        )
        self._warmed_up = True

    def get_optim_params(self):
        # Inference-only policy -- nothing here is meant to be trained.
        # If lerobot-train is ever pointed at --policy.type=act_htp by
        # mistake, this empty param list should make that fail loudly
        # and immediately rather than silently doing nothing useful.
        return []

    def reset(self):
        with self._queue_lock:
            self._action_queue.clear()
        # Note: does not forcibly stop an in-flight prefetch thread --
        # it will complete and append to the (now-cleared) queue
        # harmlessly. If reset() is meant to discard in-flight
        # predictions too (e.g. task changed mid-chunk), this needs a
        # generation counter to make stale prefetch results a no-op --
        # not implemented here, worth adding if you see stale actions
        # appearing right after a reset.

        # Confirmed via real logs that reset() fires once during rollout
        # setup, before "Base strategy control loop started" -- this is
        # the right place to pay the cold-start warm-up cost so it
        # doesn't stall the first real control tick. Guarded in a
        # try/except so this stays a harmless no-op if reset() is ever
        # called somewhere ai_edge_litert isn't installed (e.g. if you
        # test config-loading-only behavior on a non-board machine).
        try:
            self._warm_up()
        except ImportError:
            pass

    def forward(self, batch):
        raise NotImplementedError(
            "ACTHTPPolicy is inference-only. Train the original ACT "
            "checkpoint with lerobot-train, then export/compile it via "
            "Qualcomm AI Hub, and point --policy.path at the resulting "
            "directory (containing both the original checkpoint files "
            "and the compiled .tflite) with --policy.type=act_htp."
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        `batch` arrives here ALREADY NORMALIZED -- lerobot-rollout's real
        preprocessor (built from ACTHTPConfig, which shares ACTConfig's
        normalization_mapping) applies normalization before this is
        called, same as for a real ACTPolicy. We only run the compiled
        graph; un-normalization happens externally via the real
        postprocessor afterward.
        """
        self._ensure_interpreter()

        with self._interpreter_lock:
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()

            state_np = batch["observation.state"].cpu().numpy().astype(np.float32)
            top_np = batch["observation.images.top"].cpu().numpy().astype(np.float32)
            wrist_np = batch["observation.images.wrist"].cpu().numpy().astype(np.float32)

            matched = {"state": False, "top": False, "wrist": False}
            for d in input_details:
                name = d["name"].lower()
                if "state" in name:
                    self._interpreter.set_tensor(d["index"], state_np)
                    matched["state"] = True
                elif "top" in name:
                    self._interpreter.set_tensor(d["index"], top_np)
                    matched["top"] = True
                elif "wrist" in name:
                    self._interpreter.set_tensor(d["index"], wrist_np)
                    matched["wrist"] = True

            if not all(matched.values()):
                raise RuntimeError(
                    f"Could not match all three inputs by name -- found "
                    f"input names {[d['name'] for d in input_details]}. "
                    f"Fix the name-matching logic above to your model's "
                    f"actual tensor names."
                )

            self._interpreter.invoke()
            action_chunk = self._interpreter.get_tensor(output_details[0]["index"])
            return torch.from_numpy(action_chunk)

    def _prefetch_worker(self, batch: dict[str, Tensor]):
        """
        Runs on a background thread. Computes the next chunk and appends
        it to the queue under _queue_lock, then clears _prefetch_thread
        so a future call knows it's safe to start another prefetch.
        """
        import logging
        import time

        t0 = time.perf_counter()
        try:
            action_chunk = self.predict_action_chunk(batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logging.getLogger(__name__).info(f"ACTHTPPolicy: prefetch inference took {elapsed_ms:.1f} ms")
            with self._queue_lock:
                # Only queue the first n_action_steps of the model's full
                # chunk_size prediction -- matches ACT's original design
                # (only trust/execute a bounded prefix of each open-loop
                # prediction) and the confirmed working D-Robotics
                # reference, which explicitly slices [:, :n_action_steps].
                # Previously this used the full chunk_size unconditionally,
                # which silently made n_action_steps a no-op whenever it
                # differed from chunk_size.
                n = self.config.n_action_steps
                self._action_queue.extend(action_chunk[0, i] for i in range(n))
        finally:
            with self._prefetch_start_lock:
                self._prefetch_thread = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Background-prefetching version of the standard action-chunking
        queue pattern. Rather than only refilling the queue once it's
        completely empty (which stalls the single-threaded, inline
        control loop for the full inference call every n_action_steps
        ticks), this starts computing the NEXT chunk in a background
        thread once the queue drops to config.prefetch_threshold items
        remaining, while the main thread keeps popping and returning
        already-computed actions without stalling.

        If the queue is empty when a prefetch is already in flight (this
        is EXPECTED and unavoidable on the very first tick after reset(),
        since the queue always starts empty regardless of warm-up), this
        waits on that SAME thread rather than launching a second,
        redundant inference call. An earlier version launched a
        duplicate synchronous call here, which -- confirmed via real
        timing logs -- cost roughly (prefetch_time + steady_state_time)
        due to both calls serializing on _interpreter_lock, not a
        mysterious slow first call. Waiting on the existing thread
        instead means the first-tick stall is a single clean inference
        (~35ms observed), not doubled-up redundant work.

        If the queue is empty and NO prefetch is in flight (shouldn't
        happen given the logic above, but as a genuine safety net),
        falls back to a fresh synchronous call.
        """
        import logging
        import time

        with self._queue_lock:
            queue_len = len(self._action_queue)

        with self._prefetch_start_lock:
            in_flight = self._prefetch_thread
            if queue_len <= self.config.prefetch_threshold and in_flight is None:
                in_flight = Thread(target=self._prefetch_worker, args=(batch,), daemon=True)
                self._prefetch_thread = in_flight
                in_flight.start()

        with self._queue_lock:
            if len(self._action_queue) > 0:
                return self._action_queue.popleft()

        if in_flight is not None:
            # Queue is empty and a prefetch (just-started or already
            # running) will fill it shortly -- wait for THAT result
            # instead of computing a redundant duplicate chunk.
            logging.getLogger(__name__).warning(
                "ACTHTPPolicy: queue empty, waiting on in-flight prefetch "
                "rather than firing a redundant duplicate call. Expected "
                "on the first tick after reset(); if this recurs "
                "mid-session, raise --policy.prefetch_threshold."
            )
            t0 = time.perf_counter()
            in_flight.join()
            logging.getLogger(__name__).info(
                f"ACTHTPPolicy: waited {(time.perf_counter() - t0) * 1000:.1f} ms for in-flight prefetch"
            )
            with self._queue_lock:
                if len(self._action_queue) > 0:
                    return self._action_queue.popleft()

        # Genuine safety net: empty queue, no prefetch in flight at all.
        # Should not happen given the logic above, but avoids a hard
        # crash if it somehow does.
        logging.getLogger(__name__).warning(
            "ACTHTPPolicy: no prefetch in flight and queue empty -- "
            "running a fresh synchronous call. This path should not "
            "normally be reached; if you see it, something upstream of "
            "select_action() may be misbehaving (e.g. reset() not being "
            "called, or an exception in the prefetch worker)."
        )
        action_chunk = self.predict_action_chunk(batch)
        with self._queue_lock:
            n = self.config.n_action_steps
            self._action_queue.extend(action_chunk[0, i] for i in range(n))
            return self._action_queue.popleft()
