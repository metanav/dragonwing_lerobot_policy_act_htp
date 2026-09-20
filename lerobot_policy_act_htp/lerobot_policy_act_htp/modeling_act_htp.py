"""
Drop-in replacement for ACTPolicy where the ResNet+Transformer forward
pass runs on a Qualcomm Hexagon NPU via a Qualcomm-AI-Hub-compiled 
.tflite artifact, instead of PyTorch.
"""

from collections import deque
from pathlib import Path
from threading import Lock, Thread
import logging
import time
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
      
        # This dummy parameter exists only so standard nn.Module machinery
        # elsewhere in lerobot-rollout has at least one parameter to operate 
        # on without special-casing.
        self._unused_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._interpreter = None  
        self._action_queue = deque([])
        self._interpreter_lock = Lock()
        self._queue_lock = Lock()
        self._prefetch_start_lock = Lock()
        self._prefetch_thread: Thread | None = None
        self._warmed_up = False

    def _ensure_interpreter(self):
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

        logging.getLogger(__name__).info(
            f"ACTHTPPolicy: compiled model loaded and running on Hexagon HTP NPU "
            f"via QNN delegate (cache_dir={self.config.cache_dir!r}). "
            f"The 'device=cpu' shown at policy load time refers only to "
            f"PyTorch-side pre/post-processing tensors, NOT where the "
            f"ACT network itself executes."
        )

    def _warm_up(self):
        if self._warmed_up:
            return
        self._ensure_interpreter()
      
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
        # Inference-only policy
        return []

    def reset(self):
        with self._queue_lock:
            self._action_queue.clear()
        try:
            self._warm_up()
        except ImportError:
            pass

    def forward(self, batch):
        raise NotImplementedError("ACTHTPPolicy is inference-only.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
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
        t0 = time.perf_counter()
        try:
            action_chunk = self.predict_action_chunk(batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logging.getLogger(__name__).info(f"ACTHTPPolicy: prefetch inference took {elapsed_ms:.1f} ms")
            with self._queue_lock:
                n = self.config.n_action_steps
                self._action_queue.extend(action_chunk[0, i] for i in range(n))
        finally:
            with self._prefetch_start_lock:
                self._prefetch_thread = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
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
            logging.getLogger(__name__).warning(
                "ACTHTPPolicy: queue empty"
            )
            t0 = time.perf_counter()
            in_flight.join()
            logging.getLogger(__name__).info(
                f"ACTHTPPolicy: waited {(time.perf_counter() - t0) * 1000:.1f} ms for in-flight prefetch"
            )
            with self._queue_lock:
                if len(self._action_queue) > 0:
                    return self._action_queue.popleft()

        logging.getLogger(__name__).warning(
            "ACTHTPPolicy: no prefetch in flight and queue empty"
        )
        action_chunk = self.predict_action_chunk(batch)
        with self._queue_lock:
            n = self.config.n_action_steps
            self._action_queue.extend(action_chunk[0, i] for i in range(n))
            return self._action_queue.popleft()
