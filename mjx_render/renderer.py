# Before anything else is imported, set the environment variable to use the right rendering backend
import os

os.environ["MUJOCO_GL"] = "EGL"
import multiprocessing
import pickle
from multiprocessing import Event, Process, Queue
from queue import Empty as QueueEmptyException
from typing import Optional, Union

import jax
import mujoco
import numpy as np
from jax import config, core
from jax import numpy as jnp

from jax.tree_util import Partial, register_pytree_node_class
from mujoco import mjx


@register_pytree_node_class
class MJXRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
        num_workers: Optional[int] = None,
    ):
        if num_workers is None:
            # Set the number of workers to the number of threads
            num_workers = os.cpu_count()
        if num_workers is None:
            # Failed to get the number of threads, set to 1
            num_workers = 1

        self._num_workers = num_workers
        self._workers = []

        self._mp_context = multiprocessing.get_context("spawn")
        self._sync_manager = self._mp_context.Manager()
        self._share_manager = multiprocessing.managers.SharedMemoryManager()  # type: ignore
        self._share_manager.start()

        self._queue = self._sync_manager.Queue()

        self._model = model

        self._width = width
        self._height = height
        self._max_geom = max_geom

        self._init_workers()

    def _init_workers(self):
        for _ in range(self._num_workers):
            worker = JAXRenderWorker(self)
            self._workers.append(worker)

    @property
    def _bytes_per_frame(self) -> int:
        return self._width * self._height * 3 * np.dtype(np.uint8).itemsize

    def _host_render(
        renderer,
        qpos: jax.Array,
        camera: Union[int, str] = -1,
    ):
        self = renderer

        leading_dims = qpos.shape[:-1]
        qpos = qpos.reshape(-1, qpos.shape[-1])

        # Create a shared memory object to hold the qpos
        qpos_mem = self._share_manager.SharedMemory(
            size=qpos.nbytes,
        )
        qpos_arr = np.frombuffer(qpos_mem.buf, dtype=np.float32).reshape(
            -1, qpos.shape[-1]
        )
        qpos_arr[:] = qpos

        frame_count = int(np.prod(leading_dims))
        n_bytes = self._bytes_per_frame * frame_count
        result_mem = self._share_manager.SharedMemory(size=n_bytes)

        frames_per_worker = max(frame_count // self._num_workers, 1)

        start_idx = 0
        tasks = []
        while start_idx < frame_count:
            task = JAXRenderWorkerTask(
                self,
                qpos_mem,
                result_mem,
                start_idx,
                frames_per_worker,
                qpos.shape[-1],
                camera,
            )

            tasks.append(task)

            pickled_task = pickle.dumps(task)
            self._queue.put(pickled_task)

            start_idx += frames_per_worker

        for task in tasks:
            task.await_result()

        result = (
            np.frombuffer(
                result_mem.buf,
                dtype=np.uint8,
            )
            .reshape(*leading_dims, self._width, self._height, 3)
            .copy()
        )

        del qpos_arr

        qpos_mem.close()
        qpos_mem.unlink()
        result_mem.close()
        result_mem.unlink()

        return result

    @Partial(jax.jit, static_argnames=("camera",))
    def render(
        self,
        data: mjx.Data,
        camera: Union[int, str] = -1,
    ):
        # Define a callback function to bind camera to the function
        # This avoids it being interpreted as a JAX type in case it is a string
        def camera_bound_render_callback(qpos):
            return self._host_render(qpos, camera)

        img: jax.Array = jax.pure_callback(  # type: ignore
            callback=camera_bound_render_callback,
            qpos=data.qpos,
            result_shape_dtypes=jnp.zeros(
                [*data.qpos.shape[:-1], self._width, self._height, 3],
                dtype=jnp.uint8,
            ),
            vectorized=True,
        )

        return img

    def tree_flatten(self):
        return (), {
            "num_workers": self._num_workers,
            "workers": self._workers,
            "mp_context": self._mp_context,
            "sync_manager": self._sync_manager,
            "share_manager": self._share_manager,
            "queue": self._queue,
            "model": self._model,
            "width": self._width,
            "height": self._height,
            "max_geom": self._max_geom,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Rebuid the renderer with the same pieces
        renderer = cls.__new__(cls)

        renderer._num_workers = aux_data["num_workers"]
        renderer._workers = aux_data["workers"]
        renderer._mp_context = aux_data["mp_context"]
        renderer._sync_manager = aux_data["sync_manager"]
        renderer._share_manager = aux_data["share_manager"]
        renderer._queue = aux_data["queue"]
        renderer._model = aux_data["model"]
        renderer._width = aux_data["width"]
        renderer._height = aux_data["height"]
        renderer._max_geom = aux_data["max_geom"]

        return renderer

    def close(self):

        for worker in self._workers:
            worker.close()

        self._sync_manager.shutdown()
        self._share_manager.shutdown()


class JAXRenderWorkerTask:

    def __init__(
        self,
        jax_renderer: MJXRenderer,
        qpos_mem,
        result_mem,
        start_idx,
        frame_count,
        qpos_trailing_dim,
        camera: Union[int, str] = -1,
    ):
        self.fulfill_event = jax_renderer._sync_manager.Event()
        self.qpos_mem = qpos_mem
        self.result_mem = result_mem
        self.start_idx = start_idx
        self.frame_count = frame_count
        self.qpos_trailing_dim = qpos_trailing_dim
        self.result_trailing_dim = (jax_renderer._width, jax_renderer._height, 3)
        self.camera = camera

    @property
    def result_img(self):
        result_arr = np.frombuffer(
            self.result_mem.buf,
            dtype=np.uint8,
        ).reshape(-1, *self.result_trailing_dim)
        return result_arr[self.start_idx : self.start_idx + self.frame_count]

    @property
    def qpos(self):
        qpos_arr = np.frombuffer(self.qpos_mem.buf, dtype=np.float32).reshape(
            -1, self.qpos_trailing_dim
        )
        return qpos_arr[self.start_idx : self.start_idx + self.frame_count]

    def await_result(self):
        self.fulfill_event.wait()

    def fulfill(self, img: Optional[np.ndarray] = None):
        if img is not None:
            self.result_img[:] = img
        self.fulfill_event.set()

    def close(self):
        self.qpos_mem.close()
        self.result_mem.close()


class JAXRenderWorker:

    def __init__(self, jax_renderer: MJXRenderer):
        self._stop_event = jax_renderer._sync_manager.Event()
        # Launch the worker process
        self._process = jax_renderer._mp_context.Process(
            target=self._render_loop,
            args=(
                jax_renderer._queue,
                self._stop_event,
                jax_renderer._model,
                jax_renderer._width,
                jax_renderer._height,
                jax_renderer._max_geom,
            ),
        )
        self._process.start()

    @staticmethod
    def _render_loop(queue, stop_event, model, width, height, max_geom):
        os.environ["MUJOCO_GL"] = "EGL"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # Create a renderer on the host
        renderer = mujoco.Renderer(model, width, height, max_geom)
        data = mujoco.MjData(model)  # type: ignore

        # Define a function to call the renderer
        def render(task: JAXRenderWorkerTask):
            for qpos, result_img in zip(task.qpos, task.result_img):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)  # type: ignore
                renderer.update_scene(data, task.camera)
                renderer.render(out=result_img)

        def loop_body():
            try:
                pickled_task = queue.get(timeout=1)
                task = pickle.loads(pickled_task)
                render(task)
                task.fulfill()
                task.close()
            except QueueEmptyException:
                # No task, check if we should stop then continue
                pass

        # Poll the queue for render tasks and also make sure we are still running
        while not stop_event.is_set():
            loop_body()

        # Close the renderer
        renderer.close()

    def close(self):
        self._stop_event.set()
        self._process.join()
