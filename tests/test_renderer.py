import os

os.environ["MUJOCO_GL"] = "EGL"

import time

import jax
import mujoco
import numpy as np
from jax import numpy as jnp
from mujoco import mjx
from mujoco.renderer import Renderer

from mjx_renderer import MJXRenderer

from line_profiler import profile


def test_single():
    xml_text = """
    <mujoco>
        <compiler coordinate="local" angle="degree"/>
        <option timestep="0.01"/>
        <worldbody>
            <body name="floor" pos="0 0 0">
                <geom type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
            </body>
            <body name="object" pos="0 0 1">
                <joint name="object_joint" type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0.1 0.1 0.8 1"/>
            </body>
        </worldbody>
        <actuator>
            <position joint="object_joint" kp="100"/>
        </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_text)
    # jax_model = mjx.put_model(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Create a renderer
    renderer = Renderer(model, height=128, width=128)
    print("huh")
    jax_renderer = MJXRenderer(model, height=128, width=128, num_workers=1)

    renderer.update_scene(data)
    cpu_result = renderer.render()

    @jax.jit
    def render_func(data: mjx.Data):
        return jax_renderer.render(data)

    print("hi")
    jax_data = mjx.put_data(model, data)

    jax_result = render_func(jax_data)
    print("woah")

    assert jnp.allclose(cpu_result, jax_result, atol=0)


@profile
def test_1d_vectorized():
    xml_text = """
    <mujoco>
        <compiler coordinate="local" angle="degree"/>
        <option timestep="0.01"/>
        <worldbody>
            <body name="floor" pos="0 0 0">
                <geom type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
            </body>
            <body name="object" pos="0 0 1">
                <joint name="object_joint" type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0.1 0.1 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml_text)
    jax_model = mjx.put_model(model)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Create a renderer
    renderer = Renderer(model, height=128, width=128)
    jax_renderer = MJXRenderer(model, height=128, width=128, num_workers=16)

    renderer.update_scene(data)
    cpu_result = renderer.render()

    def cpu_render_func(q):
        data.qpos[0] = q
        data.qpos[:] = data.qpos.astype(np.float32)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        return renderer.render()

    def jax_render_func(q):
        data = mjx.make_data(jax_model)
        qpos = data.qpos.at[0].set(q)
        data = data.replace(qpos=qpos)
        data = mjx.step(jax_model, data)

        return jax_renderer.render(data)

    qs = jnp.linspace(-1, 1, 4096)

    cpu_result = np.stack([cpu_render_func(q) for q in qs])
    jax_result = np.asarray(jax.vmap(jax_render_func)(qs))

    cpu_result = cpu_result.astype(np.int32)
    jax_result = jax_result.astype(np.int32)

    assert np.allclose(cpu_result, jax_result, atol=2)

    jax_renderer.close()


def test_destructor():

    xml_text = """
    <mujoco>
        <compiler coordinate="local" angle="degree"/>
        <option timestep="0.01"/>
        <worldbody>
            <body name="floor" pos="0 0 0">
                <geom type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
            </body>
            <body name="object" pos="0 0 1">
                <joint name="object_joint" type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0.1 0.1 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml_text)

    jax_renderer = MJXRenderer(model, height=128, width=128, num_workers=3)

    jax_renderer.close()


if __name__ == "__main__":
    # test_single()
    test_1d_vectorized()
    # test_destructor()
