# MJXRenderer

MJXRenderer is a renderer for use with the MuJoCo physics engine and JAX. It submits render requests to worker threads which do traditional rendering using OpenGL.
## Installation

Ensure you have the following prerequisites installed:

- Python 3.6+
- [MuJoCo](https://mujoco.org/) 2.1 or later
- JAX
- NumPy

MuJoCo should be properly installed and configured according to its documentation. JAX can be installed using pip:

```bash
pip install jax jaxlib
```

> **Note:** The installation of JAX and jaxlib might vary depending on your system and CUDA version. Refer to the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed instructions.

## API Reference

### MJXRenderer

#### `__init__(model, height=240, width=320, max_geom=10000, num_workers=None)`
Initializes the renderer with the given MuJoCo model and configuration.
- **model**: Instance of `mujoco.MjModel`.
- **height**: Height of the rendered image in pixels.
- **width**: Width of the rendered image in pixels.
- **max_geom**: Maximum number of geometries to be supported by the renderer. Increase if your model is complex.
- **num_workers**: Number of worker processes for parallel rendering. Defaults to the number of CPU cores.

#### `render(data, camera=-1)`
Renders the current state of the MuJoCo simulation.
- **data**: Instance of `mujoco.MjData`.
- **camera**: Camera ID or name to use for rendering. Defaults to the free camera.

#### `close()`
Cleans up resources associated with the renderer, including terminating worker processes. It's important to call this method when you're done using the renderer to avoid resource leaks.


## Usage

Below are examples demonstrating how to use MJXRenderer to render frames from MuJoCo simulations, including integration with JAX for simulations requiring high-throughput visualization.

### JAX Integration Example

```python
import jax
import mujoco
from mjx_renderer import MJXRenderer
from jax import numpy as jnp

# Load your MuJoCo model
model = mujoco.MjModel.from_xml_path('your_model.xml')
data = mujoco.MjData(model)

# Initialize the MJXRenderer
renderer = MJXRenderer(model, height=240, width=320, num_workers=4)

# JAX function to update simulation and render
@jax.jit
def simulate_and_render(qpos):
    # Update the simulation state
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    # Render the current state
    return renderer.render(data)

# Sample usage with JAX
qpos = jnp.array(data.qpos)
image = simulate_and_render(qpos)

# Clean up resources
renderer.close()
```

## License

MJXRenderer is licensed under the MIT License. This license allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software and to permit persons to whom the software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more details, see the LICENSE file in the project repository.
