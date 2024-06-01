# llm.stuff
llm stuff

# Setup

Install torch locally using pip/conda.


For the cpp binding, run `cmake` with the `CMAKE_PREFIX_PATH` pointing to the existing pytorch installation (see below).


Build the project (in `llm.stuff/`) - 
```
# in llm.stuff

mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

cmake --build . --config Release
```

Had to add the following couple of lines to the `CMakeLists.txt` because of this [annoying issue]( https://github.com/pytorch/pytorch/issues/38122), which became a thing only after trying to add `#include <torch/extension.h>`. Was working fine when using `#include <torch/torch.h>` and `#include <pybind11/pybind11.h>` for some reason.
 

```
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
```

Finally, run the python script that calls into the cpp code with -- 

```bash
PYTHONPATH="/path/to/build" python bench.py
```
