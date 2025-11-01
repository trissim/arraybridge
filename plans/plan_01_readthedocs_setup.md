# plan_01_readthedocs_setup.md
## Component: ReadTheDocs Documentation Infrastructure and CI Setup

### Objective
Set up complete ReadTheDocs documentation infrastructure for arraybridge following the same patterns as metaclass-registry and openhcs, including Sphinx-based documentation, ReadTheDocs configuration, and comprehensive documentation content covering all features and APIs.

### Plan

#### 1. Remove Conflicting MkDocs Configuration
**Action**: Delete `mkdocs.yml` since we're using Sphinx (like metaclass-registry)
- Remove `mkdocs.yml` file
- Remove MkDocs from `pyproject.toml` dev dependencies

#### 2. Create ReadTheDocs Configuration File
**File**: `.readthedocs.yml`
**Content**: Based on metaclass-registry pattern
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: false

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

#### 3. Update pyproject.toml Dependencies
**Section**: `[project.optional-dependencies]`
**Changes**:
- Remove mkdocs-related dependencies
- Add Sphinx dependencies matching metaclass-registry:
```toml
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "sphinx-autodoc-typehints>=1.24",
]
```

#### 4. Restructure Documentation Directory
**Current**: `docs/sphinx/` with mixed structure
**Target**: `docs/` with `source/` subdirectory (matching metaclass-registry)

**Actions**:
- Move `docs/sphinx/source/*` → `docs/source/`
- Move `docs/sphinx/Makefile` → `docs/Makefile`
- Move `docs/sphinx/make.bat` → `docs/make.bat`
- Delete `docs/sphinx/` directory
- Delete `docs/ci-cd.md` (will be replaced with proper RST docs)

#### 5. Update Sphinx Configuration
**File**: `docs/source/conf.py`
**Updates needed**:
- Fix path to source code: `sys.path.insert(0, os.path.abspath('../../src'))`
- Update project metadata to match pyproject.toml
- Add sphinx-autodoc-typehints extension
- Configure autodoc to properly handle optional dependencies
- Add proper mock imports for optional frameworks

**Target configuration** (based on metaclass-registry + arraybridge needs):
```python
"""Sphinx configuration for arraybridge."""

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'arraybridge'
copyright = '2025, Tristan Simas'
author = 'Tristan Simas'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autodoc typehints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Autosummary settings
autosummary_generate = True

# Mock imports for optional dependencies
autodoc_mock_imports = [
    'cupy',
    'torch',
    'tensorflow',
    'jax',
    'jaxlib',
    'pyclesperanto_prototype',
    'pyclesperanto',
]
```

#### 6. Create Documentation Structure
**Directory**: `docs/source/`

**Files to create**:

##### 6.1 `index.rst` - Main Documentation Index
- Project overview and tagline
- Key features list
- Installation instructions
- Quick example
- Table of contents linking to all sections
- Indices and tables

##### 6.2 `quickstart.rst` - Quick Start Guide
- Installation (base + optional frameworks)
- Basic conversion example
- Memory type detection example
- Simple decorator usage
- Stack/unstack utilities example

##### 6.3 `api.rst` - Complete API Reference
- Core conversion functions
- Memory type decorators
- Stack utilities
- Type definitions
- Exception classes
- Auto-generated API docs using autosummary

##### 6.4 `frameworks.rst` - Framework Integration Guide
- Supported frameworks table
- NumPy integration
- CuPy integration (CUDA)
- PyTorch integration
- TensorFlow integration
- JAX integration
- pyclesperanto integration (OpenCL)
- DLPack support and zero-copy conversions
- Framework availability detection

##### 6.5 `decorators.rst` - Decorator Usage Guide
- `@memory_types` base decorator
- Framework-specific decorators (@numpy, @cupy, @torch, etc.)
- Input/output type specification
- OOM recovery configuration
- Thread-local GPU streams
- Dtype conversion modes
- Contract validation

##### 6.6 `conversions.rst` - Memory Conversion Guide
- Conversion architecture
- DLPack zero-copy conversions
- CPU roundtrip fallback
- Supported conversion paths matrix
- GPU device management
- Performance considerations

##### 6.7 `gpu.rst` - GPU Features Guide
- Thread-local CUDA streams
- OOM (Out of Memory) recovery
- GPU device selection
- Multi-GPU support
- Framework-specific GPU features

##### 6.8 `examples.rst` - Comprehensive Examples
- Basic conversion workflows
- Decorator patterns
- Multi-framework pipelines
- GPU processing examples
- OOM recovery examples
- Stack processing examples
- Real-world use cases

##### 6.9 `advanced.rst` - Advanced Topics
- Dtype scaling and preservation
- Slice processing utilities
- Framework operations
- Custom conversion helpers
- Performance optimization
- Thread safety

##### 6.10 `contributing.rst` - Contribution Guide
- Development setup
- Running tests
- Code style (ruff, black, mypy)
- Adding new framework support
- Documentation guidelines

#### 7. Create Makefile and make.bat
**Files**: `docs/Makefile` and `docs/make.bat`
- Standard Sphinx build files for local documentation building
- Support for `make html`, `make clean`, etc.

### Findings

#### Current State Analysis
1. **Conflicting documentation systems**: Both MkDocs (`mkdocs.yml`) and Sphinx (`docs/sphinx/`) exist
2. **No ReadTheDocs config**: Missing `.readthedocs.yml` file
3. **Incorrect directory structure**: Sphinx docs in `docs/sphinx/` instead of `docs/`
4. **Incomplete Sphinx config**: Missing proper mocking, path setup, and extensions

#### arraybridge API Surface (from codebase analysis)
**Core Modules**:
- `converters.py`: `convert_memory()`, `detect_memory_type()`
- `decorators.py`: `@memory_types`, `@numpy`, `@cupy`, `@torch`, `@tensorflow`, `@jax`, `DtypeConversion`
- `stack_utils.py`: `stack_slices()`, `unstack_slices()`
- `types.py`: `MemoryType`, `CPU_MEMORY_TYPES`, `GPU_MEMORY_TYPES`, `SUPPORTED_MEMORY_TYPES`
- `exceptions.py`: `MemoryConversionError`
- `oom_recovery.py`: OOM detection and recovery
- `gpu_cleanup.py`: GPU memory management
- `dtype_scaling.py`: Dtype conversion and scaling
- `slice_processing.py`: Slice-based processing utilities
- `framework_config.py`: Framework-specific configuration
- `framework_ops.py`: Framework operations
- `conversion_helpers.py`: Internal conversion infrastructure
- `utils.py`: Utility functions

**Key Features to Document**:
1. **Automatic Memory Conversion**: DLPack + NumPy fallback
2. **Declarative Decorators**: Type-safe function decoration
3. **GPU Management**: Thread-local streams, OOM recovery
4. **Framework Support**: NumPy, CuPy, PyTorch, TensorFlow, JAX, pyclesperanto
5. **Stack Utilities**: 2D/3D array manipulation
6. **Dtype Preservation**: Automatic dtype handling
7. **Zero-Copy Operations**: DLPack-based GPU transfers

#### Documentation Content Requirements
Based on README.md and source code analysis:

**Must cover**:
- Installation for each framework combination
- Basic conversion workflows
- Decorator usage patterns
- GPU features (streams, OOM recovery)
- Framework-specific considerations
- Performance characteristics (zero-copy vs copy)
- Thread safety guarantees
- Error handling

**Code examples needed**:
- Simple NumPy ↔ PyTorch conversion
- Decorator-based function declaration
- Multi-framework pipeline
- GPU OOM recovery
- Stack/unstack operations
- Custom dtype conversion

#### 8. Detailed Documentation Content Outline

##### 8.1 index.rst Structure
```rst
arraybridge Documentation
=========================

**Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   frameworks
   decorators
   conversions
   gpu
   api
   examples
   advanced
   contributing

Overview
--------
arraybridge provides automatic memory type conversion between 6 major array/tensor frameworks
with zero-copy GPU operations, declarative decorators, and automatic OOM recovery.

Key Features
------------
* **Unified API**: Single interface for 6 frameworks
* **Automatic Conversion**: DLPack + NumPy fallback
* **Declarative Decorators**: @numpy, @torch, @cupy for type declarations
* **GPU Management**: Thread-local streams, OOM recovery
* **Zero Dependencies**: Only requires NumPy (frameworks optional)
* **Zero-Copy**: DLPack-based GPU transfers

Installation
------------
.. code-block:: bash

   # Base installation
   pip install arraybridge

   # With specific frameworks
   pip install arraybridge[torch]
   pip install arraybridge[cupy]

   # All frameworks
   pip install arraybridge[all]

Quick Example
-------------
.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type
   import numpy as np

   # Create NumPy array
   data = np.array([[1, 2], [3, 4]])

   # Convert to PyTorch
   torch_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)

   # Detect type
   mem_type = detect_memory_type(torch_data)  # 'torch'

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

##### 8.2 quickstart.rst Structure
- Installation section with all framework combinations
- Basic conversion example (NumPy → PyTorch → CuPy)
- Memory type detection
- Simple decorator usage (@numpy, @torch)
- Stack/unstack example
- Next steps (link to full guides)

##### 8.3 frameworks.rst Structure
**For each framework (NumPy, CuPy, PyTorch, TensorFlow, JAX, pyclesperanto)**:
- Framework description
- Installation command
- CPU/GPU support
- DLPack support
- Conversion examples to/from other frameworks
- Framework-specific considerations
- Performance notes

**Conversion Matrix Table**:
- Show all supported conversion paths
- Indicate zero-copy vs copy operations
- Note DLPack support

##### 8.4 decorators.rst Structure
- Base `@memory_types` decorator
- Framework-specific decorators
- Input/output type specification
- OOM recovery parameter
- Dtype conversion modes (DtypeConversion enum)
- Contract validation
- Thread-local GPU streams
- Real-world examples

##### 8.5 conversions.rst Structure
- Conversion architecture overview
- `convert_memory()` function
- `detect_memory_type()` function
- DLPack zero-copy mechanism
- CPU roundtrip fallback
- GPU device management
- Error handling
- Performance comparison table

##### 8.6 gpu.rst Structure
- Thread-local CUDA streams explanation
- OOM recovery mechanism
- GPU device selection
- Multi-GPU support
- Framework-specific GPU features
- Memory management best practices
- Troubleshooting GPU issues

##### 8.7 examples.rst Structure
**Basic Examples**:
- Simple conversion
- Decorator usage
- Stack operations

**Intermediate Examples**:
- Multi-framework pipeline
- GPU processing with OOM recovery
- Custom dtype conversion

**Advanced Examples**:
- Thread-safe parallel processing
- Multi-GPU workflows
- Integration with existing codebases

##### 8.8 api.rst Structure
```rst
API Reference
=============

Core Functions
--------------
.. autofunction:: arraybridge.convert_memory
.. autofunction:: arraybridge.detect_memory_type

Decorators
----------
.. autofunction:: arraybridge.memory_types
.. autofunction:: arraybridge.numpy
.. autofunction:: arraybridge.cupy
.. autofunction:: arraybridge.torch
.. autofunction:: arraybridge.tensorflow
.. autofunction:: arraybridge.jax

Stack Utilities
---------------
.. autofunction:: arraybridge.stack_slices
.. autofunction:: arraybridge.unstack_slices

Types
-----
.. autoclass:: arraybridge.MemoryType
   :members:
   :undoc-members:

.. autodata:: arraybridge.CPU_MEMORY_TYPES
.. autodata:: arraybridge.GPU_MEMORY_TYPES
.. autodata:: arraybridge.SUPPORTED_MEMORY_TYPES

Exceptions
----------
.. autoexception:: arraybridge.MemoryConversionError
   :members:
```

##### 8.9 advanced.rst Structure
- Dtype scaling internals
- Slice processing utilities
- Framework operations
- Custom conversion helpers
- Performance optimization techniques
- Thread safety guarantees
- Internal architecture overview

##### 8.10 contributing.rst Structure
- Development environment setup
- Running tests (pytest)
- Code style (ruff, black, mypy)
- Adding new framework support
- Documentation guidelines
- Pull request process
- Testing guidelines

### Implementation Draft
(To be filled after smell loop approval)

