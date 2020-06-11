# Vulkan vs Cuda for Short-Lived Compute Simulations

## General Purpose

The Khronos working group has developed multiple Free and Open Source graphics APIs, including OpenGL, OpenCL, and the latest, Vulkan. The latter is currently being pushed as a general purpose API, usable for both drawing images and general computation, both on-hardware and via a web browser through the WGPU standard. The NVIDIA Cuda API, meanwhile, has been the GPU compute standard for many years, even with its propietary and NVIDIA-exclusive nature, due to ease of use and performance. This project aims to compare the performance of the existing NVIDIA Cuda API with the up-and-coming Vulkan Compute API for the purpose of short-lived simulations.

## Setup

### Requirements

The Vulkan implementation requires the `cargo` build tool and at least one Vulkan device driver to be installed. The Vulkan client loader is currently provided via a local `libvulkan.so.1` file, but this can be removed on systems which already contain a `libvulkan` client loader. 

The Cuda implementation requires the Cuda runtime. 

The `project_compile.py` and `project_run.py` scripts require Python version 3.8 or higher, the `glslangValidator` GLSL to SPIRV compiler, the `nvcc` Cuda compiler, the `gcc` C compiler, and a bash-like shell. 

The `videoencoder.c` and `make_rgb.c` utilities require the `libavcodec` library as well as a Unix-like operating system due to reliance on `mmap` and `unistd` low-level calls. 

### Configuration 

The project is compiled and ran via the `project_compile.py` and `project_run.py` scripts, respectively. These are called without arguments, and instead are configured via the `config.ini` file. These scripts allow for easily building both the Cuda implementation and the Vulkan implementation without needing to worry about matching constants, since the script sets up the constant values from the `config.ini` file. 

The configuration parameters are as follows:

| Key           | Description                                                       |
| :------------ | :---------------------------------------------------------------- |
| `start 1`     | The x position *index* of the start of the first of the 2 slits.  |
| `end 1`       | The x position *index* of the end of the first of the 2 slits.    |
| `start 2`     | The x position *index* of the start of the second of the 2 slits. |
| `end 2`       | The x position *index* of the end of the second of the 2 slits.   |
| `slit height` | The height that both slits are at.                                |
| `width`       | The number of gridpoints in the x dimension.                      |
| `height`      | The number of gridpoints in the y dimension.                      |
| `dx`          | The size of the spacial grid steps.                               |
| `num frames`  | The number of timesteps to simulate.                              |
| `dt`          | The timestep to use.                                              |
| `hbar`        | The value of the reduced Planck constant to use.                  |
| `m`           | The mass of the particle in the simulation.                       |
| `layout x`    | The size of the Vulkan workgroup / CUDA block 's x axis.          |
| `layout y`    | The size of the Vulkan workgroup / CUDA block 's y axis.          |
| `dispatch x`  | The size of the Vulkan dispatch /CUDA grid 's x axis              |
| `dispatch y`  | The size of the Vulkan dispatch /CUDA grid 's y axis              |



### Vulkan Implementation
The `vulkanrs` folder contains the Vulkan implementation, written in the Rust programming language, as well as the compute shaders written in GLSL (which are within `vulkanrs/res`). Two separate shaders were used: `vulkanrs/res/stepper.comp`, which uses a finite-difference method to forward-propogate the simulation, and `vulkanrs/res/summer.comp`, which uses a parallel summing algorithm to calculate the total matrix probability for normalization purposes. 

The state is initialized in the `initialize` function of `vulkanrs/src/cpu.rs` before the timer starts, with CPU memory also being allocated. Afer the timer starts, the Vulkan API is initialized, and two buffers of GPU memory are allocated: a CPU-accessible buffer which is used for simulation state, and a scratch buffer for renormalization. The stepper then builds a queue of calls to `stepper.comp` followed by `summer.comp` for each frame. Finally, the program waits for the queue to finish execution before reporting the total time and outputting the raw data to a `vulkan_out.data` file. 

This implementation includes some extra code in the `vulkanrs/src/cpu.rs` file which can run the simulation on the CPU; this was used for debugging purposes only.

The Vulkan implementation takes an extra command line argument of the path to the `libvulkan.so` shared library to use. This allows for running the Vulkan implementation on the Comet supercomputer, which does not currently have the client API installed even though the Vulkan device drivers are bundled in the NVIDIA graphics API stack. 

### Cuda Implementation

The Cuda implementation is located in the `cudac` folder. It is divided between `stepper.cu`, which contains the shader analogous to `vulkanrs/res/stepper.comp` of the Vulkan version, `summer.cu`, equivalent to `vulkanrs/res/summer.comp`, and `main.cu`, which contains CPU side of the implementation. In addition, the `cuda-samples` folder contains the code of the NVIDIA samples repository, located at `https://github.com/NVIDIA/cuda-samples`, for access to multiple helper headers. 

The code is a near one-for-one reimplementation of the Vulkan equivalent, except utilizing the Cuda API and the C programming language. Things to note immediately include the significantly less boilerplate at the initialization stage and the requirement to manually free all resources due to the C language's lack of the Rust compiler's borrow checker. 


### Utilities and helpers
Aside from the build and run scripts, the `utils` folder contains two separate tools which can be chained to produce a video representation of the different output `.data` files. The `make_rgb.c` file converts the `.data` file format of raw 32-bit float pairs into RGB pixel data, and the `videoencoder.c` file uses the `libavcodec` library to produce a video file from that pixel data. 



## Data and Results

The `project_run.py` script randomly chooses to start with either the Vulkan or Cuda implementation. Each report two numbers: the total execution time and the execution time divided by the number of frames, both in milliseconds. All trials were ran on an NVIDIA Tegra X1 System-on-a-Chip, via the Linux for Tegra Ubuntu derivative operating system. Note that while the raw output data is not included, it is currently available upon request; it is simply too big to upload due to the sizes involved. 

| Trial | Vulkan Time (ms) | Vulkan Time-Per-Frame (ms/frame) | CUDA Time(ms) | CUDA Time-Per-Frame (ms/frame) |
| ----: | ---------------: | -------------------------------: | ------------: | -----------------------------: |
|     1 |           165753 |                            331.5 |          3200 |                              6 |
|     2 |           165332 |                            330.7 |          3397 |                              6 |
|     2 |           168531 |                            337.1 |          3514 |                              7 |




As seen in the data, the Vulkan version required significantly more time to accomplish the task. It is currently hypothesized that this is due to the Vulkan version requiring many more CPU synchronizations for the framebuffer due to how Vulkan manages buffer locks, though this has not been verified. In addition, it may be due to the Rust version requiring many small short-lived heap allocations due to the type system of the API. Finally, the C `clock` API is not as precise nor monotomic as the Rust timing API is, introducing possible truncation and/or optimization errors. 

## `TODO`

This experiment, as a whole, contains numerous issues and concerns to be addressed in the future. These include:
    
    a) more idiomatic implementations of both graphics APIs; this project ended up with C-like implementations for both Vulkan and Cuda due to the desire to match implemenations exactly, even at the expense of real-world accuracy.

    b) testing using an algorithm requiring little to now renomalization; the renormalization requirement ended up adding extra bottlenecks to the code due to its sequential nature and causing extra synchronization flags to be used. 

    c) more devices to test; unfortunately due to lack of available resources few different GPUs could be tested and only a limitted number of times, possibly skewing the results. 