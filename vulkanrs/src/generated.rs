use std::ffi::CStr;
use std::sync::Arc;
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages,
};
use vulkano::descriptor::pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange};
use vulkano::device::Device;
use vulkano::pipeline::shader::{
    ComputeEntryPoint, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule,
};
use vulkano::OomError;

pub struct Shader {
    shader: Arc<ShaderModule>,
}
impl Shader {
    #[inline]
    pub fn load(device: Arc<Device>) -> Result<Shader, OomError> {
        let bytes: &[u8] = include_bytes!("../res/stepper.spv");
        unsafe {
            Ok(Shader {
                shader: ShaderModule::new(device, bytes)?,
            })
        }
    }
    #[inline]
    pub fn main_entry_point(&self) -> ComputeEntryPoint<(), Layout> {
        unsafe {
            self.shader.compute_entry_point(
                CStr::from_bytes_with_nul_unchecked(b"main\0"),
                Layout(ShaderStages {
                    compute: true,
                    ..ShaderStages::none()
                }),
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layout(pub ShaderStages);
unsafe impl PipelineLayoutDesc for Layout {
    fn num_sets(&self) -> usize {
        1usize
    }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0usize => Some(2usize),
            _ => None,
        }
    }
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        match (set, binding) {
            (0usize, 0usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            (0usize, 1usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            _ => None,
        }
    }
    fn num_push_constants_ranges(&self) -> usize {
        1usize
    }
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num == 0 {
            Some(PipelineLayoutDescPcRange {
                offset: 0,
                size: 4usize,
                stages: ShaderStages::all(),
            })

        }
        else { None }
    }
}
#[allow(unused)]
pub fn check_extensions(device: &Arc<Device>) {
    if !device.loaded_extensions().khr_storage_buffer_storage_class {
        panic!(
            "Device extension {:?} required",
            "khr_storage_buffer_storage_class"
        );
    }
} 
