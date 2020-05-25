use std::ffi::CStr;
use std::sync::Arc;
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages,
};
use vulkano::descriptor::pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange};
use vulkano::device::Device;
use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
use vulkano::pipeline::shader::{
    ComputeEntryPoint, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule,
    SpecializationMapEntry,
};
use vulkano::OomError;

pub struct Shader {
    shader: Arc<ShaderModule>,
}
impl Shader {
    #[inline]
    pub fn load(device: Arc<Device>) -> Result<Shader, OomError> {
        let bytes: &[u8] = include_bytes!("../res/summer.spv");
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MainInput;
unsafe impl ShaderInterfaceDef for MainInput {
    type Iter = MainInputIter;
    fn elements(&self) -> MainInputIter {
        MainInputIter { num: 0 }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct MainInputIter {
    num: u16,
}
impl Iterator for MainInputIter {
    type Item = ShaderInterfaceDefEntry;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 0usize - self.num as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for MainInputIter {}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MainOutput;
unsafe impl ShaderInterfaceDef for MainOutput {
    type Iter = MainOutputIter;
    fn elements(&self) -> MainOutputIter {
        MainOutputIter { num: 0 }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct MainOutputIter {
    num: u16,
}
impl Iterator for MainOutputIter {
    type Item = ShaderInterfaceDefEntry;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 0usize - self.num as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for MainOutputIter {}
#[derive(Debug, Clone)]
pub struct Layout(pub ShaderStages);
unsafe impl PipelineLayoutDesc for Layout {
    fn num_sets(&self) -> usize {
        1usize
    }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0usize => Some(1usize),
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
            _ => None,
        }
    }
    fn num_push_constants_ranges(&self) -> usize {
        0usize
    }
    fn push_constants_range(&self, _: usize) -> Option<PipelineLayoutDescPcRange> {
        None
    }
}
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SpecializationConstants {}
impl Default for SpecializationConstants {
    fn default() -> SpecializationConstants {
        SpecializationConstants {}
    }
}
unsafe impl SpecConstsTrait for SpecializationConstants {
    fn descriptors() -> &'static [SpecializationMapEntry] {
        static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
        &DESCRIPTORS
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
