mod mycomplex;
use mycomplex::MyComplex;
mod generated;
mod loader;
mod summer;

#[macro_use]
mod my_consts;
pub const FRAME_SIZE: usize = my_consts::WIDTH * my_consts::HEIGHT; //Numbers
pub const FRAME_BYTES: usize = FRAME_SIZE * 8;
pub const TOTAL_BUFFER_SIZE: usize = my_consts::NUM_FRAMES * FRAME_SIZE;
pub const TOTAL_BUFFER_BYTES: usize = TOTAL_BUFFER_SIZE * 8;

pub const WORKGROUP_SIZE: usize =
    my_consts::SHADER_LAYOUT[0] * my_consts::SHADER_LAYOUT[1] * my_consts::SHADER_LAYOUT[2];
pub const NUM_WORKGROUPS: usize = (my_consts::DISPATCH_LAYOUT[0]
    * my_consts::DISPATCH_LAYOUT[1]
    * my_consts::DISPATCH_LAYOUT[2]) as usize;
pub const NUM_RELEVANT_WORKGROUNDS: usize = {
    let trunc_width = my_consts::WIDTH / my_consts::SHADER_LAYOUT[0];
    let width = trunc_width + (my_consts::WIDTH % my_consts::SHADER_LAYOUT[0] != 0) as usize;
    let trunc_height = my_consts::HEIGHT / my_consts::SHADER_LAYOUT[1];
    let height = trunc_height + (my_consts::HEIGHT % my_consts::SHADER_LAYOUT[1] != 0) as usize;
    width * height
};

//Stability verification
pub const STABILITY_COEFF: f32 = {
    1.0 - 2.0 * my_consts::DT * my_consts::HBAR / (my_consts::M * my_consts::DX * my_consts::DX)
};
pub const IS_STABLE: bool = STABILITY_COEFF > 0.0;
mod cpu;

use std::sync::Arc;

use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
use vulkano::buffer::{
    BufferAccess, BufferSlice, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer,
    TypedBufferAccess,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions, QueuesIter};
use vulkano::instance::{
    loader::FunctionPointers, loader::Loader, ApplicationInfo, Instance, InstanceExtensions,
    PhysicalDevice, QueueFamily, Version,
};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;
fn main() {
    run_gpu();
}

fn load_instance() -> Arc<Instance> {
    let libpath = std::env::args()
        .last()
        .unwrap_or("libGLX_nvidia.so.0".to_owned());
    let default_instance = Instance::new(None, &InstanceExtensions::none(), None);
    default_instance
        .or_else(|_| {
            let loader = unsafe { crate::loader::DynamicLibraryLoader::new(libpath)? };
            let funcs: FunctionPointers<Box<dyn Loader + Send + Sync>> =
                FunctionPointers::new(Box::new(loader));
            let extens = InstanceExtensions::supported_by_core_with_loader(&funcs).unwrap();
            println!("{:?}", extens);
            let res =
                Instance::with_loader(funcs, None, &InstanceExtensions::none(), None).unwrap();
            Result::<_, vulkano::instance::InstanceCreationError>::Ok(res)
        })
        .unwrap()
}

fn create_framebuffer(
    device: &Arc<Device>,
    initial_frame: &[MyComplex],
) -> Arc<CpuAccessibleBuffer<[[f32; 2]]>> {
    let input_buffer = unsafe {
        CpuAccessibleBuffer::uninitialized_array(
            device.clone(),
            FRAME_SIZE * my_consts::NUM_FRAMES,
            BufferUsage::all(),
            false,
        )
        .unwrap()
    };
    input_buffer
        .write()
        .unwrap()
        .iter_mut()
        .zip(initial_frame.iter())
        .for_each(|(out, inp): (&mut [f32; 2], _)| {
            out[0] = inp.re;
            out[1] = inp.im;
        });

    input_buffer
}

fn load_vulkan<'a>(instance: &'a Arc<Instance>) -> (Arc<Device>, QueueFamily<'a>, QueuesIter) {
    let (first, queue_family) = PhysicalDevice::enumerate(&instance)
        .flat_map(|phys| phys.queue_families().map(move |fam| (phys, fam)))
        .filter(|(_, qf): &(_, QueueFamily)| qf.supports_compute())
        .next()
        .unwrap();
    let (device, queues) = Device::new(
        first,
        first.supported_features(),
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,

            ..DeviceExtensions::none()
        },
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    (device, queue_family, queues)
}

fn build_summer_command(
    device: Arc<Device>,
    pbuffer: Arc<impl BufferAccess + Send + Sync + 'static>,
    queue_family: QueueFamily,
) -> impl CommandBuffer {
    let summer = summer::Shader::load(device.clone()).unwrap();
    let summer_pipeline =
        Arc::new(ComputePipeline::new(device.clone(), &summer.main_entry_point(), &()).unwrap());
    let summer_layout = summer_pipeline.layout().descriptor_set_layout(0).unwrap();
    let summer_desc = PersistentDescriptorSet::start(summer_layout.clone())
        .add_buffer(pbuffer)
        .unwrap()
        .build()
        .unwrap();
    let summer_command = AutoCommandBufferBuilder::primary_simultaneous_use(device, queue_family)
        .unwrap()
        .dispatch([1, 1, 1], summer_pipeline, summer_desc, ())
        .unwrap()
        .build()
        .unwrap();
    summer_command
}

fn run_gpu() {
    debug_assert!(IS_STABLE, "Coeff: {:?}", STABILITY_COEFF);
    let mut initial_frame = vec![MyComplex::zero(); FRAME_SIZE];
    cpu::initialize(&mut initial_frame);
    let initial_p = cpu::sum_p(&initial_frame);

    let start = Instant::now();
    let instance = load_instance();
    let (device, queue_family, mut queues) = load_vulkan(&instance);
    let queue = queues.next().unwrap();

    let raw_allocation = create_framebuffer(&device, &initial_frame);

    let mut data = [0.0; NUM_WORKGROUPS + 1];
    data[NUM_RELEVANT_WORKGROUNDS] = initial_p;
    let pbuffer = unsafe {
        CpuAccessibleBuffer::uninitialized_array(
            device.clone(),
            data.len(),
            BufferUsage::all(),
            false,
        )
        .unwrap()
    };
    pbuffer.write().unwrap().copy_from_slice(&data);

    let pipeline = {
        let shader = generated::Shader::load(device.clone()).unwrap();
        Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap())
    };
    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(raw_allocation.clone())
            .unwrap()
            .add_buffer(pbuffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    let summer_buffer = Arc::new(build_summer_command(
        device.clone(),
        pbuffer.clone(),
        queue_family.clone(),
    ));

    let mut current_queue: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone()));
    for idx in 0..my_consts::NUM_FRAMES -1{
        let command_buffer =
            AutoCommandBufferBuilder::primary(device.clone(), queue_family.clone())
                .unwrap()
                .dispatch(
                    my_consts::DISPATCH_LAYOUT,
                    pipeline.clone(),
                    set.clone(),
                    idx as u32,
                )
                .unwrap()
                .build()
                .unwrap();
        let nxtfut = current_queue
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_semaphore()
            .then_execute_same_queue(summer_buffer.clone())
            .unwrap();
        if idx == my_consts::NUM_FRAMES - 1{
            current_queue = Box::new(nxtfut);
        }
        else {
            current_queue = Box::new(nxtfut.then_signal_semaphore());
        }
    }
    current_queue
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let end = Instant::now();
    println!(
        "{} / {}",
        end.duration_since(start).as_millis(),
        (end.duration_since(start).as_millis() as f32) / (my_consts::NUM_FRAMES as f32)
    );
    let mut output_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("vulkan_out.data")
        .unwrap();
    let framebuffer_handle = raw_allocation.read().unwrap();
    let frames_iter = framebuffer_handle.chunks_exact(FRAME_SIZE);
    for current_frame in frames_iter {
        let bbuffer = write_frame(current_frame);
        output_file.write_all(&bbuffer).unwrap();
    }
}

fn write_frame(frame: &[[f32; 2]]) -> Vec<u8> {
    let mut retvl = Vec::with_capacity(frame.len() * 8);
    for cur in frame {
        let n = MyComplex::new(cur[0], cur[1]);
        retvl.extend_from_slice(&n.to_bytes());
    }
    retvl
}
