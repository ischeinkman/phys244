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
    //cpu::run();
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
    let instance = load_instance();
    let (device, queue_family, mut queues) = load_vulkan(&instance);
    let queue = queues.next().unwrap();

    let mut initial_frame = vec![MyComplex::zero(); FRAME_SIZE];
    cpu::initialize(&mut initial_frame);
    let initial_p = cpu::sum_p(&initial_frame);
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

    let mfut = {
        let command_buffer =
            AutoCommandBufferBuilder::primary(device.clone(), queue_family.clone())
                .unwrap()
                .dispatch(my_consts::DISPATCH_LAYOUT, pipeline.clone(), set.clone(), 0)
                .unwrap()
                .build()
                .unwrap();
        let retvl = command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_execute_same_queue(summer_buffer.clone())
            .unwrap();
        retvl.flush().unwrap();
        retvl
    };
    let mut mfut: Box<dyn GpuFuture> = Box::new(mfut);
    let mfutr = Arc::new(mfut.then_signal_fence_and_flush().unwrap());
    let mut cur_signal = mfutr.clone();
    mfut = Box::new(mfutr);
    for idx in 0..my_consts::NUM_FRAMES {
        eprintln!("KKK: {}", idx);
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
        let nxtfut = command_buffer
            .execute_after(mfut, queue.clone())
            .unwrap()
            .then_execute_same_queue(summer_buffer.clone())
            .unwrap();
        if let Ok(()) = cur_signal.wait(None) {
            let wrapped_nxt : Box<dyn GpuFuture> = Box::new(nxtfut);
            let wrapped_nxt = Arc::new(wrapped_nxt.then_signal_fence_and_flush().unwrap());
            cur_signal = wrapped_nxt.clone();
            mfut = Box::new(wrapped_nxt);
        }
        else {
            nxtfut.flush().unwrap();
            mfut = Box::new(nxtfut);

        }
    }

    mfut.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    println!(
        "PBuffer ({} / {}): ",
        pbuffer.read().unwrap().len(),
        pbuffer.size()
    );

    let wk_to_gridstart = |idx: usize| {
        let wk_x = idx % my_consts::DISPATCH_LAYOUT[0] as usize;
        let gbl_x = wk_x * my_consts::SHADER_LAYOUT[0];
        let wk_y = idx / my_consts::DISPATCH_LAYOUT[1] as usize;
        let gbl_y = wk_y * my_consts::SHADER_LAYOUT[1];
        (gbl_x, gbl_y)
    };

    pbuffer
        .read()
        .unwrap()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p != 0.0)
        .for_each(|(idx, p)| {
            let grid = wk_to_gridstart(idx);
            println!("    {} /({}, {}) : {}", idx, grid.0, grid.1, p);
        });
    println!(
        "Inbuffer {} / {}: ",
        raw_allocation.read().unwrap().len(),
        raw_allocation.size()
    );
    raw_allocation
        .read()
        .unwrap()
        .iter()
        .copied()
        .enumerate()
        .take(FRAME_SIZE)
        .map(|(idx, p)| ((idx % my_consts::WIDTH, idx / my_consts::WIDTH), p))
        .filter(|(_, p)| p[0] * p[0] + p[1] * p[1] != 0.0)
        .for_each(|v| println!("    {:?}", v));
    println!("Outbuffer :");
    raw_allocation
        .read()
        .unwrap()
        .iter()
        .copied()
        .skip(FRAME_SIZE)
        .take(FRAME_SIZE)
        .enumerate()
        .map(|(idx, p)| ((idx % my_consts::WIDTH, idx / my_consts::WIDTH), p))
        .filter(|(_, p)| p[0] * p[0] + p[1] * p[1] != 0.0)
        .for_each(|v| println!("    {:?}", v));
    let mut psum_a: f32 = 0.0;
    for a in pbuffer.read().unwrap().iter().take(256) {
        psum_a += a;
    }
    let psum_b: f32 = raw_allocation
        .read()
        .unwrap()
        .iter()
        .skip(FRAME_SIZE)
        .take(FRAME_SIZE)
        .copied()
        .map(|[na, nb]| na * na + nb * nb)
        .sum();
    println!("True : {}", psum_b);
    println!("Found: {} ", psum_a);
    for idx in 0..initial_frame.len() {
        let bufv = raw_allocation.read().unwrap()[idx];
        if bufv[0] != initial_frame[idx].re && bufv[1] != initial_frame[idx].im {
            println!("MIX? {:?} => {:?} v {:?}", idx, bufv, initial_frame[idx]);
        }
    }

    println!("Post summer pbuffer:");
    pbuffer
        .read()
        .unwrap()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p != 0.0)
        .for_each(|(idx, p)| {
            println!("    {} : {}", idx, p);
        });
}
