#![allow(unused)]

/// This struct loads in vulkan from NVidia's special wrapper library, since the normal driver is not actually installed on Comet.
///
use libc::c_char;
use shared_library::{self};
use std::ffi::c_void;
use std::mem;
use std::path::Path;
use vulkano::instance::loader::{Loader, LoadingError};

/// Implementation of `Loader` that loads Vulkan from a dynamic library.
pub struct DynamicLibraryLoader {
    vk_lib: shared_library::dynamic_library::DynamicLibrary,
    get_proc_addr:
        extern "system" fn(instance: usize, pName: *const c_char) -> extern "system" fn() -> (),
}

impl DynamicLibraryLoader {
    /// Tries to load the dynamic library at the given path, and tries to
    /// load `vkGetInstanceProcAddr` in it.
    ///
    /// # Safety
    ///
    /// - The dynamic library must be a valid Vulkan implementation.
    ///
    pub unsafe fn new<P>(path: P) -> Result<DynamicLibraryLoader, LoadingError>
    where
        P: AsRef<Path>,
    {
        let vk_lib = shared_library::dynamic_library::DynamicLibrary::open(Some(path.as_ref()))
            .map_err(LoadingError::LibraryLoadFailure)?;

        let get_proc_addr = {
            let ptr: *mut c_void = vk_lib
                .symbol("vkGetInstanceProcAddr")
                .map_err(|_| LoadingError::MissingEntryPoint("vkGetInstanceProcAddr".to_owned()))
                .or_else(|_| {
                    vk_lib.symbol("glGetVkProcAddrNV").map_err(|_| {
                        LoadingError::MissingEntryPoint("glGetVkProcAddrNV".to_owned())
                    })
                })?;
            if ptr.is_null() {
                return Err(LoadingError::MissingEntryPoint(
                    "glGetVkProcAddrNV".to_owned(),
                ));
            }
            mem::transmute(ptr)
        };

        Ok(DynamicLibraryLoader {
            vk_lib,
            get_proc_addr,
        })
    }
}

unsafe impl Loader for DynamicLibraryLoader {
    #[inline]
    fn get_instance_proc_addr(
        &self,
        instance: usize,
        name: *const c_char,
    ) -> extern "system" fn() -> () {
        (self.get_proc_addr)(instance, name)
    }
}
