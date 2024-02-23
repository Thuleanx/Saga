pub use uniform_buffer::UniformBufferSeries;

pub mod uniform_buffer {
    use std::mem::size_of;

    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    use super::super::super::buffers;
    use std::ptr::copy_nonoverlapping as memcpy;

    #[derive(Clone, Debug, Default)]
    pub struct UniformBufferSeries {
        buffers: Vec<vk::Buffer>,
        buffer_memories: Vec<vk::DeviceMemory>,
    }

    impl UniformBufferSeries {
        pub fn get_buffer_at_index(&self, image_index: usize) -> Option<vk::Buffer> { 
            if let Some(buffer) = self.buffers.get(image_index) {
                return Some(buffer.clone())
            }
            return None
        }

        pub fn get_memory_at_index(&self, image_index: usize) -> Option<vk::DeviceMemory> { 
            if let Some(memory) = self.buffer_memories.get(image_index) {
                return Some(memory.clone())
            }
            return None
        }

        pub fn get_buffers(&self) -> &Vec<vk::Buffer>{ &self.buffers }
        pub fn get_buffer_memories(&self) -> &Vec<vk::DeviceMemory>{ &self.buffer_memories }
    }

    pub unsafe fn create_series<T>(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        number_of_buffers : usize,
    ) -> Result<UniformBufferSeries> {
        let size_of_one_buffer = size_of::<T>() as u64;

        _create_series(
            instance, 
            device, 
            physical_device, 
            number_of_buffers, 
            size_of_one_buffer
        )
    }

    unsafe fn _create_series(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        number_of_buffers : usize,
        size_of_one_buffer: u64,
    ) -> Result<UniformBufferSeries> {

        let mut uniform_buffers : Vec<vk::Buffer> = vec![];
        let mut uniform_buffers_memory : Vec<vk::DeviceMemory> = vec![];

        for _ in 0..number_of_buffers {
            let (uniform_buffer, uniform_buffer_memory) = buffers::create_buffer(
                instance,
                device,
                physical_device,
                size_of_one_buffer,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | 
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }


        Ok(UniformBufferSeries { 
            buffers: uniform_buffers, 
            buffer_memories: uniform_buffers_memory 
        })
    }

    pub unsafe fn destroy_series(device: &Device, uniform_buffer_series: &UniformBufferSeries) {
        uniform_buffer_series.buffers
            .iter()
            .for_each(|b| device.destroy_buffer(*b, None));
        uniform_buffer_series.buffer_memories
            .iter()
            .for_each(|b| device.free_memory(*b, None));
    }

    pub unsafe fn update_uniform_buffer_series<T>(
        device: &Device,
        uniform_buffer_object: &T,
        uniform_buffer_series: &UniformBufferSeries,
        memory_index: usize,
    ) -> Result<()> {

        if let Some(memory) = uniform_buffer_series.get_memory_at_index(memory_index) {
            update_uniform_buffer(
                device, 
                uniform_buffer_object, 
                memory
            )?;
        }

        Ok(())
    }

    pub unsafe fn update_uniform_buffer<T>(
        device: &Device,
        uniform_buffer_object: &T,
        uniform_buffer_memory: vk::DeviceMemory,
    ) -> Result<()> {

        let memory = device.map_memory(
            uniform_buffer_memory, 
            0,
            size_of::<T>() as u64,
            vk::MemoryMapFlags::empty()
        )?;

        memcpy(uniform_buffer_object, memory.cast(), 1);

        device.unmap_memory(uniform_buffer_memory);

        Ok(())
    }
}

