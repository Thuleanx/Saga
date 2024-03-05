use crate::core::graphics::{
    graphics_utility, CPUMesh, GPUMesh, Graphics, Image, ImageSampler, LoadedImage,
    UniformBufferSeries,
};
use bevy_app::App;
use bevy_ecs::{
    component::Component,
    system::{Commands, Res},
};
use cgmath::{point3, vec3, Angle, Deg, Matrix, Matrix3, Transform, Zero};

use self::saga_renderer::UniformBufferObject;

type Mat4 = cgmath::Matrix4<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Quat = cgmath::Quaternion<f32>;

#[derive(Component)]
struct Position(Vec3);

impl Position {
    fn x(&self) -> f32 {
        self.0.x
    }
    fn y(&self) -> f32 {
        self.0.y
    }
    fn z(&self) -> f32 {
        self.0.z
    }
}

#[derive(Component)]
struct Rotation(Quat);

impl Rotation {
    fn forward(&self) -> Vec3 {
        self.0 * vec3(0.0, 0.0, 1.0)
    }

    fn right(&self) -> Vec3 {
        self.0 * vec3(-1.0, 0.0, 0.0)
    }

    fn up(&self) -> Vec3 {
        self.0 * vec3(0.0, 1.0, 0.0)
    }
}

#[derive(Component)]
struct Camera {
    field_of_view: cgmath::Rad<f32>,
    far_plane_distance: f32,
    near_plane_distance: f32,
    width: u32,
    height: u32,
}

impl Camera {
    fn calculate_projection_matrix(&self) -> Mat4 {
        // this matrix transforms the typical view space into an intermediate space
        // where the forward direction is the direction that the camera
        // fulstrum captures, and the up direction is the downwards camera direction
        let intermediate_matrix: Mat4 = Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        let inverse_tan_half_fov: f32 = 1.0 / Angle::tan(self.field_of_view / 2.0);
        let inverse_aspect_ratio: f32 = match self.height {
            0 => 0.0 as f32,
            _ => (self.height as f32) / (self.width as f32),
        };

        let projection_matrix_c0r0: f32 = inverse_tan_half_fov * inverse_aspect_ratio;
        let projection_matrix_c1r1: f32 = inverse_tan_half_fov;
        let projection_matrix_c2r2: f32 =
            self.far_plane_distance / (self.far_plane_distance - self.near_plane_distance);
        let projection_matrix_c2r3: f32 = 1.0;
        let projection_matrix_c3r2: f32 = match self.far_plane_distance == self.near_plane_distance
        {
            true => 0.0 as f32,
            false => {
                -self.far_plane_distance * self.near_plane_distance
                    / (self.far_plane_distance - self.near_plane_distance)
            }
        };

        // according to the following:
        // https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/
        // this should transform the intermediate space into clip space in vulkan
        let projection_matrix: Mat4 = Mat4::new(
            projection_matrix_c0r0,
            0.0,
            0.0,
            0.0,
            0.0,
            projection_matrix_c1r1,
            0.0,
            0.0,
            0.0,
            0.0,
            projection_matrix_c2r2,
            projection_matrix_c2r3,
            0.0,
            0.0,
            projection_matrix_c3r2,
            0.0,
        );

        projection_matrix * intermediate_matrix
    }

    fn calculate_view_matrix(&self, position: &Position, rotation: &Rotation) -> Mat4 {
        let look: Vec3 = rotation.forward();
        let up: Vec3 = rotation.up();
        let right: Vec3 = look.cross(up);

        // this is row major
        //  in view space, camera captures in the opposite direction of the look
        //  vector by convention
        let rotation_matrix: Mat4 = Mat4::new(
            right.x, up.x, -look.x, 0.0, right.y, up.y, -look.y, 0.0, right.z, up.z, -look.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let translation_matrix: Mat4 = Mat4::new(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -position.x(),
            -position.y(),
            -position.z(),
            1.0,
        );

        rotation_matrix * translation_matrix
    }
}

#[derive(Component)]
pub struct CameraRenderingInfo {
    view: Mat4,
    projection: Mat4,
    uniform_buffers: UniformBufferSeries,
}

#[derive(Component)]
struct Mesh {
    gpu_mesh: GPUMesh,
}

#[derive(Component)]
struct MainTexture {
    texture: LoadedImage,
    sampler: ImageSampler,
}

fn spawn_camera(window: Res<saga_window::Window>, graphics: Res<Graphics>, mut commands: Commands) {
    log::info!("Spawn camera");
    let position = Position(vec3(4.0, 4.0, 4.0));
    let rotation = Rotation(
        Matrix3::<f32>::look_at_rh(
            point3(0.0, 0.0, 0.0),
            point3(2.0, 2.0, 1.1),
            vec3(0.0, 0.0, 1.0),
        )
        .transpose()
        .into(),
    );

    let size = window.window.inner_size();

    let uniform_buffers = unsafe {
        UniformBufferSeries::create_from_graphics::<UniformBufferObject>(&graphics).unwrap()
    };

    unsafe {
        uniform_buffers.bind_to_graphics::<UniformBufferObject>(&graphics, 0);
    };

    let camera = Camera {
        field_of_view: Deg(45.0).into(),
        far_plane_distance: 10.0,
        near_plane_distance: 0.1,
        width: size.width,
        height: size.height,
    };

    let view = camera.calculate_view_matrix(&position, &rotation);
    let projection = camera.calculate_projection_matrix();

    let spawn = commands.spawn((
        position,
        rotation,
        camera,
        CameraRenderingInfo {
            view,
            projection,
            uniform_buffers,
        },
    ));
}

fn spawn_mesh(graphics: Res<Graphics>, mut commands: Commands) {
    log::info!("Spawn mesh");

    let position = vec3(4.0, 4.0, 4.0);
    let rotation = Quat::zero();

    let path_to_obj = std::env::current_dir()
        .unwrap()
        .join("assets")
        .join("meshes")
        .join("Imphat.obj");
    let path_to_texture = std::env::current_dir()
        .unwrap()
        .join("assets")
        .join("meshes")
        .join("imphat_diffuse.png");

    let cpu_mesh = unsafe { CPUMesh::load_from_obj(&graphics, &path_to_obj) };

    if cpu_mesh.len() == 0 {
        log::error!(
            "Provided obj file at path {:?} yields no object",
            &path_to_obj
        );
        return;
    }

    let gpu_mesh = unsafe { GPUMesh::create(&graphics, &cpu_mesh[0]).unwrap() };

    let texture = Image::load(&path_to_texture).unwrap();

    let loaded_texture = unsafe { LoadedImage::create(&graphics, &texture).unwrap() };
    let texture_sampler = unsafe { ImageSampler::create_from_graphics(&graphics).unwrap() };

    unsafe {
        graphics_utility::bind_image_sampler(&graphics, &texture_sampler, &loaded_texture, 1);
    }

    let spawn = commands.spawn((
        Position(position),
        Rotation(rotation),
        Mesh { gpu_mesh },
        MainTexture {
            texture: loaded_texture,
            sampler: texture_sampler,
        },
    ));
}

mod saga_renderer {
    use anyhow::Result;
    use bevy_app::Plugin as BevyPlugin;
    use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
    use bevy_ecs::system::ResMut;
    use cgmath::{vec3, Deg, Matrix4};
    use vulkanalia::vk;

    use crate::core::graphics::{Graphics, StartRenderResult};

    use super::{saga_window::Window, Camera, CameraRenderingInfo, MainTexture, Mesh};

    pub struct Plugin;

    impl BevyPlugin for Plugin {
        fn build(&self, app: &mut bevy_app::App) {
            app.add_event::<Resize>()
                .init_schedule(Cleanup)
                .add_systems(
                    bevy_app::Update,
                    draw.pipe(handle_swapchain_recreate)
                        .pipe(recreate_swapchain)
                        .pipe(log_error_result),
                )
                .add_systems(Cleanup, cleanup_camera)
                .add_systems(Cleanup, cleanup_meshes)
                .add_systems(
                    bevy_app::PostStartup,
                    build_command_buffer.pipe(log_error_result),
                )
                .add_systems(bevy_app::Update, camera_on_screen_resize);
        }
    }

    // Events
    #[derive(bevy_ecs::event::Event)]
    pub struct Resize;

    // Schedules
    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    pub struct Cleanup;

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct UniformBufferObject {
        pub model: Matrix4<f32>,
        pub view: Matrix4<f32>,
        pub proj: Matrix4<f32>,
    }

    fn camera_on_screen_resize(
        mut resize_event: EventReader<Resize>,
        window: Res<Window>,
        mut cameras: Query<(&mut Camera, &mut CameraRenderingInfo)>,
    ) {
        let resize_happens = resize_event.read().next().is_some();
        if !resize_happens {
            return;
        }

        let size = window.window.inner_size();

        for (mut camera, mut camera_rendering_info) in cameras.iter_mut() {
            log::info!("Resize camera and recalculated projection matrix");
            camera.width = size.width;
            camera.height = size.height;
            camera_rendering_info.projection = camera.calculate_projection_matrix();
        }
    }

    fn build_command_buffer(
        graphics: Res<Graphics>,
        meshes: Query<(&Mesh, &MainTexture)>,
    ) -> Result<()> {
        build_command_buffer_from_graphics(&graphics, meshes)
    }

    fn build_command_buffer_from_graphics(
        graphics: &Graphics,
        meshes: Query<(&Mesh, &MainTexture)>,
    ) -> Result<()> {
        log::info!("Build command buffer");
        unsafe {
            graphics.record_command_buffers(
                |graphics: &Graphics, command_buffer: vk::CommandBuffer, index: usize| unsafe {
                    graphics.bind_descriptor_set_indexed(command_buffer, index);

                    for (mesh, main_texture) in &meshes {
                        mesh.gpu_mesh.bind(graphics, command_buffer);
                        mesh.gpu_mesh.draw(graphics, command_buffer);
                    }
                },
            )
        }
    }

    fn draw(
        window: Res<Window>,
        mut graphics: ResMut<Graphics>,
        camera_query: Query<(&Camera, &CameraRenderingInfo)>,
    ) -> Result<bool> {
        let image_index = unsafe {
            match graphics.start_render(&window.window) {
                StartRenderResult::Normal(Ok(image_index)) => image_index,
                StartRenderResult::Normal(Err(e)) => panic!("{}", e),
                StartRenderResult::ShouldRecreateSwapchain => {
                    return Ok(true);
                }
            }
        };

        for (camera, camera_rendering_info) in &camera_query {
            let time = graphics.get_start_time().elapsed().as_secs_f32();
            let model = Matrix4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0 * time))
                * Matrix4::from_axis_angle(vec3(1.0, 0.0, 0.0), Deg(90.0));

            let view = camera_rendering_info.view;
            let proj = camera_rendering_info.projection;

            let ubo = UniformBufferObject { model, view, proj };

            unsafe {
                graphics.update_uniform_buffer_series(
                    &camera_rendering_info.uniform_buffers,
                    image_index,
                    &ubo,
                )?;
            }
        }

        unsafe {
            let should_recreate_swapchain = graphics.end_render(&window.window, image_index);
            should_recreate_swapchain
        }
    }

    fn handle_swapchain_recreate(In(should_recreate_swapchain): In<Result<bool>>) -> bool {
        let recreate_swapchain = match should_recreate_swapchain {
            Ok(should_recreate_swapchain) => should_recreate_swapchain,
            Err(err) => {
                log::error!("Error occur during render: {}", err);
                false
            }
        };
        recreate_swapchain
    }

    fn recreate_swapchain(
        In(should_recreate_swapchain): In<bool>,
        window: Res<Window>,
        mut graphics: ResMut<Graphics>,
        meshes: Query<(&Mesh, &MainTexture)>,
    ) -> Result<()> {
        if !should_recreate_swapchain {
            return Ok(());
        }

        log::info!("Swapchain recreation");

        let window = &window.window;

        unsafe {
            // Must wait for rendering is done before swapchain init_resources
            // can be released
            graphics.device_wait_idle()?;

            graphics.recreate_swapchain(&window)?;

            build_command_buffer_from_graphics(&graphics, meshes)?;

            graphics.continue_after_swapchain_construction();
        }

        Ok(())
    }

    fn log_error_result(In(result): In<Result<()>>) {
        if let Err(error) = result {
            log::error!("Error: {}", error)
        }
    }

    fn discard_error_result(In(result): In<Result<()>>) {}

    fn cleanup_meshes(
        graphics: Res<Graphics>,
        meshes: Query<(&Mesh, &MainTexture)>,
    ) {
        log::info!("[Saga] Cleaning up all meshes");
        let graphics = graphics.as_ref();
        for (mesh, main_texture) in &meshes {
            log::info!("Cleanup...");
            unsafe {
                mesh.gpu_mesh.destroy(&graphics);
                main_texture.texture.destroy_with_graphics(&graphics);
                main_texture.sampler.destroy_with_graphics(&graphics);
            }
        }
    }

    fn cleanup_camera(
        graphics: Res<Graphics>,
        cameras: Query<&CameraRenderingInfo>,
    ) {
        log::info!("[Saga] Cleaning up all cameras");
        let graphics = graphics.as_ref();
        for camera_rendering_info in &cameras {
            log::info!("Cleanup...");
            unsafe {
                camera_rendering_info
                    .uniform_buffers
                    .destroy_uniform_buffer_series(&graphics);
            }
        }
    }
}

mod saga_window {
    use crate::{core::graphics::Graphics, doomclone::app::saga_renderer::{Cleanup, Resize}};
    use anyhow::Result;
    use bevy_app::{App, AppExit, Plugin};
    use bevy_ecs::system::Resource;
    use winit::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window as WinitWindow, WindowBuilder},
    };

    pub struct WindowPlugin;

    impl Plugin for WindowPlugin {
        fn build(&self, app: &mut App) {
            app.set_runner(winit_event_runner);
        }
    }

    #[derive(Resource)]
    pub struct Window {
        pub window: WinitWindow,
    }

    impl Window {
        pub fn new(event_loop: &EventLoop<()>) -> Self {
            let window = WindowBuilder::new()
                .with_title("Saga Engine")
                .with_inner_size(LogicalSize::new(1024, 768))
                .build(&event_loop)
                .unwrap();

            Window { window }
        }
    }

    pub fn init_resources(app: &mut App, event_loop: &EventLoop<()>) -> Result<()> {
        let window = Window::new(event_loop);
        let graphics = Graphics::create(&window.window)?;

        app.world.insert_resource(window);
        app.world.insert_resource(graphics);

        log::info!("Inserted window and graphics resources");

        Ok(())
    }

    pub fn winit_event_runner(mut app: App) {
        let event_loop = EventLoop::new();

        init_resources(&mut app, &event_loop).unwrap();

        let mut is_window_active = true;
        let mut is_window_being_destroyed = false;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            if is_window_being_destroyed {
                return;
            }

            // let window = world
            //     .get_resource::<Window>()
            //     .expect("Resource missing: Window");

            match event {
                // Render a frame if our Vulkan app is not being destroyed.
                Event::MainEventsCleared if !is_window_being_destroyed => {
                    if is_window_active {
                        app.update();
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    if size.width == 0 || size.height == 0 {
                        is_window_active = false;
                    } else {
                        is_window_active = true;

                        app.world.send_event(Resize);

                        let mut graphics = app.world
                            .get_resource_mut::<Graphics>()
                            .expect("Resource missing: Graphics");
                        graphics.trigger_resize();
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    // Destroy our Vulkan app.
                    is_window_being_destroyed = true;
                    *control_flow = ControlFlow::Exit;

                    log::info!("[Saga] Window is being destroyed");

                    // Run the app and allows AppExit behaviours to run
                    app.world.send_event(AppExit);
                    app.update();

                    // Wait for GPU idle. We won't be issuing any further commands anyways
                    // And this allows for cleanup to be done next frame without
                    // interference from working processes on the GPU
                    unsafe {
                        app.world
                            .get_resource::<Graphics>()
                            .expect("Resource missing: Graphics")
                            .device_wait_idle()
                            .unwrap();
                    }

                    log::info!("[Cleanup] Running cleanup schedule");
                    app.world.run_schedule(Cleanup);

                    let mut graphics = app.world
                        .get_resource_mut::<Graphics>()
                        .expect("Resource missing: Graphics");

                    log::info!("[Cleanup] Destroying graphics");
                    graphics.destroy();
                }
                _ => {}
            }
        });
    }
}

pub fn construct_app() -> App {
    let mut app = App::new();
    app.add_plugins((saga_window::WindowPlugin, saga_renderer::Plugin))
        .add_systems(bevy_app::Startup, spawn_camera)
        .add_systems(bevy_app::Startup, spawn_mesh);

    app
}

pub fn run_app() {
    construct_app().run();
}
