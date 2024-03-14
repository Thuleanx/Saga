use crate::{
    core::graphics::{
        CPUMesh, GPUMesh, Graphics, Image, ImageSampler, LoadedImage, UniformBufferSeries,
    },
    doomclone::app::saga_renderer::MeshVertexUniformObject,
};
use anyhow::Result;
use bevy_app::App;
use bevy_ecs::{component::Component, system::ResMut};
use cgmath::{vec3, Angle, Vector2};
use std::path::Path;
use vulkanalia::prelude::v1_0::*;

use self::saga_renderer::{MeshFragmentData, MeshFragmentUniformObject, MeshRenderingBundle};

type Mat4 = cgmath::Matrix4<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Quat = cgmath::Quaternion<f32>;

#[derive(Component, Clone, Copy)]
struct Position(Vec3);

#[derive(Component)]
struct RelativePosition(Vec3);

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

impl RelativePosition {
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

#[derive(Component)]
struct RelativeRotation(Quat);

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
struct Scale(Vec3);

impl RelativeRotation {
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
struct MovementSpeed(f32);

#[derive(Component)]
struct TurnSpeed(Vector2<cgmath::Rad<f64>>);

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

    fn calculate_view_matrix(position: &Position, rotation: &Rotation) -> Mat4 {
        let look: Vec3 = rotation.forward();
        let up: Vec3 = rotation.up();
        let right: Vec3 = rotation.right();
        // look.cross(up);

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

#[derive(Component)]
struct MeshRenderingInfo {
    descriptor_sets: Vec<vk::DescriptorSet>,
    vertex_uniform_buffers: UniformBufferSeries,
    fragment_uniform_buffers: UniformBufferSeries,
}

fn construct_mesh_with_cpu_mesh(
    graphics: &mut ResMut<Graphics>,
    path_to_texture: &Path,
    cpu_mesh: CPUMesh,
) -> Result<(MeshRenderingBundle, CPUMesh)> {
    let gpu_mesh = unsafe { GPUMesh::create(&graphics, &cpu_mesh).unwrap() };

    let texture = Image::load(&path_to_texture).unwrap();

    let loaded_texture = unsafe { LoadedImage::create(&graphics, &texture).unwrap() };
    let texture_sampler = unsafe { ImageSampler::create_from_graphics(&graphics).unwrap() };

    let descriptor_sets = unsafe {
        let device = graphics.get_device().clone();
        let set_layout = graphics.mesh_descriptor_set_layout;
        let swapchain_len = graphics.swapchain.get_length();
        graphics
            .mesh_descriptor_allocator
            .allocate(&device, set_layout, swapchain_len)
            .unwrap()
    };

    let vertex_uniform_buffers = unsafe {
        UniformBufferSeries::create_from_graphics::<MeshVertexUniformObject>(&graphics).unwrap()
    };

    let fragment_uniform_buffers = unsafe {
        UniformBufferSeries::create_from_graphics::<MeshFragmentUniformObject>(&graphics).unwrap()
    };

    let device = graphics.get_device().clone();
    graphics.descriptor_writer.queue_write_image(
        &device,
        &texture_sampler,
        &loaded_texture,
        &descriptor_sets,
        1,
    );

    vertex_uniform_buffers
        .get_buffers()
        .iter()
        .cloned()
        .enumerate()
        .for_each(|(index, uniform_buffer)| {
            let device = graphics.get_device().clone();
            graphics
                .descriptor_writer
                .queue_write_buffers::<MeshVertexUniformObject>(
                    &device,
                    uniform_buffer,
                    &[descriptor_sets[index]],
                    0,
                );
        });

    fragment_uniform_buffers
        .get_buffers()
        .iter()
        .cloned()
        .enumerate()
        .for_each(|(index, uniform_buffer)| {
            let device = graphics.get_device().clone();
            graphics
                .descriptor_writer
                .queue_write_buffers::<MeshFragmentUniformObject>(
                    &device,
                    uniform_buffer,
                    &[descriptor_sets[index]],
                    2,
                );
        });

    Ok((
        MeshRenderingBundle {
            mesh: Mesh { gpu_mesh },
            main_texture: MainTexture {
                texture: loaded_texture,
                sampler: texture_sampler,
            },
            fragment_data: MeshFragmentData {
                tint: cgmath::vec4(1.0, 1.0, 1.0, 1.0),
            },
            rendering_info: MeshRenderingInfo {
                vertex_uniform_buffers,
                fragment_uniform_buffers,
                descriptor_sets,
            },
        },
        cpu_mesh,
    ))
}

fn construct_mesh(
    graphics: &mut ResMut<Graphics>,
    path_to_obj: &Path,
    path_to_texture: &Path,
) -> Result<(MeshRenderingBundle, CPUMesh)> {
    let mut cpu_mesh = unsafe { CPUMesh::load_from_obj(&graphics, &path_to_obj) };

    if cpu_mesh.len() == 0 {
        return Err(anyhow::anyhow!(
            "Provided obj file at path {:?} does not have a mesh",
            path_to_obj
        ));
    }

    construct_mesh_with_cpu_mesh(graphics, path_to_texture, cpu_mesh.remove(0))
}

mod doomclone_game {
    use std::{
        collections::{HashMap, HashSet},
        marker::PhantomData,
        ops::Not,
        path::PathBuf,
        time::Duration,
    };

    use super::{
        construct_mesh, construct_mesh_with_cpu_mesh,
        saga_audio::{AudioEmitter, AudioRuntimeManager},
        saga_collision::{raycast, KnockbackEvent, Knockbackable, MeshCollider},
        saga_combat::{DamageEvent, DeathEvent, Health, IFrame},
        saga_input::{ButtonInput, MouseButtonEvent, MouseChangeEvent},
        saga_renderer::{self, CameraUniformBufferObject, MeshFragmentData},
        saga_window::Window,
        MainTexture, Mesh, MeshRenderingInfo, MovementSpeed, Position, RelativePosition,
        RelativeRotation, Rotation, Scale, TurnSpeed,
    };
    use crate::{
        core::graphics::{CPUMesh, Graphics, UniformBufferSeries},
        doomclone::app::{
            saga_collision::{self, CircleCollider, Movable, Velocity},
            saga_combat,
            saga_renderer::RebuildCommand,
            Camera, CameraRenderingInfo,
        },
    };
    use bevy_app::{App, Plugin};
    use bevy_ecs::{
        component::Component,
        entity::Entity,
        event::{Event, EventReader, EventWriter},
        query::{Changed, With, Without},
        schedule::{
            common_conditions::{in_state, on_event},
            IntoSystemConfigs, NextState, OnEnter, OnExit, State, States, SystemSet,
        },
        system::{Commands, Local, ParamSet, Query, Res, ResMut, Resource},
    };
    use bevy_time::{Time, Timer, TimerMode};
    use cgmath::{
        Array, Deg, Euler, InnerSpace, One, Quaternion, Rad, Rotation as _, Rotation3, Vector2,
        Vector3, Vector4, Zero,
    };
    use itertools::Itertools;
    use kira::sound::static_sound::StaticSoundSettings;
    use rand::Rng;
    use winit::event::{ElementState, MouseButton, VirtualKeyCode as Key};

    type Quat = cgmath::Quaternion<f32>;

    #[derive(States, Debug, Clone, PartialEq, Eq, Hash)]
    enum AppState {
        Gameplay,
        Win,
        Loss,
    }

    #[derive(States, Debug, Clone, PartialEq, Eq, Hash)]
    enum GameplayStage {
        Wave1,
        Wave2,
        Wave3,
    }

    #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
    struct GameplaySet;

    pub struct GamePlugin;

    impl Plugin for GamePlugin {
        fn build(&self, app: &mut App) {
            populate_wave_data(app);

            app.add_systems(bevy_app::Startup, (spawn_player, spawn_camera, spawn_gun))
                .add_systems(
                    OnEnter(AppState::Gameplay),
                    (
                        system_cleanup_everything,
                        spawn_map.after(system_cleanup_everything),
                        spawn_spawn_points.after(system_cleanup_everything),
                        system_recenter_player,
                        system_heal_player_to_full,
                    ),
                )
                .add_systems(bevy_app::Startup, spawn_music)
                .add_systems(
                    bevy_app::Update,
                    (
                        system_enemy_spawning
                            .in_set(GameplaySet)
                            .run_if(in_state(AppState::Gameplay)),
                        system_spawn_enemy_by_id
                            .run_if(in_state(AppState::Gameplay))
                            .run_if(on_event::<SpawnEnemy>())
                            .after(system_enemy_spawning),
                        system_flash_on_damage,
                        animate_gun_shot,
                        animate_look_at_player,
                        system_enemy_ai,
                        system_animate_camera,
                        system_gun_update,
                        system_player_shooting,
                        on_player_shot,
                        on_entity_death.run_if(on_event::<DeathEvent>()),
                        player_movement,
                        system_player_rotate_with_mouse_x,
                        system_loss_condition.run_if(in_state(AppState::Gameplay)),
                        system_restart_on_restart_ui_killed
                            .run_if(in_state(AppState::Win))
                            .run_if(on_event::<DeathEvent>()),
                        system_restart_on_restart_ui_killed
                            .run_if(in_state(AppState::Loss))
                            .run_if(on_event::<DeathEvent>()),
                    ),
                )
                .add_systems(
                    OnExit(AppState::Gameplay),
                    (
                        system_recenter_player,
                        system_cleanup_everything,
                        spawn_music.after(system_cleanup_everything),
                        spawn_restart_ui.after(system_cleanup_everything),
                    ),
                )
                .add_systems(bevy_app::PostUpdate, animate_gun)
                .add_event::<GunFire>()
                .add_event::<SpawnEnemy>()
                .add_event::<GunReload>();

            app.insert_state(AppState::Gameplay);
            app.insert_state(GameplayStage::Wave1);
        }
    }

    struct EnemyTemplate {
        radius: f32,
        damage_radius: f32,
        movement_speed: f32,
        max_health: u32,
        scale: f32,
        knockback_resistance: f32,
        path_to_texture: PathBuf,
    }

    #[derive(Resource)]
    struct AllEnemyTemplates(Vec<EnemyTemplate>);

    struct EnemyWaveData {
        enemy_id: u32,
        weight: f32,
    }

    struct WaveData {
        data: Vec<EnemyWaveData>,
        enemy_count: u32,
        enemy_cap: u32,
        spawning_interval: Duration,
    }

    #[derive(Default)]
    struct WaveSpawningData {
        number_of_spawned_enemies: u32,
        spawn_cooldown: Timer,
    }

    #[derive(Resource)]
    struct AllWaveData(HashMap<GameplayStage, WaveData>);

    fn populate_wave_data(app: &mut App) {
        let enemy_templates = AllEnemyTemplates(vec![
            EnemyTemplate {
                radius: 0.5,
                damage_radius: 1.7,
                movement_speed: 2.0,
                scale: 4.0,
                knockback_resistance: 1.0,
                path_to_texture: std::env::current_dir()
                    .unwrap()
                    .join("assets")
                    .join("png")
                    .join("fleshling.png"),
                max_health: 2,
            },
            EnemyTemplate {
                radius: 1.0,
                damage_radius: 2.2,
                movement_speed: 1.0,
                scale: 4.0,
                knockback_resistance: 1.0,
                path_to_texture: std::env::current_dir()
                    .unwrap()
                    .join("assets")
                    .join("png")
                    .join("gug.png"),
                max_health: 5,
            },
            EnemyTemplate {
                radius: 0.8,
                damage_radius: 2.0,
                movement_speed: 0.5,
                scale: 4.0,
                knockback_resistance: 1.0,
                path_to_texture: std::env::current_dir()
                    .unwrap()
                    .join("assets")
                    .join("png")
                    .join("owlcat.png"),
                max_health: 8,
            },
            EnemyTemplate {
                radius: 1.0,
                damage_radius: 2.2,
                movement_speed: 4.0,
                scale: 4.0,
                knockback_resistance: 1.0,
                path_to_texture: std::env::current_dir()
                    .unwrap()
                    .join("assets")
                    .join("png")
                    .join("butterfly.png"),
                max_health: 1,
            },
            EnemyTemplate {
                radius: 0.5,
                damage_radius: 1.7,
                movement_speed: 3.0,
                scale: 4.0,
                knockback_resistance: 1.0,
                path_to_texture: std::env::current_dir()
                    .unwrap()
                    .join("assets")
                    .join("png")
                    .join("akunohana.png"),
                max_health: 3,
            },
        ]);

        let wavedata_0 = WaveData {
            data: vec![
                EnemyWaveData {
                    enemy_id: 0,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 1,
                    weight: 0.5,
                }
            ],
            enemy_count: 20,
            enemy_cap: 5,
            spawning_interval: Duration::from_millis(5000),
        };

        let wavedata_1 = WaveData {
            data: vec![
                EnemyWaveData {
                    enemy_id: 0,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 1,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 2,
                    weight: 0.5,
                },
                EnemyWaveData {
                    enemy_id: 3,
                    weight: 0.5,
                },
            ],
            enemy_count: 25,
            enemy_cap: 10,
            spawning_interval: Duration::from_millis(2000),
        };

        let wavedata_2 = WaveData {
            data: vec![
                EnemyWaveData {
                    enemy_id: 0,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 1,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 2,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 3,
                    weight: 1.0,
                },
                EnemyWaveData {
                    enemy_id: 4,
                    weight: 1.0,
                },
            ],
            enemy_count: 30,
            enemy_cap: 15,
            spawning_interval: Duration::from_millis(1000),
        };

        let mut wave_datas = HashMap::new();
        wave_datas.insert(GameplayStage::Wave1, wavedata_0);
        wave_datas.insert(GameplayStage::Wave2, wavedata_1);
        wave_datas.insert(GameplayStage::Wave3, wavedata_2);

        app.insert_resource(enemy_templates);
        app.insert_resource(AllWaveData(wave_datas));
    }

    #[derive(Component)]
    struct Player;

    #[derive(Component)]
    struct SpawnPoint<T> {
        phantom: PhantomData<T>,
    }

    impl<T> SpawnPoint<T> {
        fn new() -> Self {
            Self {
                phantom: PhantomData,
            }
        }
    }

    #[derive(Component)]
    struct Enemy {
        damage_radius: f32,
    }

    #[derive(Component)]
    struct LookAtPlayer;

    enum FlashState {
        Inactive,
        Active { timer: Timer },
    }

    #[derive(Component)]
    struct FlashOnDamage {
        flashing: FlashState,
        duration: Duration,
    }

    impl FlashOnDamage {
        pub fn new(duration: Duration) -> Self {
            Self {
                flashing: FlashState::Inactive,
                duration,
            }
        }
    }

    enum GunState {
        ReadyToFire,
        WaitingForFire { timer: Timer },
        Reloading { timer: Timer },
    }

    #[derive(Component)]
    struct Gun {
        magazine_size: u32,
        number_of_loaded_bullets: u32,
        knockback_force: f32,
        firing_cooldown: Duration,
        reload_duration: Duration,
        gun_state: GunState,
    }

    impl Gun {
        fn new(
            magazine_size: u32,
            knockback_force: f32,
            firing_cooldown: Duration,
            reload_duration: Duration,
        ) -> Self {
            Self {
                magazine_size,
                number_of_loaded_bullets: magazine_size,
                knockback_force,
                firing_cooldown,
                reload_duration,
                gun_state: GunState::ReadyToFire,
            }
        }
    }

    #[derive(bevy_ecs::event::Event)]
    struct GunFire;

    #[derive(bevy_ecs::event::Event)]
    struct GunBlankFire;

    #[derive(bevy_ecs::event::Event)]
    struct GunReload;

    fn system_loss_condition(
        mut app_state: ResMut<NextState<AppState>>,
        player: Query<&Health, With<Player>>,
    ) {
        let player_health = player.single();

        if player_health.current_health == 0 {
            log::trace!("Player is Dead");
            app_state.set(AppState::Loss);
        }
    }

    fn system_recenter_player(mut player: Query<(&mut Position, &mut Rotation), With<Player>>) {
        if player.is_empty() {
            return;
        }
        log::trace!("Player recentered ");
        let (mut position, mut rotation) = player.single_mut();
        position.0 = Vector3::new(0.0, 2.0, 0.0);
        rotation.0 = Quaternion::one();
    }

    #[derive(Event)]
    struct SpawnEnemy(u32);

    fn system_enemy_spawning(
        mut current_spawning_data: Local<WaveSpawningData>,
        time: Res<Time>,
        wave_data: Res<AllWaveData>,
        app_state: Res<State<AppState>>,
        current_wave: Res<State<GameplayStage>>,
        mut next_wave: ResMut<NextState<GameplayStage>>,
        mut next_app_state: ResMut<NextState<AppState>>,
        mut enemy_signal_writer: EventWriter<SpawnEnemy>,
        live_enemies: Query<Entity, With<Enemy>>,
    ) {
        let game_state = current_wave.get();

        let wave_data = &wave_data.0[game_state];

        let is_finished_with_spawning =
            current_spawning_data.number_of_spawned_enemies == wave_data.enemy_count;
        let number_of_enemies_alive = live_enemies.iter().count();
        let overcrowded = number_of_enemies_alive as u32 >= wave_data.enemy_cap;

        if is_finished_with_spawning && number_of_enemies_alive == 0 {
            current_spawning_data.number_of_spawned_enemies = 0;
            current_spawning_data.spawn_cooldown = Timer::from_seconds(0.0, TimerMode::Once);
            match game_state {
                GameplayStage::Wave1 => next_wave.set(GameplayStage::Wave2),
                GameplayStage::Wave2 => next_wave.set(GameplayStage::Wave3),
                GameplayStage::Wave3 => next_app_state.set(AppState::Win),
            }
            return;
        }

        if !overcrowded {
            current_spawning_data.spawn_cooldown.tick(time.delta());
        }
        let should_spawn = current_spawning_data.spawn_cooldown.finished()
            && !is_finished_with_spawning
            && !overcrowded;
        if should_spawn {
            // Spawning
            let total_weight = wave_data
                .data
                .iter()
                .fold(0.0, |sum, enemy| sum + enemy.weight);

            if total_weight == 0.0 {
                return;
            }

            let mut chosen_value = rand::thread_rng().gen_range(0.0..total_weight);

            let enemy_id = wave_data
                .data
                .iter()
                .find_or_last(|enemy| {
                    chosen_value -= enemy.weight;
                    chosen_value <= 0.0
                })
                .unwrap()
                .enemy_id;

            enemy_signal_writer.send(SpawnEnemy(enemy_id));
            current_spawning_data.spawn_cooldown =
                Timer::new(wave_data.spawning_interval, TimerMode::Once);
            current_spawning_data.number_of_spawned_enemies += 1;
        }
    }

    fn spawn_blood_pool(
        graphics: &mut ResMut<Graphics>,
        commands: &mut Commands,
        mut location: Vector3<f32>,
    ) {
        let cpu_mesh = CPUMesh::get_simple_plane();

        let path_to_texture = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("png")
            .join("blood.png");

        let (mesh_rendering_bundle, _) =
            construct_mesh_with_cpu_mesh(graphics, &path_to_texture, cpu_mesh).unwrap();

        location.y = 0.01;
        let rotation = Quaternion::from(Euler {
            x: Deg(50.0),
            y: Deg(180.0),
            z: Deg(0.0),
        });

        commands.spawn((
            Position(location),
            Rotation(rotation),
            mesh_rendering_bundle,
        ));
    }

    fn system_spawn_enemy_by_id(
        mut enemy_spawn_commands: EventReader<SpawnEnemy>,
        mut graphics: ResMut<Graphics>,
        mut commands: Commands,
        enemy_templates: Res<AllEnemyTemplates>,
        spawn_points: Query<&Position, With<SpawnPoint<Enemy>>>,
    ) {
        let total_spawn_points = spawn_points.iter().len();

        for spawn_enemy_command in enemy_spawn_commands.read() {
            let cpu_mesh = CPUMesh::get_simple_plane();
            let spawn_point = if total_spawn_points == 0 {
                cgmath::vec3(0.0, 2.0, 0.0)
            } else {
                let index = rand::thread_rng().gen_range(0..total_spawn_points);
                spawn_points
                    .iter()
                    .nth(index)
                    .expect("This index should be within the spawn point length")
                    .0
            };

            let template = &enemy_templates.0[spawn_enemy_command.0 as usize];

            let (mesh_rendering_bundle, _) =
                construct_mesh_with_cpu_mesh(&mut graphics, &template.path_to_texture, cpu_mesh)
                    .unwrap();

            commands.spawn((
                Enemy {
                    damage_radius: template.damage_radius,
                },
                Position(spawn_point),
                Rotation(Quat::one()),
                Scale(Vector3::from_value(template.scale)),
                CircleCollider {
                    radius: template.radius,
                },
                LookAtPlayer,
                Movable,
                Velocity(Vector2::zero()),
                MovementSpeed(template.movement_speed),
                Health::new(template.max_health),
                FlashOnDamage::new(Duration::from_millis(100)),
                Knockbackable::new(template.knockback_resistance),
                mesh_rendering_bundle,
            ));
        }
    }

    fn system_heal_player_to_full(mut player: Query<&mut Health, With<Player>>) {
        for mut player_health in player.iter_mut() {
            player_health.current_health = player_health.max_health;
        }
    }

    #[derive(Component)]
    struct Music;

    // only player and gun survives any transition
    fn system_cleanup_everything(
        graphics: Res<Graphics>,
        all_entities_to_clean_up: Query<
            (
                Entity,
                Option<&Mesh>,
                Option<&MainTexture>,
                Option<&MeshRenderingInfo>,
            ),
            (Without<Player>, Without<Gun>, Without<Camera>, Without<Music>),
        >,
        mut rebuild_command_writer: EventWriter<RebuildCommand>,
        mut commands: Commands,
    ) {
        unsafe {
            graphics.device_wait_idle().unwrap();
        }
        all_entities_to_clean_up.iter().for_each(
            |(entity, mesh, main_texture, mesh_rendering_info)| {
                if let (Some(mesh), Some(main_texture), Some(mesh_rendering_info)) =
                    (mesh, main_texture, mesh_rendering_info)
                {
                    saga_renderer::remove_mesh(
                        graphics.as_ref(),
                        mesh,
                        main_texture,
                        mesh_rendering_info,
                    );
                }
                commands.entity(entity).despawn();
            },
        );
        rebuild_command_writer.send(RebuildCommand);
    }

    fn animate_gun(
        mut is_not_first_frame: Local<bool>,
        time: Res<Time>,
        mut gun: Query<(
            &mut Position,
            &mut RelativePosition,
            &mut Rotation,
            &mut RelativeRotation,
            &Gun,
        )>,
        player: Query<(&Position, &Rotation), (With<Player>, Without<Gun>)>,
    ) {
        const RESTING_POSITION: Vector3<f32> = cgmath::vec3(-0.575, -0.685, 1.23);
        const RELOAD_POSITION: Vector3<f32> = cgmath::vec3(0.0, -2.334, 1.23);

        const POSITIONAL_SMOOTHING: f32 = 0.1;
        const ROTATIONAL_SMOOTHING: f32 = 0.1;

        let resting_rotation: Quat =
            Quaternion::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0), Deg(180.0));
        let reload_rotation: Quat = Quaternion::from(Euler {
            x: Deg(50.0),
            y: Deg(180.0),
            z: Deg(0.0),
        });

        if is_not_first_frame.not() {
            *is_not_first_frame = true;

            for (_, mut relative_position, _, mut relative_rotation, _) in gun.iter_mut() {
                relative_position.0 = RESTING_POSITION;
                relative_rotation.0 = resting_rotation;
            }
        }

        for (mut position, mut relative_position, mut rotation, mut relative_rotation, gun) in
            gun.iter_mut()
        {
            let (desired_position, desired_rotation) =
                if matches!(gun.gun_state, GunState::Reloading { .. }) {
                    (RELOAD_POSITION, reload_rotation)
                } else {
                    (RESTING_POSITION, resting_rotation)
                };

            let smoothing_power = 1.0 - POSITIONAL_SMOOTHING.powf(time.delta_seconds());

            relative_position.0 =
                cgmath::VectorSpace::lerp(relative_position.0, desired_position, smoothing_power);

            relative_rotation.0 =
                cgmath::Quaternion::slerp(relative_rotation.0, desired_rotation, smoothing_power);

            for (player_position, player_rotation) in player.iter() {
                position.0 = player_rotation.0 * relative_position.0 + player_position.0;
                rotation.0 = player_rotation.0 * relative_rotation.0;
            }
        }
    }

    fn animate_gun_shot(
        mut firing_event: EventReader<GunFire>,
        mut gun: Query<(&mut RelativePosition, &mut RelativeRotation), With<Gun>>,
    ) {
        const FIRING_POSITION: Vector3<f32> = cgmath::vec3(-0.645, -0.582, 1.093);
        let firing_rotation: Quat = Quaternion::from(Euler {
            x: Deg(-19.163),
            y: Deg(181.413),
            z: Deg(-5.326),
        });

        for _ in firing_event.read() {
            for (mut relative_position, mut relative_rotation) in gun.iter_mut() {
                relative_position.0 = FIRING_POSITION;
                relative_rotation.0 = firing_rotation;
            }
        }
    }

    fn system_flash_on_damage(
        mut is_not_first_frame: Local<bool>,
        time: Res<Time>,
        mut flashing_entities: ParamSet<(
            Query<(&mut MeshFragmentData, &mut FlashOnDamage), Changed<Health>>,
            Query<(&mut MeshFragmentData, &mut FlashOnDamage)>,
        )>,
    ) {
        if *is_not_first_frame {
            flashing_entities.p0().iter_mut().for_each(
                |(mut mesh_fragment_data, mut flash_on_damage)| {
                    flash_on_damage.flashing = FlashState::Active {
                        timer: Timer::new(flash_on_damage.duration, TimerMode::Once),
                    };
                    mesh_fragment_data.tint = Vector4::new(1.0, 0.0, 0.0, 1.0);
                },
            );
        }
        *is_not_first_frame = true;

        flashing_entities.p1().iter_mut().for_each(
            |(mut mesh_fragment_data, mut flash_on_damage)| {
                let should_deactivate = match flash_on_damage.flashing {
                    FlashState::Inactive => false,
                    FlashState::Active { ref mut timer } => {
                        timer.tick(time.delta());
                        timer.finished()
                    }
                };
                if should_deactivate {
                    mesh_fragment_data.tint = Vector4::from_value(1.0);
                }
            },
        );
    }

    fn on_player_shot(
        mut player_fire_event: EventReader<GunFire>,
        mut damage_event: EventWriter<DamageEvent>,
        mut knockback_event: EventWriter<KnockbackEvent>,
        gun: Query<&Gun>,
        player: Query<(Entity, &Position, &Rotation), With<Player>>,
        movable_objects: Query<(Entity, &Position, &CircleCollider)>,
        static_objects: Query<(Entity, &MeshCollider), Without<Movable>>,
    ) {
        for player_fire in player_fire_event.read() {
            let (player_entity, player_position, player_rotation) = player.single();
            let hit = unsafe {
                raycast(
                    player_position.0.xz(),
                    player_rotation.forward().xz(),
                    &[player_entity],
                    movable_objects.iter(),
                    static_objects.iter(),
                )
            };

            let gun = gun.single();

            if let Some(hit) = hit {
                let (t, target_entity) = hit;
                let collision_point = player_position.0 + t * player_rotation.forward();

                log::trace!(
                    "Hiting entity {:?} at location {:?}",
                    target_entity,
                    collision_point
                );

                saga_combat::deal_damage(&mut damage_event, 1, target_entity, player_entity);
                saga_collision::apply_knockback(
                    &mut knockback_event,
                    target_entity,
                    gun.knockback_force * player_rotation.forward().xz(),
                );
            }
        }
    }

    fn on_entity_death(
        graphics: Res<Graphics>,
        mut death_event_reader: EventReader<DeathEvent>,
        mut rebuild_command_writer: EventWriter<RebuildCommand>,
        entities_with_mesh: Query<
            (
                Entity,
                &Position,
                Option<&Mesh>,
                Option<&MainTexture>,
                Option<&MeshRenderingInfo>,
            ),
            Without<Player>,
        >,
        mut commands: Commands,
    ) {
        log::trace!("Cleaning up dead target");
        let all_dead_targets: HashSet<Entity> = death_event_reader
            .read()
            .map(|death_event| death_event.target)
            .collect();
        unsafe {
            graphics.device_wait_idle().unwrap();
        }
        entities_with_mesh
            .iter()
            .filter(|(entity, _, _, _, _)| all_dead_targets.contains(entity))
            .for_each(|(entity, position, mesh, main_texture, mesh_rendering_info)| {
                if let (Some(mesh), Some(main_texture), Some(mesh_rendering_info)) =
                    (mesh, main_texture, mesh_rendering_info)
                {
                    saga_renderer::remove_mesh(
                        graphics.as_ref(),
                        mesh,
                        main_texture,
                        mesh_rendering_info,
                    );
                }
                commands.entity(entity).despawn();
            });
        rebuild_command_writer.send(RebuildCommand);
    }

    fn system_player_shooting(
        mut mouse_button_events: EventReader<MouseButtonEvent>,
        mut player_fire_event: EventWriter<GunFire>,
        mut player_reload_event: EventWriter<GunReload>,
        mut gun: Query<&mut Gun>,
    ) {
        let left_mouse_pressed = mouse_button_events.read().any(|button_event| {
            let MouseButtonEvent { button, state } = button_event;
            (*button, *state) == (MouseButton::Left, ElementState::Pressed)
        });

        let mut gun = gun.single_mut();
        let ready_to_fire = matches!(gun.gun_state, GunState::ReadyToFire);
        if !left_mouse_pressed || !ready_to_fire {
            return;
        }

        let has_bullet_left = gun.number_of_loaded_bullets > 0;

        if has_bullet_left {
            player_fire_event.send(GunFire);
            gun.number_of_loaded_bullets -= 1;
        } else {
            // & !has_bullet_left
            player_reload_event.send(GunReload);
            gun.number_of_loaded_bullets = gun.magazine_size;
        }

        gun.gun_state = if gun.number_of_loaded_bullets > 0 {
            GunState::WaitingForFire {
                timer: Timer::new(gun.firing_cooldown, TimerMode::Once),
            }
        } else {
            GunState::Reloading {
                timer: Timer::new(gun.reload_duration, TimerMode::Once),
            }
        }
    }

    fn system_gun_update(time: Res<Time>, mut guns: Query<&mut Gun>) {
        for mut gun in guns.iter_mut() {
            let should_be_ready = match gun.gun_state {
                GunState::WaitingForFire { ref mut timer } => {
                    timer.tick(time.delta());
                    timer.finished()
                }
                GunState::Reloading { ref mut timer } => {
                    timer.tick(time.delta());
                    timer.finished()
                }
                GunState::ReadyToFire => false,
            };

            if should_be_ready {
                gun.gun_state = GunState::ReadyToFire;
            }
        }
    }

    fn animate_look_at_player(
        mut meshes: Query<(&Position, &mut Rotation), (With<LookAtPlayer>, Without<Player>)>,
        player: Query<&Position, With<Player>>,
    ) {
        for player_position in player.iter() {
            for (position, mut rotation) in meshes.iter_mut() {
                let mut look_direction = player_position.0 - position.0;
                look_direction.y = 0.0;

                let up = cgmath::vec3(0.0, 1.0, 0.0);

                if look_direction.is_zero() {
                    continue;
                }
                look_direction = -look_direction.normalize();
                look_direction.x = -look_direction.x;

                rotation.0 = Quaternion::look_at(-look_direction, up);
            }
        }
    }

    fn system_enemy_ai(
        player: Query<(Entity, &Position, Option<&IFrame>), With<Player>>,
        mut damage_event_writer: EventWriter<DamageEvent>,
        mut knockback_event_writer: EventWriter<KnockbackEvent>,
        mut enemies: Query<(Entity, &Position, &mut Velocity, &MovementSpeed, &Enemy)>,
    ) {
        let (player_entity, player_position, iframe) = player.single();
        for (enemy_entity, position, mut velocity, movement_speed, enemy) in enemies.iter_mut() {
            let displacement_to_player = player_position.0 - position.0;
            let direction_to_player = if displacement_to_player.magnitude2() == 0.0 {
                displacement_to_player
            } else {
                displacement_to_player.normalize()
            };

            velocity.0 = (movement_speed.0 * direction_to_player).xz();

            let close_enough_to_deal_damage =
                displacement_to_player.xz().magnitude2() <= enemy.damage_radius.powi(2);
            let invulnerable = match iframe {
                Some(iframe) => iframe.is_active(),
                None => false,
            };

            let knockback_strength = 2.0;

            if !invulnerable && close_enough_to_deal_damage {
                saga_combat::deal_damage(&mut damage_event_writer, 1, player_entity, enemy_entity);
                saga_collision::apply_knockback(
                    &mut knockback_event_writer,
                    player_entity,
                    direction_to_player.xz() * knockback_strength,
                );
            }
        }
    }

    #[rustfmt::skip]
    fn player_movement(
        button_input: Res<ButtonInput>,
        mut players: Query<(&Position, &Rotation, &mut Velocity, &MovementSpeed), With<Player>>
    ) {
        let movement_w = if button_input.is_key_down(Key::W) {1} else {0};
        let movement_a = if button_input.is_key_down(Key::A) {1} else {0};
        let movement_s = if button_input.is_key_down(Key::S) {1} else {0};
        let movement_d = if button_input.is_key_down(Key::D) {1} else {0};

        let mut movement = cgmath::vec2(
            (movement_d - movement_a) as f32, (movement_w - movement_s) as f32
        );

        if !movement.is_zero() {
            movement = movement.normalize();
        }

        for (position, rotation, mut velocity, movement_speed) in players.iter_mut() {
            log::trace!("Position: {:?}", position.0);
            let mut forward = rotation.forward();
            forward.y = 0.0;
            if !forward.is_zero() {
                forward = forward.normalize();
            }

            let mut right = rotation.right();
            right.y = 0.0;
            if !right.is_zero() {
                right = right.normalize();
            }

            let movement = (forward * movement.y + right * movement.x) * movement_speed.0;

            velocity.0 = movement.xz();
        }
    }

    fn system_player_rotate_with_mouse_x(
        mut mouse_change_events: EventReader<MouseChangeEvent>,
        mut players: Query<(&Position, &mut Rotation, &TurnSpeed), With<Player>>,
    ) {
        for mouse_change_event in mouse_change_events.read() {
            for (position, mut rotation, turn_speed) in players.iter_mut() {
                let turn_amount = turn_speed.0.x * -mouse_change_event.delta.x;
                let turn_amount_f32 = Rad(turn_amount.0 as f32);
                let turn_axis = cgmath::vec3(0.0, 1.0, 0.0);

                let turn_rotation =
                    Quaternion::from_axis_angle(turn_axis, turn_amount_f32).normalize();

                rotation.0 = turn_rotation * rotation.0;
            }
        }
    }

    fn system_player_rotate_with_mouse_y(
        mut mouse_change_events: EventReader<MouseChangeEvent>,
        mut players: Query<(&Position, &mut Rotation, &TurnSpeed), With<Player>>,
    ) {
        for mouse_change_event in mouse_change_events.read() {
            for (position, mut rotation, turn_speed) in players.iter_mut() {
                let turn_amount = turn_speed.0.y * mouse_change_event.delta.y;
                let turn_amount_f32 = Rad(turn_amount.0 as f32);

                let turn_axis = cgmath::vec3(rotation.forward().z, 0.0, -rotation.forward().x);

                let turn_rotation =
                    Quaternion::from_axis_angle(turn_axis, turn_amount_f32).normalize();
                rotation.0 = turn_rotation * rotation.0;
            }
        }
    }

    fn system_animate_camera(
        mut camera: Query<(&mut Position, &mut Rotation), With<Camera>>,
        player: Query<(&Position, &Rotation), (With<Player>, Without<Camera>)>,
    ) {
        if player.is_empty() || camera.is_empty() {
            return;
        }
        let (player_position, player_rotation) = player.single();
        let (mut camera_position, mut camera_rotation) = camera.single_mut();
        camera_position.0 = player_position.0;
        camera_rotation.0 = player_rotation.0;
    }

    fn spawn_gun(mut graphics: ResMut<Graphics>, mut commands: Commands) {
        let path_to_obj = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("meshes")
            .join("gun.obj");
        let path_to_texture = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("png")
            .join("gun_texture.png");

        let (mesh_rendering_bundle, _) =
            construct_mesh(&mut graphics, &path_to_obj, &path_to_texture).unwrap();

        let position = cgmath::vec3(0.0, 0.0, 0.0);
        let rotation = Quat::one();

        let spawn = commands.spawn((
            Gun::new(
                4,
                1.0,
                Duration::from_millis(200),
                Duration::from_millis(1000),
            ),
            Position(position),
            Rotation(rotation),
            RelativePosition(position),
            RelativeRotation(rotation),
            mesh_rendering_bundle,
        ));
    }

    #[derive(Component)]
    struct RestartUI;

    fn spawn_restart_ui(
        app_state: Res<State<AppState>>,
        mut graphics: ResMut<Graphics>,
        mut commands: Commands,
    ) {
        let state = app_state.get();
        let path_to_texture = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("png")
            .join(
                match app_state.get() {
                    AppState::Gameplay => "win.png",
                    AppState::Win => "win.png",
                    AppState::Loss =>"loss.png",
                });

        let cpu_mesh = CPUMesh::get_simple_plane();

        let (mesh_rendering_bundle, _) =
            construct_mesh_with_cpu_mesh(&mut graphics, &path_to_texture, cpu_mesh).unwrap();

        let spawn = commands.spawn((
            RestartUI,
            Position(cgmath::vec3(0.0, 2.0, -4.0)),
            Rotation(Quat::one()),
            Scale((4.0 as f32) * cgmath::vec3(1.0, 1.0, 1.0)),
            CircleCollider { radius: 1.0 },
            LookAtPlayer,
            Health::new(1),
            mesh_rendering_bundle,
        ));

        spawn_map(graphics, commands);
    }

    fn system_restart_on_restart_ui_killed(
        mut app_state: ResMut<NextState<AppState>>,
        mut wave_state: ResMut<NextState<GameplayStage>>,
        mut death_event_reader: EventReader<DeathEvent>,
        query_restart_ui: Query<&RestartUI>,
    ) {
        let restart_hit = death_event_reader.read().any(|death_event| {
            if let Ok(_) = query_restart_ui.get(death_event.target) {
                true
            } else {
                false
            }
        });
        wave_state.set(GameplayStage::Wave1);
        app_state.set(AppState::Gameplay);
    }

    fn spawn_music(mut audio_manager: ResMut<AudioRuntimeManager>, mut commands: Commands) {
        let path_to_music = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("audio")
            .join("music")
            .join("ASharpMinor0.wav");

        let sound_setting = StaticSoundSettings::default().loop_region(0.0..);

        let audio_emitter =
            AudioEmitter::new(audio_manager.as_mut(), path_to_music, true, sound_setting).unwrap();

        let spawn = commands.spawn((audio_emitter, Music));
    }

    fn spawn_map(mut graphics: ResMut<Graphics>, mut commands: Commands) {
        log::info!("Spawn map");
        spawn_walls(&mut graphics, &mut commands);
        spawn_floor(&mut graphics, &mut commands);
    }

    fn spawn_floor(graphics: &mut ResMut<Graphics>, commands: &mut Commands) {
        let path_to_obj = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("meshes")
            .join("map_ground.obj");
        let path_to_texture = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("png")
            .join("floor.png");

        let (mesh_rendering_bundle, _) =
            construct_mesh(graphics, &path_to_obj, &path_to_texture).unwrap();

        let position = cgmath::vec3(0.0, 0.0, 0.0);
        let rotation = Quat::one();

        let spawn = commands.spawn((
            Position(position),
            Rotation(rotation),
            mesh_rendering_bundle,
        ));
    }

    fn spawn_walls(graphics: &mut ResMut<Graphics>, commands: &mut Commands) {
        let path_to_obj = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("meshes")
            .join("map_walls.obj");
        let path_to_texture = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join("png")
            .join("walls.png");

        let (mesh_rendering_bundle, cpu_mesh) =
            construct_mesh(graphics, &path_to_obj, &path_to_texture).unwrap();

        let mesh_collider: MeshCollider = MeshCollider::from(cpu_mesh);

        let position = cgmath::vec3(0.0, 0.0, 0.0);
        let rotation = Quat::one();

        let spawn = commands.spawn((
            Position(position),
            Rotation(rotation),
            mesh_rendering_bundle,
            mesh_collider,
        ));
    }

    fn spawn_player(mut commands: Commands) {
        let position = Position(cgmath::vec3(0.0, 2.0, -4.0));
        let rotation = Rotation(Quat::one());
        let turn_speed = TurnSpeed(cgmath::vec2(
            cgmath::Rad(1.0 / 100.0),
            cgmath::Rad(1.0 / 100.0),
        ));

        let spawn = commands.spawn((
            Player,
            MovementSpeed(8.0),
            CircleCollider { radius: 1.0 },
            Health::new(5),
            position,
            rotation,
            turn_speed,
            Movable,
            Velocity(Vector2::zero()),
            Knockbackable::new(1.0),
            IFrame::new(Duration::from_millis(1000)),
        ));
    }

    fn spawn_camera(window: Res<Window>, mut graphics: ResMut<Graphics>, mut commands: Commands) {
        log::info!("Spawn camera");
        let position = Position(cgmath::vec3(0.0, 2.0, -4.0));
        let rotation = Rotation(Quat::one());
        let size = window.window.inner_size();

        let uniform_buffers = unsafe {
            UniformBufferSeries::create_from_graphics::<CameraUniformBufferObject>(&graphics)
                .unwrap()
        };

        uniform_buffers
            .get_buffers()
            .iter()
            .cloned()
            .enumerate()
            .for_each(|(index, uniform_buffer)| {
                let descriptor_set = graphics.global_descriptor_sets[index];
                let device = graphics.get_device().clone();
                graphics
                    .descriptor_writer
                    .queue_write_buffers::<CameraUniformBufferObject>(
                        &device,
                        uniform_buffer,
                        &[descriptor_set],
                        0,
                    )
            });

        let camera = Camera {
            field_of_view: Deg(90.0).into(),
            far_plane_distance: 100.0,
            near_plane_distance: 0.1,
            width: size.width,
            height: size.height,
        };

        let view = Camera::calculate_view_matrix(&position, &rotation);
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

    fn spawn_spawn_points(mut commands: Commands) {
        let spawn_points: Vec<Vector3<f32>> =
            vec![cgmath::vec3(0.0, 2.0, 8.0), cgmath::vec3(0.0, 2.0, -8.0)];

        spawn_points.iter().for_each(|&point| {
            commands.spawn((Position(point), SpawnPoint::<Enemy>::new()));
        });
    }
}

mod saga_combat {
    use std::time::Duration;

    use bevy_app::Plugin;
    use bevy_ecs::{
        component::Component,
        entity::Entity,
        event::{Event, EventReader, EventWriter},
        query::{Changed, QueryEntityError},
        system::{ParamSet, Query, Res},
    };
    use bevy_time::{Time, Timer};

    pub struct CombatPlugin;

    impl Plugin for CombatPlugin {
        fn build(&self, app: &mut bevy_app::App) {
            app.add_event::<DamageEvent>()
                .add_event::<DeathEvent>()
                .add_systems(bevy_app::Update, system_register_damage)
                .add_systems(bevy_app::Update, system_iframe);
        }
    }

    #[derive(Component)]
    pub struct Health {
        pub current_health: u32,
        pub max_health: u32,
    }

    #[derive(Component)]
    pub struct IFrame(Timer);

    impl IFrame {
        pub fn new(duration: Duration) -> Self {
            let mut time = Timer::new(duration, bevy_time::TimerMode::Once);
            time.tick(duration);
            Self(time)
        }

        pub fn is_active(&self) -> bool {
            !self.0.finished()
        }

        pub fn activate(&mut self) {
            self.0.reset()
        }

        pub fn tick(&mut self, delta: Duration) {
            self.0.tick(delta);
        }
    }

    impl Health {
        pub fn new(max_health: u32) -> Self {
            Self {
                current_health: max_health,
                max_health,
            }
        }
    }

    #[derive(Event)]
    pub struct DamageEvent {
        pub damage: u32,
        pub target: Entity,
        pub source: Entity,
    }

    #[derive(Event)]
    pub struct DeathEvent {
        pub target: Entity,
    }

    pub fn deal_damage(
        damage_event_writer: &mut EventWriter<DamageEvent>,
        damage: u32,
        target: Entity,
        source: Entity,
    ) {
        damage_event_writer.send(DamageEvent {
            damage,
            target,
            source,
        });
    }

    fn system_register_damage(
        mut damage_events: EventReader<DamageEvent>,
        mut death_event_invoker: EventWriter<DeathEvent>,
        mut entities_with_health: Query<&mut Health>,
    ) {
        for damage_event in damage_events.read() {
            let health_query = entities_with_health.get_mut(damage_event.target);

            let mut health = match health_query {
                Ok(health) => health,
                Err(QueryEntityError::QueryDoesNotMatch(_)) => continue,
                Err(QueryEntityError::NoSuchEntity(_)) => continue,
                Err(QueryEntityError::AliasedMutability(_)) => continue,
            };

            // Decrement health
            health.current_health = if health.current_health > damage_event.damage {
                health.current_health - damage_event.damage
            } else {
                death_event_invoker.send(DeathEvent {
                    target: damage_event.target,
                });
                0
            };
        }
    }

    fn system_iframe(
        time: Res<Time>,
        mut param_set: ParamSet<(Query<&mut IFrame, Changed<Health>>, Query<&mut IFrame>)>,
    ) {
        param_set
            .p0()
            .iter_mut()
            .for_each(|mut iframe| iframe.activate());
        param_set
            .p1()
            .iter_mut()
            .for_each(|mut iframe| iframe.tick(time.delta()));
    }
}

mod saga_renderer {
    use anyhow::Result;
    use bevy_app::Plugin as BevyPlugin;
    use bevy_ecs::system::ResMut;
    use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
    use cgmath::{Matrix3, Matrix4, SquareMatrix, Vector4};
    use vulkanalia::vk;

    use crate::core::graphics::{graphics_utility, Graphics, StartRenderResult};

    use super::{saga_window::Window, Camera, CameraRenderingInfo, MainTexture, Mesh};
    use super::{MeshRenderingInfo, Position, Rotation, Scale};

    pub struct Plugin;

    impl BevyPlugin for Plugin {
        fn build(&self, app: &mut bevy_app::App) {
            app.add_event::<Resize>()
                .init_schedule(Cleanup)
                .add_event::<RebuildCommand>()
                .add_systems(
                    bevy_app::PostStartup,
                    (
                        system_build_command_buffer,
                        system_update_mesh_fragment_information,
                        system_finalize_descriptors,
                    ),
                )
                .add_systems(bevy_app::Update, system_camera_on_screen_resize)
                .add_systems(
                    bevy_app::PostUpdate,
                    system_update_mesh_fragment_information,
                )
                .add_systems(bevy_app::PostUpdate, system_update_camera_view)
                .add_systems(bevy_app::PostUpdate, system_signal_rebuild_on_mesh_added)
                .add_systems(
                    bevy_app::Last,
                    system_draw
                        .pipe(system_handle_swapchain_recreate)
                        .pipe(system_recreate_swapchain)
                        .pipe(system_log_error_result),
                )
                .add_systems(
                    bevy_app::Last,
                    system_build_command_buffer
                        .run_if(on_event::<RebuildCommand>())
                        .before(system_draw),
                )
                .add_systems(Cleanup, system_cleanup_camera)
                .add_systems(Cleanup, system_cleanup_meshes);
        }
    }

    // Events
    #[derive(bevy_ecs::event::Event)]
    pub struct Resize;

    #[derive(bevy_ecs::event::Event)]
    pub struct RebuildCommand;

    // Schedules
    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    pub struct Cleanup;

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct CameraUniformBufferObject {
        pub view: Matrix4<f32>,
        pub proj: Matrix4<f32>,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct MeshVertexUniformObject {
        pub model: Matrix4<f32>,
    }

    #[derive(Copy, Clone, Debug, Component)]
    pub struct MeshFragmentData {
        pub tint: Vector4<f32>,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct MeshFragmentUniformObject {
        pub tint: Vector4<f32>,
    }

    impl MeshFragmentUniformObject {
        pub fn new(data: &MeshFragmentData) -> Self {
            Self { tint: data.tint }
        }
    }

    #[derive(Bundle)]
    pub struct MeshRenderingBundle {
        pub mesh: Mesh,
        pub main_texture: MainTexture,
        pub fragment_data: MeshFragmentData,
        pub rendering_info: MeshRenderingInfo,
    }

    fn system_signal_rebuild_on_mesh_added(
        graphics: ResMut<Graphics>,
        mut rebuild_command: EventWriter<RebuildCommand>,
        meshes_added: Query<(), Added<Mesh>>,
    ) {
        let did_any_mesh_added = meshes_added.iter().next().is_some();
        unsafe {
            graphics.device_wait_idle().unwrap();
        }
        if did_any_mesh_added {
            log::trace!("Mesh added");
            rebuild_command.send(RebuildCommand);
            system_finalize_descriptors(graphics);
        }
    }

    fn system_camera_on_screen_resize(
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

    fn system_build_command_buffer(
        graphics: Res<Graphics>,
        meshes: Query<(&Mesh, &MainTexture, &MeshRenderingInfo)>,
    ) {
        build_command_buffer_from_graphics(&graphics, meshes).unwrap()
    }

    /// Must be called before trying to queue up destroying the mesh
    pub fn remove_mesh(
        graphics: &Graphics,
        mesh: &Mesh,
        main_texture: &MainTexture,
        rendering_info: &MeshRenderingInfo,
    ) {
        unsafe {
            rendering_info
                .vertex_uniform_buffers
                .destroy_uniform_buffer_series(&graphics);
            rendering_info
                .fragment_uniform_buffers
                .destroy_uniform_buffer_series(&graphics);
            mesh.gpu_mesh.destroy(&graphics);
            main_texture.texture.destroy_with_graphics(&graphics);
            main_texture.sampler.destroy_with_graphics(&graphics);
        }
    }

    fn build_command_buffer_from_graphics(
        graphics: &Graphics,
        meshes: Query<(&Mesh, &MainTexture, &MeshRenderingInfo)>,
    ) -> Result<()> {
        log::info!("Build command buffer");
        unsafe {
            graphics.record_command_buffers(
                |graphics: &Graphics, command_buffer: vk::CommandBuffer, index: usize| unsafe {
                    graphics.bind_descriptor_set(
                        command_buffer,
                        &[graphics.global_descriptor_sets[index]],
                        0,
                    );
                    for (mesh, main_texture, rendering_info) in &meshes {
                        graphics.bind_descriptor_set(
                            command_buffer,
                            &[rendering_info.descriptor_sets[index]],
                            1,
                        );
                        mesh.gpu_mesh.bind(graphics, command_buffer);
                        mesh.gpu_mesh.draw(graphics, command_buffer);
                    }
                },
            )
        }
    }

    fn system_finalize_descriptors(mut graphics: ResMut<Graphics>) {
        log::trace!("Write finalized descriptors");
        graphics_utility::descriptor_writer_write(graphics.as_mut());
    }

    pub fn update_mesh_fragment_information(
        graphics: &ResMut<Graphics>,
        mesh_rendering_info: &MeshRenderingInfo,
        mesh_fragment_info: &MeshFragmentData,
    ) -> Result<()> {
        let number_of_buffers_to_update = mesh_rendering_info
            .fragment_uniform_buffers
            .get_number_of_buffers();
        let mesh_fragment_uniform_object = MeshFragmentUniformObject::new(mesh_fragment_info);
        for index in 0..number_of_buffers_to_update {
            unsafe {
                graphics.update_uniform_buffer_series(
                    &mesh_rendering_info.fragment_uniform_buffers,
                    index,
                    &mesh_fragment_uniform_object,
                )?;
            }
        }
        Ok(())
    }

    fn system_update_mesh_fragment_information(
        graphics: ResMut<Graphics>,
        mesh_query: Query<(&MeshRenderingInfo, &MeshFragmentData), Changed<MeshFragmentData>>,
    ) {
        for (mesh_rendering_info, mesh_fragment_info) in mesh_query.iter() {
            update_mesh_fragment_information(&graphics, mesh_rendering_info, mesh_fragment_info)
                .unwrap();
        }
    }

    fn update_mesh_transform_information(
        graphics: &ResMut<Graphics>,
        mesh_query: Query<(&Position, &Rotation, &MeshRenderingInfo, Option<&Scale>)>,
    ) -> Result<()> {
        for (position, rotation, rendering_info, scale) in mesh_query.iter() {
            let rotation_matrix = Matrix4::from(Matrix3::from(rotation.0));
            let translation_matrix = Matrix4::from_translation(position.0);
            let scale_matrix = if let Some(scale) = scale {
                Matrix4::from_nonuniform_scale(scale.0.x, scale.0.y, scale.0.z)
            } else {
                Matrix4::identity()
            };
            let model = translation_matrix * rotation_matrix * scale_matrix;

            let ubo = MeshVertexUniformObject { model };

            let number_of_buffers_to_update =
                rendering_info.vertex_uniform_buffers.get_buffers().len();
            for index in 0..number_of_buffers_to_update {
                unsafe {
                    graphics.update_uniform_buffer_series(
                        &rendering_info.vertex_uniform_buffers,
                        index,
                        &ubo,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn system_update_camera_view(
        mut cameras: Query<(&Position, &Rotation, &mut CameraRenderingInfo)>,
    ) {
        for (position, rotation, mut rendering_info) in cameras.iter_mut() {
            rendering_info.view = Camera::calculate_view_matrix(&position, rotation);
        }
    }

    fn update_camera_transform_information(
        graphics: &ResMut<Graphics>,
        camera_query: Query<(&Camera, &CameraRenderingInfo)>,
        image_index: usize,
    ) -> Result<()> {
        for (camera, camera_rendering_info) in camera_query.iter() {
            let view = camera_rendering_info.view;
            let proj = camera_rendering_info.projection;

            let ubo = CameraUniformBufferObject { view, proj };

            unsafe {
                graphics.update_uniform_buffer_series(
                    &camera_rendering_info.uniform_buffers,
                    image_index,
                    &ubo,
                )?;
            }
        }

        Ok(())
    }

    fn system_draw(
        window: Res<Window>,
        mut graphics: ResMut<Graphics>,
        camera_query: Query<(&Camera, &CameraRenderingInfo)>,
        mesh_query: Query<(&Position, &Rotation, &MeshRenderingInfo, Option<&Scale>)>,
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

        update_mesh_transform_information(&graphics, mesh_query)?;
        update_camera_transform_information(&graphics, camera_query, image_index)?;

        unsafe {
            let should_recreate_swapchain = graphics.end_render(&window.window, image_index);
            should_recreate_swapchain
        }
    }

    fn system_handle_swapchain_recreate(In(should_recreate_swapchain): In<Result<bool>>) -> bool {
        let recreate_swapchain = match should_recreate_swapchain {
            Ok(should_recreate_swapchain) => should_recreate_swapchain,
            Err(err) => {
                log::error!("Error occur during render: {}", err);
                false
            }
        };
        recreate_swapchain
    }

    fn system_recreate_swapchain(
        In(should_recreate_swapchain): In<bool>,
        window: Res<Window>,
        mut graphics: ResMut<Graphics>,
        meshes: Query<(&Mesh, &MainTexture, &MeshRenderingInfo)>,
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

    fn system_log_error_result(In(result): In<Result<()>>) {
        if let Err(error) = result {
            log::error!("Error: {}", error)
        }
    }

    fn discard_error_result(In(result): In<Result<()>>) {}

    fn system_cleanup_meshes(
        graphics: Res<Graphics>,
        meshes: Query<(&Mesh, &MainTexture, &MeshRenderingInfo)>,
    ) {
        let graphics = graphics.as_ref();
        for (mesh, main_texture, rendering_info) in &meshes {
            unsafe {
                rendering_info
                    .vertex_uniform_buffers
                    .destroy_uniform_buffer_series(&graphics);
                rendering_info
                    .fragment_uniform_buffers
                    .destroy_uniform_buffer_series(&graphics);
                mesh.gpu_mesh.destroy(&graphics);
                main_texture.texture.destroy_with_graphics(&graphics);
                main_texture.sampler.destroy_with_graphics(&graphics);
            }
        }
        log::info!("[Saga] Cleaning up all {} meshes", meshes.iter().count());
    }

    fn system_cleanup_camera(graphics: Res<Graphics>, cameras: Query<&CameraRenderingInfo>) {
        let graphics = graphics.as_ref();
        for camera_rendering_info in &cameras {
            unsafe {
                camera_rendering_info
                    .uniform_buffers
                    .destroy_uniform_buffer_series(&graphics);
            }
        }
        log::info!("[Saga] Cleaning up all {} cameras", cameras.iter().count());
    }
}

mod saga_window {
    use super::saga_input::{self, MouseChangeEvent};
    use crate::{
        core::graphics::Graphics,
        doomclone::app::saga_renderer::{Cleanup, Resize},
    };
    use anyhow::Result;
    use bevy_app::{App, AppExit, Plugin};
    use bevy_ecs::system::Resource;
    use cgmath::{Vector2, Zero};
    use winit::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window as WinitWindow, WindowBuilder},
    };

    pub struct WindowPlugin;

    impl Plugin for WindowPlugin {
        fn build(&self, app: &mut App) {
            app.set_runner(winit_event_runner)
                .add_plugins(saga_input::InputPlugin);
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
        let mut is_first_mouse_location_input = true;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            if is_window_being_destroyed {
                return;
            }

            match event {
                Event::WindowEvent {
                    // Note this deeply nested pattern match
                    event:
                        WindowEvent::KeyboardInput {
                            input: key_event, ..
                        },
                    ..
                } => {
                    let input = app.world.get_resource_mut::<saga_input::ButtonInput>();
                    if let Some(mut input) = input {
                        let key_event = input.process_and_generate_key_event(key_event);
                        drop(input);
                        if let Some(key_event) = key_event {
                            app.world.send_event(key_event);
                        }
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::MouseInput { button, state, .. },
                    ..
                } => {
                    let input = app.world.get_resource_mut::<saga_input::ButtonInput>();
                    if let Some(mut input) = input {
                        let mouse_button_event =
                            input.process_and_generate_mouse_event(button, state);
                        drop(input);
                        app.world.send_event(mouse_button_event);
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    if let Some(mut mouse_position) =
                        app.world.get_resource_mut::<saga_input::MousePosition>()
                    {
                        let last_position = mouse_position.0;
                        let current_position = cgmath::vec2(position.x, position.y);
                        let delta = if is_first_mouse_location_input {
                            Vector2::zero()
                        } else {
                            current_position - last_position
                        };

                        mouse_position.0 = current_position;
                        drop(mouse_position);

                        app.world.send_event(MouseChangeEvent {
                            delta,
                            position: current_position,
                        });

                        is_first_mouse_location_input = false;
                    }
                }
                _ => {}
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

                        let mut graphics = app
                            .world
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

                    let mut graphics = app
                        .world
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

mod saga_input {
    pub use winit::event::VirtualKeyCode as Key;

    use anyhow::Result;
    use bevy_app::{App, Plugin};
    use cgmath::{Vector2, Zero};
    use std::collections::HashSet;
    use winit::event::{ElementState, KeyboardInput, MouseButton, ScanCode};

    #[derive(bevy_ecs::system::Resource)]
    pub struct ButtonInput {
        key_pressed: HashSet<usize>,
        mouse_button_pressed: HashSet<MouseButton>,
    }

    #[derive(bevy_ecs::system::Resource)]
    pub struct MousePosition(pub Vector2<f64>);

    #[derive(bevy_ecs::event::Event)]
    pub struct KeyboardEvent {
        pub scancode: ScanCode,
        pub state: ElementState,
        pub keycode: Key,
    }

    #[derive(bevy_ecs::event::Event)]
    pub struct MouseButtonEvent {
        pub button: MouseButton,
        pub state: ElementState,
    }

    #[derive(bevy_ecs::event::Event)]
    pub struct MouseChangeEvent {
        pub delta: Vector2<f64>,
        pub position: Vector2<f64>,
    }

    impl ButtonInput {
        pub fn process_and_generate_key_event(
            &mut self,
            key_input: KeyboardInput,
        ) -> Option<KeyboardEvent> {
            if let KeyboardInput {
                scancode,
                state,
                virtual_keycode: Some(keycode),
                ..
            } = key_input
            {
                match state {
                    winit::event::ElementState::Pressed => {
                        self.key_pressed.insert(keycode as usize);
                    }
                    winit::event::ElementState::Released => {
                        self.key_pressed.remove(&(keycode as usize));
                    }
                }

                return Some(KeyboardEvent {
                    scancode,
                    state,
                    keycode,
                });
            }

            None
        }

        pub fn process_and_generate_mouse_event(
            &mut self,
            button: MouseButton,
            state: ElementState,
        ) -> MouseButtonEvent {
            match state {
                winit::event::ElementState::Pressed => {
                    self.mouse_button_pressed.insert(button);
                }
                winit::event::ElementState::Released => {
                    self.mouse_button_pressed.remove(&button);
                }
            }

            MouseButtonEvent { button, state }
        }

        pub fn is_key_down(&self, keycode: Key) -> bool {
            self.key_pressed.contains(&(keycode as usize))
        }

        pub fn is_button_down(&self, button: MouseButton) -> bool {
            self.mouse_button_pressed.contains(&button)
        }
    }

    pub struct InputPlugin;

    impl Plugin for InputPlugin {
        fn build(&self, app: &mut App) {
            init_resources(app).unwrap();
            app.add_event::<MouseButtonEvent>()
                .add_event::<KeyboardEvent>()
                .add_event::<MouseChangeEvent>();
        }
    }

    fn init_resources(app: &mut App) -> Result<()> {
        app.world.insert_resource(ButtonInput {
            key_pressed: Default::default(),
            mouse_button_pressed: Default::default(),
        });
        app.world.insert_resource(MousePosition(Vector2::zero()));

        Ok(())
    }
}

mod saga_collision {
    use std::collections::HashMap;

    use bevy_app::{App, Plugin};
    use bevy_ecs::{
        component::Component,
        entity::Entity,
        event::{Event, EventReader, EventWriter},
        query::{With, Without},
        schedule::{common_conditions::on_event, IntoSystemConfigs},
        system::{Query, Res},
    };
    use bevy_time::Time;
    use cgmath::{InnerSpace, Vector2, Zero};
    use itertools::Itertools;

    use crate::core::graphics::CPUMesh;

    use super::{saga_utils, Position, Rotation};

    pub struct CollisionPlugin;
    impl Plugin for CollisionPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(bevy_app::Last, collision_system)
                .add_event::<KnockbackEvent>()
                .add_systems(
                    bevy_app::Update,
                    system_knockback_handler.run_if(on_event::<KnockbackEvent>()),
                );
        }
    }

    trait HasNormal {
        fn get_normal_scaled(&self, position: Vector2<f32>) -> Vector2<f32>;
    }

    #[derive(Component)]
    pub struct CircleCollider {
        pub radius: f32,
    }

    #[derive(Component)]
    pub struct MeshCollider {
        lines: Vec<LineSegment>,
    }

    static TRIANGLE_SIZE: usize = 3;
    impl From<CPUMesh> for MeshCollider {
        fn from(mesh: CPUMesh) -> Self {
            let all_line_segments: Vec<LineSegment> = mesh
                .indices
                .chunks(TRIANGLE_SIZE)
                .map(|triangle| {
                    let segments: Vec<LineSegment> = triangle
                        .iter()
                        .map(|&i| {
                            // transform to flattened points
                            mesh.vertices[i as usize].pos.xz()
                        })
                        .map(|v| (v.x.to_bits(), v.y.to_bits())) // map to bits to perform unique
                        .unique()
                        .map(|(x, y)| cgmath::vec2(f32::from_bits(x), f32::from_bits(y)))
                        .combinations(2)
                        .map(|pair| {
                            if pair[0] == pair[1] {
                                None
                            } else {
                                Some(LineSegment(pair[0], pair[1]))
                            }
                        })
                        .flatten()
                        .collect();
                    segments
                })
                .flatten()
                .collect();

            MeshCollider {
                lines: all_line_segments,
            }
        }
    }

    #[derive(Component)]
    pub struct Movable;

    #[derive(Component)]
    pub struct Velocity(pub Vector2<f32>);

    #[derive(Component)]
    pub struct Knockbackable {
        amount: Vector2<f32>,
        resistance_as_scale: f32,
    }

    #[derive(Event)]
    pub struct KnockbackEvent {
        entity: Entity,
        amount: Vector2<f32>,
    }

    impl Knockbackable {
        pub fn new(resistance_as_scale: f32) -> Self {
            Knockbackable {
                amount: Vector2::zero(),
                resistance_as_scale,
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct Circle {
        position: Vector2<f32>,
        radius: f32,
    }

    impl HasNormal for Circle {
        fn get_normal_scaled(&self, position: Vector2<f32>) -> Vector2<f32> {
            position - self.position
        }
    }

    #[derive(Clone, Copy)]
    pub struct LineSegment(Vector2<f32>, Vector2<f32>);

    impl LineSegment {
        fn len2(&self) -> f32 {
            self.direction().magnitude2()
        }

        fn len(&self) -> f32 {
            self.direction().magnitude()
        }

        fn direction(&self) -> Vector2<f32> {
            self.1 - self.0
        }

        fn is_point_on_segment(&self, point: Vector2<f32>) -> bool {
            self.is_point_on_line(point) && self.is_point_in_band(point)
        }

        fn is_point_in_band(&self, point: Vector2<f32>) -> bool {
            cgmath::dot(point - self.0, point - self.1) <= 0.0
        }

        fn is_point_on_line(&self, point: Vector2<f32>) -> bool {
            let a = point - self.0;
            let b = self.direction();
            b.magnitude2() * a == cgmath::dot(a, b) * b
        }
    }

    /// Get the part of a that is perpendicular to b.
    /// If b is 0 then return a
    fn get_perpendicular(a: Vector2<f32>, b: Vector2<f32>) -> Vector2<f32> {
        a - get_projection(a, b)
    }

    fn get_projection(a: Vector2<f32>, b: Vector2<f32>) -> Vector2<f32> {
        if b.is_zero() {
            Vector2::zero()
        } else {
            // Wow no sqrt isn't this cool
            cgmath::dot(a, b) * b / b.magnitude2()
        }
    }

    impl HasNormal for LineSegment {
        fn get_normal_scaled(&self, position: Vector2<f32>) -> Vector2<f32> {
            get_perpendicular(position - self.0, self.direction())
        }
    }

    fn sort_f32(a: &f32, b: &f32) -> std::cmp::Ordering {
        a.partial_cmp(b).expect("Found NaN while sorting")
    }

    fn order_penetration_test_result(
        a: &PenetrationTestResult,
        b: &PenetrationTestResult,
    ) -> std::cmp::Ordering {
        todo!()
    }

    enum Collision {
        WithCircle { circle: Circle },
        WithLineSegment { line_segment: LineSegment },
        WithPoint { point: Vector2<f32> },
        None,
    }

    pub fn apply_knockback(
        knockback_event_writer: &mut EventWriter<KnockbackEvent>,
        entity: Entity,
        amount: Vector2<f32>,
    ) {
        knockback_event_writer.send(KnockbackEvent { entity, amount });
    }

    fn system_knockback_handler(
        mut knockback_event_reader: EventReader<KnockbackEvent>,
        mut knockbackables: Query<(Entity, &mut Knockbackable)>,
    ) {
        let mut knockback_accumulated: HashMap<Entity, Vector2<f32>> = HashMap::new();
        for knockback_event in knockback_event_reader.read() {
            let KnockbackEvent { entity, amount } = knockback_event;
            if let Some(base_amount) = knockback_accumulated.get_mut(entity) {
                *base_amount += *amount;
            } else {
                knockback_accumulated.insert(*entity, *amount);
            }
        }

        knockbackables
            .iter_mut()
            .for_each(|(entity, mut knockbackable)| {
                if let Some(&knockback_amount) = knockback_accumulated.get(&entity) {
                    knockbackable.amount += knockback_amount;
                }
            });
    }

    fn collision_system(
        time: Res<Time>,
        movables: Query<
            (
                &mut Position,
                Option<&mut Velocity>,
                Option<&mut Knockbackable>,
                &CircleCollider,
            ),
            With<Movable>,
        >,
        static_objects: Query<(&Position, &Rotation, &MeshCollider), Without<Movable>>,
    ) {
        const MAX_TRANSLATIONS: usize = 5;
        const MINIMUM_TRANSLATION_DISTANCE: f32 = 0.0001;

        unsafe {
            for (index, (mut position, velocity, knockback, circle_collider)) in
                movables.iter_unsafe().enumerate()
            {
                let mut tracked_velocity = if let Some(ref velocity) = velocity {
                    velocity.0
                } else {
                    Vector2::zero()
                };

                let mut movement = tracked_velocity * time.delta_seconds();

                if let Some(mut knockback) = knockback {
                    movement += knockback.amount * knockback.resistance_as_scale;
                    knockback.amount = Vector2::zero();
                }

                let circle_bound = Circle {
                    position: cgmath::vec2(position.0.x, position.0.z),
                    radius: circle_collider.radius,
                };

                for _ in 0..MAX_TRANSLATIONS {
                    if movement.magnitude2()
                        < MINIMUM_TRANSLATION_DISTANCE * MINIMUM_TRANSLATION_DISTANCE
                    {
                        break;
                    }

                    let collision_result = get_first_collision(
                        index,
                        circle_bound,
                        movement,
                        &movables,
                        &static_objects,
                    );

                    let (collision_time, collision_object) = collision_result;

                    position.0 += cgmath::vec3(movement.x, 0.0, movement.y) * collision_time;
                    movement = (1.0 - collision_time) * movement;

                    let position_2d = position.0.xz();

                    let scaled_normal = match collision_object {
                        Collision::None => Vector2::zero(),
                        Collision::WithCircle { circle } => circle.get_normal_scaled(position_2d),
                        Collision::WithLineSegment { line_segment } => {
                            line_segment.get_normal_scaled(position_2d)
                        }
                        Collision::WithPoint { point } => position_2d - point,
                    };

                    tracked_velocity = get_perpendicular(tracked_velocity, scaled_normal);
                    movement = get_perpendicular(movement, scaled_normal);
                }

                if let Some(mut velocity) = velocity {
                    velocity.0 = tracked_velocity;
                }
            }
        }
    }

    fn sort_result<T>(a: &(f32, T), b: &(f32, T)) -> std::cmp::Ordering {
        sort_f32(&a.0, &b.0)
    }

    unsafe fn get_first_collision(
        movable_index: usize,
        circle_bound: Circle,
        direction: Vector2<f32>,
        movables: &Query<
            (
                &mut Position,
                Option<&mut Velocity>,
                Option<&mut Knockbackable>,
                &CircleCollider,
            ),
            With<Movable>,
        >,
        static_objects: &Query<(&Position, &Rotation, &MeshCollider), Without<Movable>>,
    ) -> (f32, Collision) {
        let collisions_with_dynamic_objects = movables.iter_unsafe().enumerate().map(
            |(index_dynamic, (position, _, _, circle_collider))| {
                if movable_index == index_dynamic {
                    return None;
                }
                let other_circle_bound = Circle {
                    position: position.0.xz(),
                    radius: circle_collider.radius,
                };

                let penetration_time =
                    penetration_time_circle_circle(circle_bound, direction, other_circle_bound);

                if let Some(penetration_time) = penetration_time {
                    Some((
                        penetration_time,
                        Collision::WithCircle {
                            circle: other_circle_bound,
                        },
                    ))
                } else {
                    None
                }
            },
        );

        let collisions_with_static_objects =
            static_objects
                .iter()
                .map(|(position, rotation, mesh_collider)| {
                    mesh_collider
                    .lines
                    .iter()
                    .map(|&segment| {
                        let circle_line_penetration = match penetration_time_circle_line(
                            circle_bound,
                            segment,
                            direction,
                            ) {
                            Some(t) => Some((
                                    t,
                                    Collision::WithLineSegment {
                                        line_segment: segment,
                                    },
                                    )),
                            None => None,
                        };

                        let circle_first_point_penetration =
                            match penetration_time_circle_point(
                                circle_bound,
                                segment.0,
                                direction,
                                ) {
                                Some(t) => {
                                    Some((t, Collision::WithPoint { point: segment.0 }))
                                }
                                None => None,
                            };

                        let circle_second_point_penetration =
                            match penetration_time_circle_point(
                                circle_bound,
                                segment.0,
                                direction,
                                ) {
                                Some(t) => {
                                    Some((t, Collision::WithPoint { point: segment.0 }))
                                }
                                None => None,
                            };

                        [
                            // We must test collision with the line and both endpoints
                            // since the line collision function ignore endpoints
                            circle_line_penetration,
                            circle_first_point_penetration,
                            circle_second_point_penetration,
                        ]
                    })
                .flatten()
                    .flatten()
                    .min_by(sort_result)
                });

        let collision_result = collisions_with_static_objects
            .chain(collisions_with_dynamic_objects)
            .flatten()
            .chain([(1.0, Collision::None)])
            .min_by(sort_result);

        collision_result.unwrap()
    }

    pub unsafe fn raycast<'a, 'b, DynamicObjectIterator, StaticObjectIterator>(
        position: Vector2<f32>,
        direction: Vector2<f32>,
        ignores: &[Entity],
        dynamic_objects: DynamicObjectIterator,
        static_objects: StaticObjectIterator,
    ) -> Option<(f32, Entity)>
    where
        DynamicObjectIterator: Iterator<Item = (Entity, &'a Position, &'a CircleCollider)>,
        StaticObjectIterator: Iterator<Item = (Entity, &'b MeshCollider)>,
    {
        let collisions_with_dynamic_objects: Vec<(f32, Entity)> = dynamic_objects
            .filter(|(entity, _, _)| !ignores.contains(entity))
            .map(|(entity, circle_position, circle_collider)| {
                if let Some(t) = penetration_time_point_circle(
                    position,
                    Circle {
                        position: circle_position.0.xz(),
                        radius: circle_collider.radius,
                    },
                    direction,
                ) {
                    Some((t, entity))
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        let collisions_with_static_objects: Vec<(f32, Entity)> = static_objects
            .filter(|(entity, _)| !ignores.contains(entity))
            .map(|(entity, mesh_collider)| {
                mesh_collider.lines.iter().map(move |&segment| {
                    if let Some(t) = penetration_time_point_line(position, segment, direction) {
                        Some((t, entity))
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .flatten()
            .collect();

        let collision_result = collisions_with_static_objects
            .iter()
            .chain(collisions_with_dynamic_objects.iter())
            .cloned()
            .min_by(sort_result);

        collision_result
    }

    pub enum PenetrationTestResult {
        WillPenetrate { earliest_time: f32 },
        AlreadyPenetrating { exit_time: f32 },
        Never,
    }

    fn penetration_time_circle_circle(
        moving_circle: Circle,
        direction: Vector2<f32>,
        stationary_circle: Circle,
    ) -> Option<f32> {
        // Have one circle donates its radius to the other
        penetration_time_circle_point(
            Circle {
                position: moving_circle.position,
                radius: moving_circle.radius + stationary_circle.radius,
            },
            stationary_circle.position,
            direction,
        )
    }

    fn penetration_time_point_line(
        point: Vector2<f32>,
        line_segment: LineSegment,
        direction: Vector2<f32>,
    ) -> Option<f32> {
        if line_segment.is_point_on_line(point) {
            return if line_segment.is_point_in_band(point) {
                Some(0.0)
            } else {
                None
            };
        }

        let normal_scaled = line_segment.get_normal_scaled(point); // should be none zero now
        let direction_projected_on_normal = get_projection(direction, normal_scaled);
        let to_point_projected_on_normal = get_projection(point - line_segment.0, normal_scaled);

        let is_line_behind_point = cgmath::dot(normal_scaled, direction) >= 0.0;
        if is_line_behind_point {
            return None;
        }

        let t = (to_point_projected_on_normal.magnitude2()
            / direction_projected_on_normal.magnitude2())
        .sqrt();
        let projected_point = point + direction * t;

        if line_segment.is_point_in_band(projected_point) {
            Some(t)
        } else {
            None
        }
    }

    fn penetration_time_circle_line(
        circle: Circle,
        line_segment: LineSegment,
        direction: Vector2<f32>,
    ) -> Option<f32> {
        if direction.is_zero() {
            return None;
        }

        let normal_direction = line_segment.get_normal_scaled(circle.position);

        // this means the circle is moving perpendicular or away from the line segment
        // we can skip computation
        let moving_away = cgmath::dot(normal_direction, direction) >= 0.0;
        if moving_away {
            return None;
        }

        // We are going to use an algorithm that works in 3D as well and try
        // our best to avoid division. But we unfortunately have to use 1 sqrt operation

        // This query is equivalent to asking if a ray trace is going to pass through a
        // cylinder of radius = radius of the circle.

        // Let P = circle.position + direction * t be the point at which the ray intersects the cylinder.
        // Notice that (P - line.0) cross (P - line.1) represents the area of a parallelogram with
        // the same area as that of a 2D cross section of the cylinder through the center.
        // So (P - line.0) cross (P - line.1) = circle.radius * len(line)
        // We can square both sides and use the quadratic formula to solve for t
        let end_position = direction + circle.position;

        // Let a = circle.position
        // Let b = circle.position + direction
        // Let c = line.0
        // Let d = line.1
        let badc = saga_utils::cross_2d(
            end_position - circle.position,
            line_segment.1 - line_segment.0,
        );
        let acdc = saga_utils::cross_2d(
            circle.position - line_segment.0,
            line_segment.1 - line_segment.0,
        );

        let quadratic_formula_a = badc * badc;
        let quadratic_formula_b = 2.0 * badc * acdc;
        let quadratic_formula_c = acdc * acdc - circle.radius * circle.radius * line_segment.len2();

        if let Ok(solutions) = saga_utils::solve_quadratic(
            quadratic_formula_a,
            quadratic_formula_b,
            quadratic_formula_c,
        ) {
            let mut candidate_t = None;
            let mut has_positive_solution = false;
            let mut has_negative_solution = false;
            let mut first_solution_valid = false;

            for (index, &t) in solutions.iter().enumerate() {
                has_negative_solution |= t < 0.0;
                has_positive_solution |= t >= 0.0;

                let p = direction * t + circle.position;
                let projection_onto_line =
                    cgmath::dot(p - line_segment.0, line_segment.direction());

                let line_intersects_cylinder_segment =
                    projection_onto_line >= 0.0 && projection_onto_line <= line_segment.len2();

                if index == 0 {
                    first_solution_valid = line_intersects_cylinder_segment;
                }

                let should_choose_this_t =
                    line_intersects_cylinder_segment && t >= 0.0 && candidate_t == None;
                if should_choose_this_t {
                    candidate_t = Some(t);
                }
            }

            let is_already_penetrating =
                has_negative_solution && has_positive_solution && first_solution_valid;
            if is_already_penetrating {
                return Some(0.0);
            }

            return candidate_t;
        }

        None
    }

    fn penetration_time_point_circle(
        point: Vector2<f32>,
        circle: Circle,
        direction: Vector2<f32>,
    ) -> Option<f32> {
        penetration_time_circle_point(circle, point, -direction)
    }

    fn penetration_time_circle_point(
        circle: Circle,
        point: Vector2<f32>,
        direction: Vector2<f32>,
    ) -> Option<f32> {
        let circle_to_point = point - circle.position;

        if cgmath::dot(-circle_to_point, direction) >= 0.0 {
            return None;
        }

        let quadratic_formula_a = cgmath::dot(direction, direction);
        let quadratic_formula_b = -2.0 * cgmath::dot(direction, circle_to_point);
        let quadratic_formula_c =
            cgmath::dot(circle_to_point, circle_to_point) - circle.radius * circle.radius;

        let quadratic_solutions = saga_utils::solve_quadratic(
            quadratic_formula_a,
            quadratic_formula_b,
            quadratic_formula_c,
        );

        if let Ok(quadratic_solutions) = quadratic_solutions {
            let already_penetrated = quadratic_solutions.len() == 2
                && quadratic_solutions[0] < 0.0
                && quadratic_solutions[1] >= 0.0;
            if already_penetrated {
                return Some(0.0);
            }

            return quadratic_solutions.iter().cloned().find(|&t| t >= 0.0);
        }

        None
    }
}

mod saga_utils {
    use anyhow::Result;
    use cgmath::{num_traits::Float, Vector2};

    pub fn cross_2d<F: Float>(a: Vector2<F>, b: Vector2<F>) -> F {
        a.x * b.y - a.y * b.x
    }

    pub fn square<F: Float>(a: F) -> F {
        a * a
    }

    pub fn solve_quadratic(a: f32, b: f32, c: f32) -> Result<Vec<f32>> {
        let infinite_solutions = a == 0.0 && b == 0.0 && c == 0.0;
        if infinite_solutions {
            return Err(anyhow::anyhow!(
                "There are infinitely many solutions to this"
            ));
        }
        let no_solution = a == 0.0 && b == 0.0; // implicit c != 0.0
        if no_solution {
            return Ok(vec![]);
        }
        let one_solution_only = a == 0.0;
        if one_solution_only {
            return Ok(vec![-c / b]);
        }

        let mut d = b * b - 4.0 * a * c;
        let no_real_solution = d < 0.0;
        if no_real_solution {
            return Ok(vec![]);
        }

        let one_solution_only = d == 0.0;
        if d == 0.0 {
            return Ok(vec![-b / (2.0 * a)]);
        }

        d = d.sqrt();

        // Quadratic formula
        let mut c1 = (-b - d) / (2.0 * a);
        let mut c2 = (-b + d) / (2.0 * a);
        if c1 > c2 {
            std::mem::swap(&mut c1, &mut c2);
        }

        Ok(vec![c1, c2])
    }
}

mod saga_audio {
    use anyhow::Result;
    use bevy_app::Plugin;
    use bevy_ecs::{
        component::Component,
        system::{Query, ResMut, Resource},
    };
    use cgmath::{One, Quaternion, Vector3, Zero};
    use kira::{
        manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings},
        sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings},
        spatial::{
            listener::{ListenerHandle, ListenerSettings},
            scene::{SpatialSceneHandle, SpatialSceneSettings},
        },
        tween::Tween,
    };
    use std::{path::Path, time::Duration};

    #[derive(Resource)]
    pub struct AudioRuntimeManager {
        audio_manager: AudioManager,
        spatial_space: SpatialSceneHandle,
    }

    pub struct AudioPlugin;
    impl Plugin for AudioPlugin {
        fn build(&self, app: &mut bevy_app::App) {
            init_resources(app);
            app.add_systems(bevy_app::PostStartup, play_sound_on_load);
        }
    }

    #[derive(Component)]
    pub struct AudioEmitter {
        sound_data: StaticSoundData,
        sound_handler: Option<StaticSoundHandle>,
        should_play_on_load: bool,
    }

    impl AudioEmitter {
        pub fn new(
            audio_runtime_manager: &AudioRuntimeManager,
            path_to_sound: impl AsRef<Path>,
            should_play_on_load: bool,
            sound_settings: StaticSoundSettings,
        ) -> Result<Self> {
            let sound_data = StaticSoundData::from_file(path_to_sound, sound_settings)?;

            Ok(AudioEmitter {
                sound_data,
                sound_handler: None,
                should_play_on_load,
            })
        }

        pub fn play(&mut self, audio_runtime_manager: &mut AudioRuntimeManager) -> Result<()> {
            if let Some(_) = self.sound_handler {
            } else {
                let event_instance = audio_runtime_manager
                    .audio_manager
                    .play(self.sound_data.clone())?;

                self.sound_handler = Some(event_instance);
            }
            Ok(())
        }

        pub fn stop(&mut self, audio_runtime_manager: &mut AudioRuntimeManager) -> Result<()> {
            if let Some(sound_handler) = &mut self.sound_handler {
                sound_handler.stop(Tween {
                    start_time: kira::StartTime::Immediate,
                    duration: Duration::ZERO,
                    easing: kira::tween::Easing::Linear,
                })?;
            }
            self.sound_handler = None;
            Ok(())
        }
    }

    #[derive(Component)]
    pub struct AudioListener {
        listener: ListenerHandle,
    }

    impl AudioListener {
        pub fn new(audio_runtime_manager: &mut AudioRuntimeManager) -> Self {
            let listener = audio_runtime_manager
                .spatial_space
                .add_listener(
                    convert_vector(Vector3::zero()),
                    convert_rotation(Quaternion::one()),
                    ListenerSettings::new(),
                )
                .unwrap();

            AudioListener { listener }
        }
    }

    fn convert_vector(vector: Vector3<f32>) -> mint::Vector3<f32> {
        mint::Vector3 {
            x: vector.x,
            y: vector.y,
            z: vector.z,
        }
    }

    fn convert_rotation(quaternion: Quaternion<f32>) -> mint::Quaternion<f32> {
        mint::Quaternion {
            v: convert_vector(quaternion.v),
            s: quaternion.s,
        }
    }

    fn init_resources(app: &mut bevy_app::App) {
        let mut audio_manager =
            AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).unwrap();

        let mut spatial_space_settings = SpatialSceneSettings::default();
        spatial_space_settings.listener_capacity = 1;
        spatial_space_settings.emitter_capacity = 128;

        let audio_space = audio_manager
            .add_spatial_scene(spatial_space_settings)
            .unwrap();

        app.insert_resource(AudioRuntimeManager {
            audio_manager,
            spatial_space: audio_space,
        });
    }

    fn play_sound_on_load(
        mut audio_manager: ResMut<AudioRuntimeManager>,
        mut all_sound_emitters: Query<&mut AudioEmitter>,
    ) {
        all_sound_emitters
            .iter_mut()
            .filter(|sound_emitter| sound_emitter.should_play_on_load)
            .for_each(|mut sound_emitter| {
                sound_emitter.play(audio_manager.as_mut()).unwrap();
            });
    }
}

pub fn construct_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        saga_window::WindowPlugin,
        saga_renderer::Plugin,
        saga_collision::CollisionPlugin,
        saga_audio::AudioPlugin,
        saga_combat::CombatPlugin,
        bevy_time::TimePlugin,
        doomclone_game::GamePlugin,
    ));

    app
}

pub fn run_app() {
    construct_app().run();
}
