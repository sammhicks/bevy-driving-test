use bevy::{math::Mat2, prelude::*};

fn clamp(t: f32, min: f32, max: f32) -> (bool, f32) {
    assert!(min <= max);
    if t < min {
        (true, min)
    } else if t > max {
        (true, max)
    } else {
        (false, t)
    }
}

#[derive(Default)]
struct WeightMarker {
    position: Vec2,
}

#[derive(Default)]
struct CarInputs {
    throttle: f32,
    brake: f32,
    e_brake: f32,
}

struct CarConfig {
    gravity: f32,
    mass: f32,
    inertia_scale: f32,
    half_width: f32,
    centre_of_gravity_to_front: f32,
    centre_of_gravity_to_rear: f32,
    centre_of_gravity_to_front_axle: f32,
    centre_of_gravity_to_rear_axle: f32,
    centre_of_gravity_height: f32,
    wheel_radius: f32,
    wheel_width: f32,
    engine_force: f32,
    brake_force: f32,
    e_brake_force: f32,
    weight_transfer: f32,
    max_steer: f32,
    corner_stiffness_front: f32,
    corner_stiffness_rear: f32,
    air_resistance: f32,
    roll_resistance: f32,
    e_brake_grip_ratio_front: f32,
    total_tire_grip_front: f32,
    e_brake_grip_ratio_rear: f32,
    total_tire_grip_rear: f32,
    steer_speed: f32,
    speed_steer_correction: f32,
    speed_turning_stability: f32,
    axle_distance_correction: f32,
}

impl Default for CarConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            mass: 1500.0,
            inertia_scale: 1.0,
            half_width: 0.64,
            centre_of_gravity_to_front: 1.2,
            centre_of_gravity_to_rear: 1.2,
            centre_of_gravity_to_front_axle: 1.0,
            centre_of_gravity_to_rear_axle: 1.0,
            centre_of_gravity_height: 0.55,
            wheel_radius: 0.5,
            wheel_width: 0.2,
            engine_force: 8000.0,
            brake_force: 12000.0,
            e_brake_force: 4800.0,
            weight_transfer: 0.2,
            max_steer: 0.6,
            corner_stiffness_front: 5.0,
            corner_stiffness_rear: 5.2,
            air_resistance: 2.5,
            roll_resistance: 8.0,
            e_brake_grip_ratio_front: 0.9,
            total_tire_grip_front: 2.5,
            e_brake_grip_ratio_rear: 0.4,
            total_tire_grip_rear: 2.5,
            steer_speed: 2.5,
            speed_steer_correction: 60.0,
            speed_turning_stability: 11.8,
            axle_distance_correction: 1.7,
        }
    }
}

#[derive(Default)]
struct CarState {
    heading: f32,
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
    local_acceleration: Vec2,
    yaw_rate: f32,
    steer: f32,
    steer_angle: f32,
}

#[derive(Debug)]
struct CarStats {
    speed_mps: f32,
    speed_kph: f32,
    speed_mph: f32,
    steering: f32,
    steer_angle: f32,
    front_left_active_weight: f32,
    front_right_active_weight: f32,
    rear_left_active_weight: f32,
    rear_right_active_weight: f32,
    front_left_friction: f32,
    front_right_friction: f32,
    rear_left_friction: f32,
    rear_right_friction: f32,
    front_left_slipping: bool,
    front_right_slipping: bool,
    rear_left_slipping: bool,
    rear_right_slipping: bool,
    weight_position: Vec2,
}

fn physics_step(
    dt_seconds: f32,
    inputs: &CarInputs,
    config: &CarConfig,
    state: &mut CarState,
) -> CarStats {
    let inertia = config.mass * config.inertia_scale;
    let track_width = config.half_width * 2.0;
    let wheel_base = config.centre_of_gravity_to_front_axle + config.centre_of_gravity_to_rear_axle;
    let axle_weight_ratio_front = config.centre_of_gravity_to_rear_axle / wheel_base;
    let axle_weight_ratio_rear = config.centre_of_gravity_to_front_axle / wheel_base;

    let local_velocity = Mat2::from_angle(-state.heading) * state.velocity;

    let transfer_x =
        config.weight_transfer * config.centre_of_gravity_height * state.local_acceleration.x
            / wheel_base;
    let transfer_y =
        config.weight_transfer * state.local_acceleration.y * config.centre_of_gravity_height
            / track_width
            * 20.0;

    let weight_front = config.mass * (axle_weight_ratio_front * config.gravity - transfer_x);
    let weight_rear = config.mass * (axle_weight_ratio_rear * config.gravity + transfer_x);

    let front_left_active_weight = weight_front - transfer_y;
    let front_right_active_weight = weight_front + transfer_y;
    let rear_left_active_weight = weight_rear - transfer_y;
    let rear_right_active_weight = weight_rear + transfer_y;

    let weight_position = {
        let front_left_weight_offset = front_left_active_weight;
        let front_right_weight_offset = front_right_active_weight;
        let rear_left_weight_offset = rear_left_active_weight;
        let rear_right_weight_offset = rear_right_active_weight;

        let position = front_left_weight_offset
            * Vec2::new(config.centre_of_gravity_to_front_axle, config.half_width)
            + front_right_weight_offset
                * Vec2::new(config.centre_of_gravity_to_front_axle, -config.half_width)
            + rear_left_weight_offset
                * Vec2::new(-config.centre_of_gravity_to_rear_axle, config.half_width)
            + rear_right_weight_offset
                * Vec2::new(-config.centre_of_gravity_to_rear_axle, -config.half_width);

        let total_weight = front_left_weight_offset
            + front_right_weight_offset
            + rear_left_weight_offset
            + rear_right_weight_offset;

        if total_weight > f32::EPSILON {
            position / total_weight
        } else {
            Vec2::ZERO
        }
    };

    let yaw_speed_front = config.centre_of_gravity_to_front_axle * state.yaw_rate;
    let yaw_speed_rear = -config.centre_of_gravity_to_rear_axle * state.yaw_rate;

    let slip_angle_front = f32::atan2(local_velocity.y + yaw_speed_front, local_velocity.x.abs())
        - local_velocity.x.signum() * state.steer_angle;

    let slip_angle_rear = f32::atan2(local_velocity.y + yaw_speed_rear, local_velocity.x.abs());

    let brake = f32::min(
        inputs.brake * config.brake_force + inputs.e_brake * config.e_brake_force,
        config.brake_force,
    );
    let throttle = inputs.throttle * config.engine_force;

    let rear_torque = throttle / config.wheel_radius;

    let front_grip = config.total_tire_grip_front
        * (1.0 - inputs.e_brake * (1.0 - config.e_brake_grip_ratio_front));
    let rear_grip = config.total_tire_grip_rear
        * (1.0 - inputs.e_brake * (1.0 - config.e_brake_grip_ratio_rear));

    let (front_left_slipping, front_left_friction) = clamp(
        -config.corner_stiffness_front * slip_angle_front,
        -front_grip,
        front_grip,
    );
    let front_left_friction = front_left_friction * front_left_active_weight;
    let (front_right_slipping, front_right_friction) = clamp(
        -config.corner_stiffness_front * slip_angle_front,
        -front_grip,
        front_grip,
    );
    let front_right_friction = front_right_friction * front_right_active_weight;
    let front_friction = 0.5 * (front_left_friction + front_right_friction);

    let (rear_left_slipping, rear_left_friction) = clamp(
        -config.corner_stiffness_rear * slip_angle_rear,
        -rear_grip,
        rear_grip,
    );
    let rear_left_friction = rear_left_friction * rear_left_active_weight;
    let (rear_right_slipping, rear_right_friction) = clamp(
        -config.corner_stiffness_rear * slip_angle_rear,
        -rear_grip,
        rear_grip,
    );
    let rear_right_friction = rear_right_friction * rear_right_active_weight;
    let rear_friction = 0.5 * (rear_left_friction + rear_right_friction);

    let traction_force_x = rear_torque - brake * local_velocity.x.signum();
    let traction_force_y = 0.0;

    let drag_force = -config.roll_resistance * local_velocity
        - config.air_resistance * local_velocity * local_velocity.abs();

    let total_force_x = traction_force_x + drag_force.x;
    let mut total_force_y =
        traction_force_y + drag_force.y + state.steer_angle.cos() * front_friction + rear_friction;

    if state.velocity.length() > 10.0 {
        total_force_y *= (state.velocity.length() + 1.0) / (21.0 - config.speed_turning_stability);
    }

    let total_force_y = total_force_y;

    state.local_acceleration.x = total_force_x / config.mass;
    state.local_acceleration.y = total_force_y / config.mass;

    state.acceleration = Mat2::from_angle(state.heading) * state.local_acceleration;

    state.velocity += state.acceleration * dt_seconds;

    let mut absolute_velocity = state.velocity.length();

    let mut angular_torque = front_friction * config.centre_of_gravity_to_front_axle
        - rear_friction * config.centre_of_gravity_to_rear_axle;

    if absolute_velocity < 0.5 && throttle < f32::EPSILON {
        state.local_acceleration = Vec2::ZERO;
        absolute_velocity = 0.0;
        state.velocity = Vec2::ZERO;
        angular_torque = 0.0;
        state.yaw_rate = 0.0;
        state.acceleration = Vec2::ZERO;
    }

    let absolute_velocity = absolute_velocity;
    let angular_torque = angular_torque;

    let speed_kph = absolute_velocity * 3.6;
    let speed_mph = speed_kph * 0.621371;

    let angular_acceleration = angular_torque / inertia;

    state.yaw_rate += angular_acceleration * dt_seconds;

    if ((absolute_velocity < 1.0 || state.local_acceleration.y.abs() < 2.5)
        && state.steer_angle.abs() < f32::EPSILON)
        || speed_kph < 0.2
    {
        state.yaw_rate = 0.0;
    }

    state.heading += state.yaw_rate * dt_seconds;
    state.position += state.velocity * dt_seconds;

    CarStats {
        speed_mps: absolute_velocity,
        speed_kph,
        speed_mph,
        steering: state.steer,
        steer_angle: state.steer_angle,
        front_left_active_weight,
        front_right_active_weight,
        rear_left_active_weight,
        rear_right_active_weight,
        front_left_friction,
        front_right_friction,
        rear_left_friction,
        rear_right_friction,
        front_left_slipping,
        front_right_slipping,
        rear_left_slipping,
        rear_right_slipping,
        weight_position,
    }
}

#[derive(Bundle)]
struct TireBundle {
    #[bundle]
    sprite: SpriteBundle,
}

impl TireBundle {
    fn new(position: Vec2, material: Handle<ColorMaterial>, config: &CarConfig) -> Self {
        Self {
            sprite: SpriteBundle {
                sprite: Sprite {
                    size: Vec2::new(2.0 * config.wheel_radius, config.wheel_width),
                    ..Default::default()
                },
                material,
                transform: Transform::from_translation(position.extend(1.0)),
                ..Default::default()
            },
        }
    }

    fn new_front(y: f32, material: Handle<ColorMaterial>, config: &CarConfig) -> Self {
        Self::new(
            Vec2::new(
                config.centre_of_gravity_to_front_axle,
                y * config.half_width,
            ),
            material,
            config,
        )
    }
    fn new_rear(y: f32, material: Handle<ColorMaterial>, config: &CarConfig) -> Self {
        Self::new(
            Vec2::new(
                -config.centre_of_gravity_to_rear_axle,
                y * config.half_width,
            ),
            material,
            config,
        )
    }
}

#[derive(Default)]
struct FrontTire {
    steer_angle: f32,
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn_bundle({
        let mut camera = OrthographicCameraBundle::new_2d();

        camera.orthographic_projection.scale = 1.0 / 16.0;

        camera
    });
    commands.spawn_bundle(UiCameraBundle::default());

    commands.spawn_bundle(TextBundle {
        style: Style {
            position_type: PositionType::Absolute,
            position: Rect {
                top: Val::Px(5.0),
                left: Val::Px(15.0),
                ..Default::default()
            },
            ..Default::default()
        },
        text: Text::with_section(
            "Debug Info",
            TextStyle {
                font: asset_server.load("fonts/fira_sans/FiraSans-Regular.ttf"),
                font_size: 16.0,
                color: Color::WHITE,
            },
            TextAlignment {
                horizontal: HorizontalAlign::Left,
                ..Default::default()
            },
        ),
        ..Default::default()
    });

    let config = CarConfig::default();

    let tire_material = materials.add(ColorMaterial::color(Color::BLACK));

    let front_left = commands
        .spawn_bundle(TireBundle::new_front(1.0, tire_material.clone(), &config))
        .insert(FrontTire::default())
        .id();
    let front_right = commands
        .spawn_bundle(TireBundle::new_front(-1.0, tire_material.clone(), &config))
        .insert(FrontTire::default())
        .id();
    let rear_left = commands
        .spawn_bundle(TireBundle::new_rear(1.0, tire_material.clone(), &config))
        .id();
    let rear_right = commands
        .spawn_bundle(TireBundle::new_rear(-1.0, tire_material, &config))
        .id();

    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                size: Vec2::new(
                    config.centre_of_gravity_to_front + config.centre_of_gravity_to_rear,
                    2.0 * config.half_width,
                ),
                ..Default::default()
            },
            material: materials.add(ColorMaterial::color(Color::ORANGE_RED)),
            ..Default::default()
        })
        .insert(config)
        .insert(CarState::default())
        .with_children(|parent| {
            parent
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        size: 0.5 * Vec2::ONE,
                        ..Default::default()
                    },
                    material: materials.add(ColorMaterial::color(Color::PURPLE)),
                    transform: Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
                    ..Default::default()
                })
                .insert(WeightMarker::default());
        })
        .push_children(&[front_left, front_right, rear_left, rear_right]);
}

fn step(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mut cars: Query<(&CarConfig, &mut CarState, &mut Transform, &Children)>,
    mut weight_marker: Query<&mut WeightMarker>,
    mut front_tires: Query<&mut FrontTire>,
    mut text: Query<&mut Text, Without<CarState>>,
) {
    let input = |code: KeyCode| {
        if keyboard_input.pressed(code) {
            1.0
        } else {
            0.0
        }
    };

    let inputs = CarInputs {
        throttle: input(KeyCode::Up),
        brake: input(KeyCode::Down),
        e_brake: input(KeyCode::Space),
    };

    for (config, mut state, mut transform, children) in cars.iter_mut() {
        let input_steer = input(KeyCode::Left) - input(KeyCode::Right);
        let target_steer = input_steer
            * (1.0 - (state.velocity.length() / config.speed_steer_correction).min(1.0));

        let max_steer_offset = config.steer_speed * time.delta_seconds();

        if target_steer > (state.steer + max_steer_offset) {
            state.steer += max_steer_offset;
        } else if target_steer < (state.steer - max_steer_offset) {
            state.steer -= max_steer_offset;
        } else {
            state.steer = target_steer;
        }

        state.steer_angle = config.max_steer * state.steer;

        let stats = physics_step(time.delta_seconds(), &inputs, config, &mut state);

        if keyboard_input.pressed(KeyCode::R) {
            state.position = Vec2::ZERO;
        }

        transform.translation = state.position.extend(1.0);
        transform.rotation = Quat::from_rotation_z(state.heading);

        for &entity in children.iter() {
            if let Ok(mut weight_marker) = weight_marker.get_mut(entity) {
                weight_marker.position = stats.weight_position;
            }

            if let Ok(mut front_tire) = front_tires.get_mut(entity) {
                front_tire.steer_angle = state.steer_angle;
            }
        }

        text.single_mut().unwrap().sections[0].value = format!("{:#?}", stats);
    }
}

fn place_weight_marker(mut query: Query<(&WeightMarker, &mut Transform)>) {
    for (marker, mut transform) in query.iter_mut() {
        transform.translation = marker.position.extend(1.0);
    }
}

fn animate_wheels(mut front_tires: Query<(&FrontTire, &mut Transform)>) {
    for (tire, mut transform) in front_tires.iter_mut() {
        transform.rotation = Quat::from_rotation_z(tire.steer_angle);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SystemLabel)]
enum MyStages {
    Physics,
}

fn main() {
    App::build()
        .insert_resource(ClearColor(Color::GRAY))
        .insert_resource(WindowDescriptor {
            title: "Driving Test".to_string(),
            width: 1600.0,
            height: 900.0,
            resizable: true,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(step.system().label(MyStages::Physics))
        .add_system_set(
            SystemSet::new()
                .with_system(place_weight_marker.system())
                .with_system(animate_wheels.system())
                .after(MyStages::Physics),
        )
        .run();
}
