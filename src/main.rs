use bevy::prelude::*;

struct WeightMarker;

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
    tire_grip: f32,
    lock_grip: f32,
    engine_force: f32,
    brake_force: f32,
    e_brake_force: f32,
    weight_transfer: f32,
    max_steer: f32,
    corner_stiffness_front: f32,
    corner_stiffness_rear: f32,
    air_resistance: f32,
    roll_resistance: f32,
}

impl Default for CarConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            mass: 1200.0,
            inertia_scale: 1.0,
            half_width: 1.0,
            centre_of_gravity_to_front: 2.0,
            centre_of_gravity_to_rear: 2.0,
            centre_of_gravity_to_front_axle: 1.25,
            centre_of_gravity_to_rear_axle: 1.25,
            centre_of_gravity_height: 0.55,
            wheel_radius: 0.3,
            wheel_width: 0.2,
            tire_grip: 1.0,
            lock_grip: 0.7,
            engine_force: 8000.0,
            brake_force: 12000.0,
            e_brake_force: 4800.0,
            weight_transfer: 0.2,
            max_steer: 0.6,
            corner_stiffness_front: 5.0,
            corner_stiffness_rear: 5.2,
            air_resistance: 2.5,
            roll_resistance: 8.0,
        }
    }
}

#[derive(Default)]
struct CarState {
    heading: f32,
    position: Vec2,
    velocity: Vec2,
    velocity_c: Vec2,
    acceleration: Vec2,
    acceleration_c: Vec2,
    absolute_velocity: f32,
    yaw_rate: f32,
    steer: f32,
    steer_angle: f32,
}

#[derive(Debug)]
struct CarStats {
    speed: f32,
    acceleration: f32,
    yaw_rate: f32,
    weight_front: f32,
    weight_rear: f32,
    slip_angle_front: f32,
    slip_angle_rear: f32,
    friction_front: f32,
    friction_rear: f32,
}

fn physics_step(
    dt_seconds: f32,
    inputs: &CarInputs,
    config: &CarConfig,
    state: &mut CarState,
) -> CarStats {
    let inertia = config.mass * config.inertia_scale;
    let wheel_base = config.centre_of_gravity_to_front_axle + config.centre_of_gravity_to_rear_axle;
    let axle_weight_ratio_front = config.centre_of_gravity_to_rear_axle / wheel_base;
    let axle_weight_ratio_rear = config.centre_of_gravity_to_front_axle / wheel_base;

    let (sn, cs) = state.heading.sin_cos();

    state.velocity_c.x = cs * state.velocity.x + sn * state.velocity.y;
    state.velocity_c.y = cs * state.velocity.y - sn * state.velocity.x;

    let axel_weight_front = config.mass
        * (axle_weight_ratio_front * config.gravity
            - config.weight_transfer * state.acceleration_c.x * config.centre_of_gravity_height
                / wheel_base);

    let axel_weight_rear = config.mass
        * (axle_weight_ratio_rear * config.gravity
            + config.weight_transfer * state.acceleration_c.x * config.centre_of_gravity_height
                / wheel_base);

    let yaw_speed_front = config.centre_of_gravity_to_front_axle * state.yaw_rate;
    let yaw_speed_rear = -config.centre_of_gravity_to_rear_axle * state.yaw_rate;

    let slip_angle_front = f32::atan2(
        state.velocity_c.y + yaw_speed_front,
        state.velocity_c.x.abs(),
    ) - state.velocity_c.x.signum() * state.steer_angle;

    let slip_angle_rear = f32::atan2(
        state.velocity_c.y + yaw_speed_rear,
        state.velocity_c.x.abs(),
    );

    let tire_grip_front = config.tire_grip;
    let tire_grip_rear = config.tire_grip * (1.0 - inputs.e_brake * (1.0 - config.lock_grip));

    let friction_force_front_cy = f32::clamp(
        -config.corner_stiffness_front * slip_angle_front,
        -tire_grip_front,
        tire_grip_front,
    ) * axel_weight_front;

    let friction_force_rear_cy = f32::clamp(
        -config.corner_stiffness_rear * slip_angle_rear,
        -tire_grip_rear,
        tire_grip_rear,
    ) * axel_weight_rear;

    let brake = f32::min(
        inputs.brake * config.brake_force + inputs.e_brake * config.e_brake_force,
        config.brake_force,
    );
    let throttle = inputs.throttle * config.engine_force;

    let traction_force_cx = throttle - brake * state.velocity_c.x.signum();
    let traction_force_cy = 0.0;

    let drag_force_cx = -config.roll_resistance * state.velocity_c.x
        - config.air_resistance * state.velocity_c.x * state.velocity_c.x.abs();
    let drag_force_cy = -config.roll_resistance * state.velocity_c.y
        - config.air_resistance * state.velocity_c.y * state.velocity_c.y.abs();

    let total_force_cx = drag_force_cx + traction_force_cx;
    let total_force_cy = drag_force_cy
        + traction_force_cy
        + state.steer_angle.cos() * friction_force_front_cy
        + friction_force_rear_cy;

    state.acceleration_c.x = total_force_cx / config.mass;
    state.acceleration_c.y = total_force_cy / config.mass;

    state.acceleration.x = cs * state.acceleration_c.x - sn * state.acceleration_c.y;
    state.acceleration.y = sn * state.acceleration_c.x + cs * state.acceleration_c.y;

    state.velocity.x += state.acceleration.x * dt_seconds;
    state.velocity.y += state.acceleration.y * dt_seconds;

    state.absolute_velocity = (state.velocity.x.powi(2) + state.velocity.y.powi(2)).sqrt();

    let mut angular_torque = (friction_force_front_cy + traction_force_cy)
        * config.centre_of_gravity_to_front_axle
        - friction_force_rear_cy * config.centre_of_gravity_to_rear_axle;

    if state.absolute_velocity.abs() < 0.5 && inputs.throttle < f32::EPSILON {
        state.velocity.x = 0.0;
        state.velocity.y = 0.0;
        state.absolute_velocity = 0.0;

        angular_torque = 0.0;
        state.yaw_rate = 0.0;
    }

    let angular_torque = angular_torque;

    let angular_acceleration = angular_torque / inertia;

    state.yaw_rate += angular_acceleration * dt_seconds;
    state.heading += state.yaw_rate * dt_seconds;

    state.position.x += state.velocity.x * dt_seconds;
    state.position.y += state.velocity.y * dt_seconds;

    CarStats {
        speed: state.velocity_c.x * 3.6 * 0.621371,
        acceleration: state.acceleration_c.x,
        yaw_rate: state.yaw_rate,
        weight_front: axel_weight_front,
        weight_rear: axel_weight_rear,
        slip_angle_front,
        slip_angle_rear,
        friction_front: friction_force_front_cy,
        friction_rear: friction_force_rear_cy,
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
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
            transform: Transform::from_scale(8.0 * Vec3::ONE),
            ..Default::default()
        })
        .insert(config)
        .insert(CarState::default())
        .with_children(|parent| {
            parent
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        size: 1.0 * Vec2::ONE,
                        ..Default::default()
                    },
                    material: materials.add(ColorMaterial::color(Color::PURPLE)),
                    transform: Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
                    ..Default::default()
                })
                .insert(WeightMarker);
        });
}

fn step(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mut cars: Query<(&CarConfig, &mut CarState, &mut Transform, &Children)>,
    mut weight_marker: Query<&mut Transform, (With<WeightMarker>, Without<CarState>)>,
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
        state.steer = input(KeyCode::Left) - input(KeyCode::Right);
        state.steer_angle = state.steer * config.max_steer;

        let stats = physics_step(time.delta_seconds(), &inputs, config, &mut state);

        transform.translation = transform.scale * state.position.extend(0.0);
        transform.rotation = Quat::from_rotation_z(state.heading);

        let weight_position = stats.weight_front / (stats.weight_front + stats.weight_rear);
        let weight_position = 2.0 * weight_position - 1.0;

        weight_marker.get_mut(children[0]).unwrap().translation.x = 10.0 * weight_position;

        text.single_mut().unwrap().sections[0].value = format!("{:#?}", stats);
    }
}

fn main() {
    App::build()
        .insert_resource(ClearColor(Color::GRAY))
        .insert_resource(WindowDescriptor {
            title: "Driving Test".to_string(),
            width: 1024.0,
            height: 768.0,
            resizable: false,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(step.system())
        .run();
}
