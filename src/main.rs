use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    math::Mat2,
    prelude::*,
    reflect::TypeUuid,
    render::{mesh::VertexAttributeValues, pipeline::PrimitiveTopology},
    transform::TransformSystem,
    utils::BoxedFuture,
};

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

trait IntoArray: Sized {
    type A;

    fn into_array(self) -> Self::A;
}

impl IntoArray for Vec3 {
    type A = [f32; 3];

    fn into_array(self) -> Self::A {
        [self.x, self.y, self.z]
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

#[derive(Debug, serde::Deserialize, TypeUuid)]
#[uuid = "e8dbac6d-624d-466b-b38f-84737004b095"]
#[serde(default)]
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
            centre_of_gravity_to_front: 1.7,
            centre_of_gravity_to_rear: 1.7,
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
pub struct CarConfigLoader;

impl AssetLoader for CarConfigLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async move {
            let config = serde_json::from_str::<CarConfig>(std::str::from_utf8(bytes)?)?;
            load_context.set_default_asset(LoadedAsset::new(config));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["car"]
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
    fps: i32,
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
    front_left_is_skidding: bool,
    front_right_is_skidding: bool,
    rear_left_is_skidding: bool,
    rear_right_is_skidding: bool,
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

    let centre_of_gravity_to_front_axle =
        config.centre_of_gravity_to_front_axle * config.axle_distance_correction;
    let centre_of_gravity_to_rear_axle =
        config.centre_of_gravity_to_rear_axle * config.axle_distance_correction;

    let wheel_base = centre_of_gravity_to_front_axle + centre_of_gravity_to_rear_axle;
    let axle_weight_ratio_front = centre_of_gravity_to_rear_axle / wheel_base;
    let axle_weight_ratio_rear = centre_of_gravity_to_front_axle / wheel_base;

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
            * Vec2::new(centre_of_gravity_to_front_axle, config.half_width)
            + front_right_weight_offset
                * Vec2::new(centre_of_gravity_to_front_axle, -config.half_width)
            + rear_left_weight_offset
                * Vec2::new(-centre_of_gravity_to_rear_axle, config.half_width)
            + rear_right_weight_offset
                * Vec2::new(-centre_of_gravity_to_rear_axle, -config.half_width);

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

    let yaw_speed_front = centre_of_gravity_to_front_axle * state.yaw_rate;
    let yaw_speed_rear = -centre_of_gravity_to_rear_axle * state.yaw_rate;

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

    let (front_left_is_skidding, front_left_friction) = clamp(
        -config.corner_stiffness_front * slip_angle_front,
        -front_grip,
        front_grip,
    );
    let front_left_friction = front_left_friction * front_left_active_weight;
    let (front_right_is_skidding, front_right_friction) = clamp(
        -config.corner_stiffness_front * slip_angle_front,
        -front_grip,
        front_grip,
    );
    let front_right_friction = front_right_friction * front_right_active_weight;
    let front_friction = 0.5 * (front_left_friction + front_right_friction);

    let (rear_left_is_skidding, rear_left_friction) = clamp(
        -config.corner_stiffness_rear * slip_angle_rear,
        -rear_grip,
        rear_grip,
    );
    let rear_left_friction = rear_left_friction * rear_left_active_weight;
    let (rear_right_is_skidding, rear_right_friction) = clamp(
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

    let mut angular_torque = front_friction * centre_of_gravity_to_front_axle
        - rear_friction * centre_of_gravity_to_rear_axle;

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
        fps: (1.0 / dt_seconds) as i32,
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
        front_left_is_skidding,
        front_right_is_skidding,
        rear_left_is_skidding,
        rear_right_is_skidding,
        weight_position,
    }
}

struct PreviousGlobalTransform(GlobalTransform);

struct Tire {
    is_skidding: bool,
}

struct CurrentSkid {
    material: Handle<ColorMaterial>,
    mesh: Option<Handle<Mesh>>,
}

#[derive(Bundle)]
struct TireBundle {
    #[bundle]
    sprite: SpriteBundle,
    tire: Tire,
    skid: CurrentSkid,
    previous_global_transform: PreviousGlobalTransform,
}

impl TireBundle {
    fn new(material: Handle<ColorMaterial>) -> Self {
        let skid_material = material.clone();

        Self {
            sprite: SpriteBundle {
                sprite: Sprite {
                    size: Vec2::ONE,
                    ..Default::default()
                },
                material,
                ..Default::default()
            },
            tire: Tire { is_skidding: false },
            skid: CurrentSkid {
                material: skid_material,
                mesh: None,
            },
            previous_global_transform: PreviousGlobalTransform(GlobalTransform::default()),
        }
    }
}

struct Tires {
    front_left: Entity,
    front_right: Entity,
    rear_left: Entity,
    rear_right: Entity,
}

struct Bumper;

#[derive(Bundle)]
struct BumperBundle {
    #[bundle]
    sprite: SpriteBundle,
    bumper: Bumper,
}

impl BumperBundle {
    fn new(material: Handle<ColorMaterial>) -> Self {
        Self {
            sprite: SpriteBundle {
                sprite: Sprite {
                    size: Vec2::ONE,
                    ..Default::default()
                },
                material,
                ..Default::default()
            },
            bumper: Bumper,
        }
    }
}

struct Bumpers {
    front: Entity,
    rear: Entity,
}

struct CarComponents {
    tires: Tires,
    bumpers: Bumpers,
    weight_marker: Entity,
}

#[derive(Bundle)]
struct CarBundle {
    config: Handle<CarConfig>,
    components: CarComponents,
    state: CarState,
    transform: Transform,
    global_transform: GlobalTransform,
}

struct Skid;

#[derive(Bundle)]
struct SkidBundle {
    #[bundle]
    sprite: SpriteBundle,
    skid: Skid,
}

impl SkidBundle {
    fn new(mesh: Handle<Mesh>, material: Handle<ColorMaterial>) -> Self {
        Self {
            sprite: SpriteBundle {
                sprite: Sprite {
                    size: Vec2::ONE,
                    ..Default::default()
                },
                mesh,
                material,
                ..Default::default()
            },
            skid: Skid,
        }
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    asset_server.watch_for_changes().unwrap();

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

    let tire_material = materials.add(ColorMaterial::color(Color::BLACK));

    let front_left = commands
        .spawn_bundle(TireBundle::new(tire_material.clone()))
        .id();
    let front_right = commands
        .spawn_bundle(TireBundle::new(tire_material.clone()))
        .id();
    let rear_left = commands
        .spawn_bundle(TireBundle::new(tire_material.clone()))
        .id();
    let rear_right = commands.spawn_bundle(TireBundle::new(tire_material)).id();

    let tires = Tires {
        front_left,
        front_right,
        rear_left,
        rear_right,
    };

    let bumper_material = materials.add(ColorMaterial::color(Color::DARK_GRAY));

    let front_bumper = commands
        .spawn_bundle(BumperBundle::new(bumper_material.clone()))
        .id();

    let rear_bumper = commands
        .spawn_bundle(BumperBundle::new(bumper_material))
        .id();

    let bumpers = Bumpers {
        front: front_bumper,
        rear: rear_bumper,
    };

    let weight_marker = commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                size: 0.5 * Vec2::ONE,
                ..Default::default()
            },
            material: materials.add(ColorMaterial::color(Color::PURPLE)),
            ..Default::default()
        })
        .insert(WeightMarker::default())
        .id();

    commands
        .spawn_bundle(CarBundle {
            config: asset_server.load("config.car"),
            components: CarComponents {
                tires,
                bumpers,
                weight_marker,
            },
            state: CarState::default(),
            transform: Transform::default(),
            global_transform: GlobalTransform::default(),
        })
        .push_children(&[
            front_left,
            front_right,
            rear_left,
            rear_right,
            front_bumper,
            rear_bumper,
            weight_marker,
        ]);
}

fn step(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    configs: ResMut<Assets<CarConfig>>,
    mut cars: Query<(
        &Handle<CarConfig>,
        &mut CarState,
        &mut Transform,
        &CarComponents,
    )>,
    mut weight_marker: Query<&mut WeightMarker>,
    mut tires: Query<&mut Tire>,
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

    for (config, mut state, mut transform, car_components) in cars.iter_mut() {
        let config = match configs.get(config.clone()) {
            Some(config) => config,
            None => continue,
        };

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

        weight_marker
            .get_mut(car_components.weight_marker)
            .unwrap()
            .position = stats.weight_position;

        tires
            .get_mut(car_components.tires.front_left)
            .unwrap()
            .is_skidding = stats.front_left_is_skidding;
        tires
            .get_mut(car_components.tires.front_right)
            .unwrap()
            .is_skidding = stats.front_right_is_skidding;
        tires
            .get_mut(car_components.tires.rear_left)
            .unwrap()
            .is_skidding = stats.rear_left_is_skidding;
        tires
            .get_mut(car_components.tires.rear_right)
            .unwrap()
            .is_skidding = stats.rear_right_is_skidding;

        text.single_mut().unwrap().sections[0].value = format!("{:#?}", stats);
    }
}

fn place_weight_marker(mut query: Query<(&WeightMarker, &mut Transform)>) {
    for (marker, mut transform) in query.iter_mut() {
        transform.translation = marker.position.extend(1.0);
    }
}

fn place_tires(
    configs: ResMut<Assets<CarConfig>>,
    car: Query<(&Handle<CarConfig>, &CarComponents, &CarState)>,
    mut tires: Query<&mut Transform, With<Tire>>,
) {
    for (config, components, state) in car.iter() {
        let config = match configs.get(config.clone()) {
            Some(config) => config,
            None => continue,
        };

        {
            let mut tire = tires.get_mut(components.tires.front_left).unwrap();

            tire.translation = Vec3::new(
                config.centre_of_gravity_to_front_axle,
                config.half_width,
                1.0,
            );

            tire.rotation = Quat::from_rotation_z(state.steer_angle);

            tire.scale = Vec3::new(2.0 * config.wheel_radius, config.wheel_width, 1.0);
        }

        {
            let mut tire = tires.get_mut(components.tires.front_right).unwrap();

            tire.translation = Vec3::new(
                config.centre_of_gravity_to_front_axle,
                -config.half_width,
                1.0,
            );

            tire.rotation = Quat::from_rotation_z(state.steer_angle);

            tire.scale = Vec3::new(2.0 * config.wheel_radius, config.wheel_width, 1.0);
        }

        {
            let mut tire = tires.get_mut(components.tires.rear_left).unwrap();

            tire.translation = Vec3::new(
                -config.centre_of_gravity_to_rear_axle,
                config.half_width,
                1.0,
            );

            tire.scale = Vec3::new(2.0 * config.wheel_radius, config.wheel_width, 1.0);
        }

        {
            let mut tire = tires.get_mut(components.tires.rear_right).unwrap();

            tire.translation = Vec3::new(
                -config.centre_of_gravity_to_rear_axle,
                -config.half_width,
                1.0,
            );

            tire.scale = Vec3::new(2.0 * config.wheel_radius, config.wheel_width, 1.0);
        }
    }
}

fn place_bumpers(
    configs: ResMut<Assets<CarConfig>>,
    car: Query<(&Handle<CarConfig>, &CarComponents)>,
    mut bumpers: Query<&mut Transform, With<Bumper>>,
) {
    for (config, components) in car.iter() {
        let config = match configs.get(config.clone()) {
            Some(config) => config,
            None => continue,
        };

        {
            let mut bumper = bumpers.get_mut(components.bumpers.front).unwrap();
            bumper.translation = Vec3::new(config.centre_of_gravity_to_front, 0.0, 1.0);
            bumper.scale = Vec3::new(0.1, 2.0 * config.half_width, 1.0);
        }

        {
            let mut bumper = bumpers.get_mut(components.bumpers.rear).unwrap();
            bumper.translation = Vec3::new(-config.centre_of_gravity_to_rear, 0.0, 1.0);
            bumper.scale = Vec3::new(0.1, 2.0 * config.half_width, 1.0);
        }
    }
}

fn skid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tire: Query<(
        &Tire,
        &mut CurrentSkid,
        &GlobalTransform,
        &PreviousGlobalTransform,
    )>,
) {
    for (tire, mut skid, &global_transform, &PreviousGlobalTransform(previous_global_transform)) in
        tire.iter_mut()
    {
        let previous_position = previous_global_transform.translation;
        let current_position = global_transform.translation;

        let offset = current_position - previous_position;

        let sideways = 0.5
            * global_transform.scale.y
            * Vec3::new(-offset.y, offset.x, 0.0).normalize_or_zero();

        let p1 = (current_position - sideways).into_array();
        let p2 = (current_position + sideways).into_array();
        let n1 = [0.0, 0.0, 1.0];
        let n2 = n1;
        let uv1 = [0.0, 0.0];
        let uv2 = [0.0, 0.0];

        match (
            tire.is_skidding,
            skid.mesh.as_ref().and_then(|handle| meshes.get_mut(handle)),
        ) {
            (true, None) => {
                let mut mesh = Mesh::new(PrimitiveTopology::TriangleStrip);
                mesh.set_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    VertexAttributeValues::Float3(vec![p1, p2]),
                );
                mesh.set_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    VertexAttributeValues::Float3(vec![n1, n2]),
                );
                mesh.set_attribute(
                    Mesh::ATTRIBUTE_UV_0,
                    VertexAttributeValues::Float2(vec![uv1, uv2]),
                );

                let handle = meshes.add(mesh);

                skid.mesh = Some(handle.clone());

                commands.spawn_bundle(SkidBundle::new(handle, skid.material.clone()));
            }
            (true, Some(mesh)) => {
                match mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap() {
                    VertexAttributeValues::Float3(positions) => {
                        positions.push(p1);
                        positions.push(p2);
                    }
                    _ => panic!(),
                }

                match mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL).unwrap() {
                    VertexAttributeValues::Float3(positions) => {
                        positions.push(n1);
                        positions.push(n2);
                    }
                    _ => panic!(),
                }

                match mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0).unwrap() {
                    VertexAttributeValues::Float2(positions) => {
                        positions.push(uv1);
                        positions.push(uv2);
                    }
                    _ => panic!(),
                }
            }
            (false, None) => (),
            (false, Some(_mesh)) => {
                skid.mesh = None;
            }
        }
    }
}

fn cleanup_skids(
    mut commands: Commands,
    keyboard_input: Res<Input<KeyCode>>,
    mut meshes: ResMut<Assets<Mesh>>,
    skids: Query<(Entity, &Handle<Mesh>), With<Skid>>,
) {
    if keyboard_input.just_pressed(KeyCode::C) {
        for (entity, handle) in skids.iter() {
            commands.entity(entity).despawn();
            meshes.remove(handle);
        }
    }
}

fn update_previous_global_transform(
    mut query: Query<(&mut PreviousGlobalTransform, &GlobalTransform)>,
) {
    for (mut previous, &current) in query.iter_mut() {
        previous.0 = current;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SystemLabel)]
enum MyStages {
    Physics,
    UpdatePreviousGlobalTransform,
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
        .add_asset::<CarConfig>()
        .init_asset_loader::<CarConfigLoader>()
        .add_startup_system(setup.system())
        .add_system(step.system().label(MyStages::Physics))
        .add_system_set(
            SystemSet::new()
                .with_system(place_weight_marker.system())
                .with_system(place_bumpers.system())
                .with_system(place_tires.system())
                .with_system(cleanup_skids.system())
                .after(MyStages::Physics),
        )
        .add_system_set_to_stage(
            CoreStage::PostUpdate,
            SystemSet::new()
                .with_system(
                    skid.system()
                        .before(MyStages::UpdatePreviousGlobalTransform)
                        .after(TransformSystem::TransformPropagate),
                )
                .with_system(
                    update_previous_global_transform
                        .system()
                        .label(MyStages::UpdatePreviousGlobalTransform)
                        .after(TransformSystem::TransformPropagate),
                ),
        )
        .run();
}
