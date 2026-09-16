use cgmath::{InnerSpace, Point3, perspective, Deg};

use crate::math::*;


/// Define camera movement directions
/// 
/// ## Variants
/// 
/// - `Forward`
/// - `Backward`
/// - `Left`
/// - `Right`
/// - `Up`
/// - `Down`
#[derive(Debug, Clone, Copy)]
pub enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down
}

/// Versatile camera that can be configurated for different use cases:
/// * **First-Person Camera**: Simulates viewing the world through the eyes of a character.
/// * **Third-Person Camera**: Follows a character from behind or another fixed relative position.
/// * **Orbit Camera**: Rotates around a fixed point, useful for object inspection.
/// * **Free Camera**: Allows unrestricted movement in all directions.
/// 
/// ## Fields
/// 
/// - `position` ([`Vec3`]) - Camera's location in world coordinates.
/// - `front` ([`Vec3`]) - Forward direction (where camera is looking).
/// - `up` ([`Vec3`]) - Camera's local up direction (for roll control).
/// - `right` ([`Vec3`]) - Camera's local right direction (perpendicular to front and up).
/// - `world_up` ([`Vec3`]) - Global up vector reference (typically Y-axis).
/// - `yaw` (`f32`) - Horizontal rotation in **degree** around the world up-axis (left-right looking).
/// - `pitch` (`f32`) - Vertical rotation in **degree** around the camera's right axis (up-down looking).
/// - `movement_speed` (`f32`) - Units per second for translation movement.
/// - `mouse_sensitivity` (`f32`) - Multiplier for mouse input to rotation angle conversion.
/// - `zoom` (`f32`) - Field of view control for perspective projection.
#[derive(Debug)]
pub struct Camera {
    // Spatial-positioning and orientation
    position: Vec3,
    front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,

    // Rotation representation using Euler angles (store in degrees).
    // Provides intuitive control while managing gimbal lock and other rotation complexities.
    yaw: f32,
    pitch: f32,

    // User interaction and behavior parameters
    // These control how the camera responds to input and environmental factors
    movement_speed: f32,
    mouse_sensitivity: f32,
    zoom: f32,
}

impl Camera {
    pub fn new(
        position: Vec3,
        up: Vec3,
        yaw: f32,
        pitch: f32,
        movement_speed: f32,
        mouse_sensitivity: f32,
        zoom: f32,
    ) -> Self {
        let world_up = up;

        let front = Vec3::new(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos()
        ).normalize();
        let right = front.cross(world_up).normalize();
        let up = right.cross(front).normalize();

        Self {
            position,
            front,
            up,
            right,
            world_up,
            yaw,
            pitch,
            movement_speed,
            mouse_sensitivity,
            zoom
        }
    }

    /// Internal coordinate system maintenance.
    /// Ensures mathematical consistency when orientation changes occurs.
    fn update_camera_vectors(&mut self) {
        // Calculate the new front vector
        let new_front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos()
        );
        self.front = new_front.normalize();

        // Re-calculate the right and up vectors
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let center = Point3::new(self.position.x + self.front.x, self.position.y + self.front.y, self.position.z + self.front.z);

        Mat4::look_at_rh(
            Point3::new(self.position.x, self.position.y, self.position.z),
            center,
            self.up
        )
    }

    pub fn get_projection_matrix(
        &self,
        aspect_ratio: f32,
        near_plane: Option<f32>,
        far_plane: Option<f32>
    ) -> Mat4 {
        let near_plane = near_plane.unwrap_or(0.1);
        let far_plane = far_plane.unwrap_or(100.0);

        perspective(
            Deg(self.zoom),
            aspect_ratio,
            near_plane,
            far_plane
        )

    }

    pub fn process_keyboard(
        &mut self,
        dir: CameraMovement,
        delta_time: f32,
    ) {
        let velocity = self.movement_speed * delta_time;

        match dir {
            CameraMovement::Forward => self.position += self.front * velocity,
            CameraMovement::Backward => self.position -= self.front * velocity,
            CameraMovement::Left => self.position -= self.right * velocity,
            CameraMovement::Right => self.position += self.right * velocity,
            CameraMovement::Up => self.position += self.up * velocity,
            CameraMovement::Down => self.position -= self.up * velocity,
        }
    }

    pub fn process_mouse_movement(
        &mut self,
        x_offset: f32,
        y_offset: f32,
        constrain_pitch: Option<(f32, f32)>,
    ) {
        let x_offset = x_offset * self.mouse_sensitivity;
        let y_offset = y_offset * self.mouse_sensitivity;

        self.yaw += x_offset;
        self.pitch += y_offset;

        if let Some((constrain_min, constrain_max)) = constrain_pitch {
            self.pitch = self.pitch.clamp(constrain_min, constrain_max);
        }

        self.update_camera_vectors();
    }

    pub fn process_mouse_scroll(&mut self, y_offset: f32) {
		self.zoom -= y_offset;
		self.zoom = self.zoom.clamp(1.0, 100.0);
    }

    pub fn get_position(&self) -> Vec3 { self.position }
    pub fn get_front(&self) -> Vec3 { self.front }
    pub fn get_right(&self) -> Vec3 { self.right }
    pub fn get_up(&self) -> Vec3 { self.up }
    pub fn get_zoom(&self) -> f32 { self.zoom }

    // command control
    pub fn set_movement_speed(&mut self, movement_speed: f32) { self.movement_speed = movement_speed; }
    pub fn set_sensitivity(&mut self, mouse_sensitivity: f32) { self.mouse_sensitivity = mouse_sensitivity; }

    pub fn goto(&mut self, new_position: Vec3) {
        self.position = new_position;
    }

    pub fn look_at(&mut self, target: Vec3, constrain_pitch: Option<(f32, f32)>) {
        let diff = target - self.position;
        if diff.magnitude2() < 1e-6 {
            return;
        }
        let dir = diff.normalize();
        

        self.yaw = dir.z.atan2(dir.x).to_degrees();
        self.pitch = dir.y.asin().to_degrees();
        if let Some((constrain_min, constrain_max)) = constrain_pitch {
            self.pitch = self.pitch.clamp(constrain_min, constrain_max);
        }

        self.update_camera_vectors();
    }

    // utils
    pub fn builder() -> CameraBuilder { CameraBuilder::new() }
}


pub struct CameraBuilder {
    // Spatial-positioning and orientation
    position: Vec3,
    world_up: Vec3,

    // Rotation representation using Euler angles (store in degrees).
    // Provides intuitive control while managing gimbal lock and other rotation complexities.
    yaw: f32,
    pitch: f32,

    // User interaction and behavior parameters
    // These control how the camera responds to input and environmental factors
    movement_speed: f32,
    mouse_sensitivity: f32,
    zoom: f32,
}

impl Default for CameraBuilder {
    fn default() -> Self {
        Self {
            position:				Vec3::new(0.0, 0.0, 0.0),
            world_up:				Vec3::new(0.0, 1.0, 0.0),
            yaw:					-90.0,
            pitch:					0.0,
            movement_speed:			2.5,
            mouse_sensitivity:		0.1,
            zoom:					45.0,
        }
    }
}

impl CameraBuilder {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn position(mut self, position: Vec3) -> Self {
		self.position = position;
		self
	}

	pub fn world_up(mut self, world_up: Vec3) -> Self {
		self.world_up = world_up;
		self
	}

	pub fn yaw(mut self, yaw: f32) -> Self {
		self.yaw = yaw;
		self
	}

	pub fn pitch(mut self, pitch: f32) -> Self {
		self.pitch = pitch;
		self
	}

	pub fn movement_speed(mut self, movement_speed: f32) -> Self {
		self.movement_speed = movement_speed;
		self
	}

	pub fn mouse_sensitivity(mut self, mouse_sensitivity: f32) -> Self {
		self.mouse_sensitivity = mouse_sensitivity;
		self
	}

	pub fn zoom(mut self, zoom: f32) -> Self {
		self.zoom = zoom;
		self
	}

	pub fn build(self) -> Camera {
		Camera::new(
			self.position,
			self.world_up,
			self.yaw,
			self.pitch,
			self.movement_speed,
			self.mouse_sensitivity,
			self.zoom
		)
	}
}