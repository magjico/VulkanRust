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
/// - `yaw` (`f32`) - Horizontal rotation around the world up-axis (left-right looking).
/// - `pitch` (`f32`) - Vertical rotation around the camera's right axis (up-down looking).
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

    // Rotation representation using Euler angles.
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
    /// Internal coordinate system maintenance.
    /// Ensures mathematical consistency when orientation changes occur
    /// TODO
    fn update_camera_vectors() {}

    // pub fn create(
    //     position: Vec3,
    //     up: Vec3,
    //     yaw: f32,
    //     pitch: f32
    // ) -> Self {

    // }

    // pub fn get_view_matrix(&self) -> Mat4 {

    // }

    // pub fn get_projection_matrix(
    //     &self,
    //     aspect_ratio: f32,
    //     near_plane: Option<f32>,
    //     far_plane: Option<f32>
    // ) -> Mat4 {
    //     let near_plane = near_plane.unwrap_or(0.1);
    //     let far_plane = far_plane.unwrap_or(100.0);

    // }

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
        &self,
        x_offset: f32,
        y_offset: f32,
        constrain_pitch: Option<bool>
    ) {
        let constrain_pitch = constrain_pitch.unwrap_or(true);
    }

    pub fn process_mouse_scroll(&self, y_offset: f32) {

    }

    pub fn get_position(&self) -> Vec3 { self.position }
    pub fn get_front(&self) -> Vec3 { self.front }
    pub fn get_zoom(&self) -> f32 { self.zoom }
}