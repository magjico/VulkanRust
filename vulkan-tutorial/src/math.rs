pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Vec4 = cgmath::Vector4<f32>;
pub type Quat = cgmath::Quaternion<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

pub type UVec4 = cgmath::Vector4<u16>;

pub fn vec4_to_quat(v: Vec4) -> Quat {
    Quat::new(
        v.w,
        v.x,
        v.y,
        v.z,
    )
}