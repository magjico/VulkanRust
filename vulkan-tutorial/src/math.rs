use cgmath::InnerSpace;

pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Vec4 = cgmath::Vector4<f32>;
pub type Quat = cgmath::Quaternion<f32>;
pub type Mat3 = cgmath::Matrix3<f32>;
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

pub fn extract_scale_from_mat4(m: &Mat4) -> Vec3 {
    Vec3::new(
        Vec3::new(m.x.x, m.x.y, m.x.z).magnitude(),
        Vec3::new(m.y.x, m.y.y, m.y.z).magnitude(),
        Vec3::new(m.z.x, m.z.y, m.z.z).magnitude(),
    )
}

pub fn extract_normalize_rot_from_mat4(m: &Mat4) -> (Vec3, Vec3, Vec3) {
    let x = Vec3::new(m.x.x, m.x.y, m.x.z).normalize();
    let y = Vec3::new(m.y.x, m.y.y, m.y.z).normalize();
    let z = Vec3::new(m.z.x, m.z.y, m.z.z).normalize();

    (x, y, z)
}

/// decompose an homogeous matrix into:
/// * a **translation** [Vec3]
/// * a **rotation** [Quat]ernion
/// * a **scale** [Vec3]
/// 
/// in this order.
pub fn decompose(m: &Mat4) -> (Vec3, Quat, Vec3) {
    let translation = Vec3::new(m.w.x, m.w.y, m.w.z);

    let scale = extract_scale_from_mat4(m);

    let rotation = Quat::from(Mat3::new(
        m.x.x / scale.x, m.x.y / scale.x, m.x.z / scale.x,
        m.y.x / scale.y, m.y.y / scale.y, m.y.z / scale.y,
        m.z.x / scale.z, m.z.y / scale.z, m.z.z / scale.z,
    ));

    (translation, rotation, scale)
}

