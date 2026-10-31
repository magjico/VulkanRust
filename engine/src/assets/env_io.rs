use anyhow::Result;

use exr::prelude::read_first_rgba_layer_from_file;

pub struct EquirectImage {
	pub pixels: Vec<f32>,
	pub width: usize,
	pub height: usize,
}

pub fn load_equirect_exr(path: &str) -> Result<EquirectImage> {
	let image = read_first_rgba_layer_from_file(
		path,
		|res, _| EquirectImage {
			pixels: vec![0.0f32; res.width() * res.height() * 4],
			width: res.width(),
			height: res.height(),
		},
		|img, position, (r, g, b, a): (f32, f32, f32, f32)| {
			let idx = (position.y() * img.width + position.x()) * 4;
			img.pixels[idx] = r;
			img.pixels[idx + 1] = g;
			img.pixels[idx + 2] = b;
			img.pixels[idx + 3] = a;
		}
	)?;

	Ok(image.layer_data.channel_data.pixels)
}