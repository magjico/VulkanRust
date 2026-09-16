mod resources;
mod core;
mod manager;

pub use manager::AppManager;
pub use core::{App, AppData};
pub use resources::{Persistent, SwapchainDependant};