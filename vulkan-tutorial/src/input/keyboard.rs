//! Manage Keyboard input for the app (structs)
use std::collections::HashMap;
use anyhow::{anyhow, Result};

use serde::Deserialize;
use winit::keyboard::KeyCode;

use crate::scene::CameraMovement;

#[derive(Deserialize)]
struct InputConfig {
    camera_control: CameraInputConfig,
}

#[derive(Deserialize)]
struct CameraInputConfig {
    forward:    String,
    backward:   String,
    left:       String,
    right:      String,
    up:         String,
    down:       String,
}

#[derive(Debug, Default, Clone)]
pub struct InputBindings {
    pub camera_bindings: HashMap<KeyCode, CameraMovement>,
}

impl InputBindings {
    pub fn bind_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let toml: InputConfig = toml::from_str(&content)?;

        let mut camera_bindings = HashMap::new();
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.forward)?, CameraMovement::Forward);
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.backward)?, CameraMovement::Backward);
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.left)?, CameraMovement::Left);
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.right)?, CameraMovement::Right);
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.up)?, CameraMovement::Up);
        camera_bindings.insert(InputBindings::parse_keycode(&toml.camera_control.down)?, CameraMovement::Down);

        Ok(Self {
            camera_bindings,
        })
    }

    fn parse_keycode(s: &str) -> Result<KeyCode> {
        match s.to_lowercase().as_str() {
            // Letters
            "a" => Ok(KeyCode::KeyA),
            "b" => Ok(KeyCode::KeyB),
            "c" => Ok(KeyCode::KeyC),
            "d" => Ok(KeyCode::KeyD),
            "e" => Ok(KeyCode::KeyE),
            "f" => Ok(KeyCode::KeyF),
            "g" => Ok(KeyCode::KeyG),
            "h" => Ok(KeyCode::KeyH),
            "i" => Ok(KeyCode::KeyI),
            "j" => Ok(KeyCode::KeyJ),
            "k" => Ok(KeyCode::KeyK),
            "l" => Ok(KeyCode::KeyL),
            "m" => Ok(KeyCode::KeyM),
            "n" => Ok(KeyCode::KeyN),
            "o" => Ok(KeyCode::KeyO),
            "p" => Ok(KeyCode::KeyP),
            "q" => Ok(KeyCode::KeyQ),
            "r" => Ok(KeyCode::KeyR),
            "s" => Ok(KeyCode::KeyS),
            "t" => Ok(KeyCode::KeyT),
            "u" => Ok(KeyCode::KeyU),
            "v" => Ok(KeyCode::KeyV),
            "w" => Ok(KeyCode::KeyW),
            "x" => Ok(KeyCode::KeyX),
            "y" => Ok(KeyCode::KeyY),
            "z" => Ok(KeyCode::KeyZ),

            // Numbers
            "0" => Ok(KeyCode::Digit0),
            "1" => Ok(KeyCode::Digit1),
            "2" => Ok(KeyCode::Digit2),
            "3" => Ok(KeyCode::Digit3),
            "4" => Ok(KeyCode::Digit4),
            "5" => Ok(KeyCode::Digit5),
            "6" => Ok(KeyCode::Digit6),
            "7" => Ok(KeyCode::Digit7),
            "8" => Ok(KeyCode::Digit8),
            "9" => Ok(KeyCode::Digit9),

            // function
            "f1"  => Ok(KeyCode::F1),
            "f2"  => Ok(KeyCode::F2),
            "f3"  => Ok(KeyCode::F3),
            "f4"  => Ok(KeyCode::F4),
            "f5"  => Ok(KeyCode::F5),
            "f6"  => Ok(KeyCode::F6),
            "f7"  => Ok(KeyCode::F7),
            "f8"  => Ok(KeyCode::F8),
            "f9"  => Ok(KeyCode::F9),
            "f10" => Ok(KeyCode::F10),
            "f11" => Ok(KeyCode::F11),
            "f12" => Ok(KeyCode::F12),
            "f13" => Ok(KeyCode::F13),
            "f14" => Ok(KeyCode::F14),
            "f15" => Ok(KeyCode::F15),
            "f16" => Ok(KeyCode::F16),
            "f17" => Ok(KeyCode::F17),
            "f18" => Ok(KeyCode::F18),
            "f19" => Ok(KeyCode::F19),
            "f20" => Ok(KeyCode::F20),

            // left modifier
            "ctrl_left" | "ctrl"	=> Ok(KeyCode::ControlLeft),
            "shift_left" | "shift"	=> Ok(KeyCode::ShiftLeft),
            "alt_left" | "alt"		=> Ok(KeyCode::AltLeft),
            "superleft"				=> Ok(KeyCode::SuperLeft),

            // right modifier
            "ctrl_right"			=> Ok(KeyCode::ControlRight),
            "shift_right"			=> Ok(KeyCode::ShiftRight),
            "alt_right" | "altgr"	=> Ok(KeyCode::AltRight),
            "superright"			=> Ok(KeyCode::SuperRight),

            // navigation
            "up" | "arrowup"		=> Ok(KeyCode::ArrowUp),
            "down" | "arrowdown"	=> Ok(KeyCode::ArrowDown),
            "left" | "arrowleft"	=> Ok(KeyCode::ArrowLeft),
            "right" | "arrowright"	=> Ok(KeyCode::ArrowRight),
            "home"					=> Ok(KeyCode::Home),
            "end"					=> Ok(KeyCode::End),
            "pageup"				=> Ok(KeyCode::PageUp),
            "pagedown"				=> Ok(KeyCode::PageDown),
            "insert"				=> Ok(KeyCode::Insert),
            "delete" | "del"		=> Ok(KeyCode::Delete),

            // main keys
            "space"					=> Ok(KeyCode::Space),
            "enter" | "return"		=> Ok(KeyCode::Enter),
            "escape" | "esc"		=> Ok(KeyCode::Escape),
            "tab"					=> Ok(KeyCode::Tab),
            "backspace"				=> Ok(KeyCode::Backspace),
            "capslock" | "maj"		=> Ok(KeyCode::CapsLock),

			// numpad
			"numpad0" | "num0"		=> Ok(KeyCode::Numpad0),
			"numpad1" | "num1"		=> Ok(KeyCode::Numpad1),
			"numpad2" | "num2"		=> Ok(KeyCode::Numpad2),
			"numpad3" | "num3"		=> Ok(KeyCode::Numpad3),
			"numpad4" | "num4"		=> Ok(KeyCode::Numpad4),
			"numpad5" | "num5"		=> Ok(KeyCode::Numpad5),
			"numpad6" | "num6"		=> Ok(KeyCode::Numpad6),
			"numpad7" | "num7"		=> Ok(KeyCode::Numpad7),
			"numpad8" | "num8"		=> Ok(KeyCode::Numpad8),
			"numpad9" | "num9"		=> Ok(KeyCode::Numpad9),
			"numpadenter"			=> Ok(KeyCode::NumpadEnter),
			"numpadadd" | "num+"	=> Ok(KeyCode::NumpadAdd),
			"numpadsub" | "num-"	=> Ok(KeyCode::NumpadSubtract),
			"numpadmul" | "num*"	=> Ok(KeyCode::NumpadMultiply),
			"numpaddiv" | "num/"	=> Ok(KeyCode::NumpadDivide),
			"numpaddot" | "num."	=> Ok(KeyCode::NumpadDecimal),
			"numlock"				=> Ok(KeyCode::NumLock),

			// AZERTY special characters
			"&" => Ok(KeyCode::Digit1),  // & = Shift+1
			"é" => Ok(KeyCode::Digit2),  // é = 2
			"\"" => Ok(KeyCode::Digit3),  // " = Shift+3
			"'" => Ok(KeyCode::Digit4),  // ' = 4
			"(" => Ok(KeyCode::Digit5),  // ( = 5
			"-" => Ok(KeyCode::Minus),
			"è" => Ok(KeyCode::Digit7),  // è = 7
			"_" => Ok(KeyCode::Minus),   // _ = Shift+-
			"ç" => Ok(KeyCode::Digit9),  // ç = 9
			"à" => Ok(KeyCode::Digit0),  // à = 0
			")" => Ok(KeyCode::Digit6),  // ) = 6
			"²" => Ok(KeyCode::Backquote),

			// Ponctuation
			","  | "comma" 			=> Ok(KeyCode::Comma),
			"."  | "period"			=> Ok(KeyCode::Period),
			";"  | "semicolon"		=> Ok(KeyCode::Semicolon),
			":"  | "colon"			=> Ok(KeyCode::Semicolon),
			"!"  | "exclamation"	=> Ok(KeyCode::Digit8),
			"/"  | "slash"			=> Ok(KeyCode::Slash),
			"\\" | "backslash"		=> Ok(KeyCode::Backslash),
			"["  | "bracketleft"	=> Ok(KeyCode::BracketLeft),
			"]"  | "bracketright"	=> Ok(KeyCode::BracketRight),
			"="  | "equal"			=> Ok(KeyCode::Equal),
			"`"  | "backtick"		=> Ok(KeyCode::Backquote),

			other => Err(anyhow!("Unknown keycode: {}", other))
        }
    }
}