mod agent;
mod app;
mod environment;
mod episode;
mod maze;
mod mouse;
mod q_learning;
mod runner;

pub use app::TheseusApp;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(TheseusApp::new()))),
        )
        .await
}
