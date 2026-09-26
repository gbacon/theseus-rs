use theseus_rs::TheseusApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Theseus",
        options,
        Box::new(|_cc| Ok(Box::new(TheseusApp::new()))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
