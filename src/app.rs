use eframe::egui;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashSet;

use crate::{
    agent::Agent,
    environment::Environment,
    episode::Episode,
    maze::{Direction, Maze, Position},
    q_learning::QLearningAgent,
    runner::{EvaluationResult, RunResult},
};

pub struct TheseusApp {
    environment: Environment,
    mouse: Position,
    start: Position,
    agent: QLearningAgent,
    training_rng: StdRng,
    seed: u64,
    episode_count: usize,
    last_result: Option<RunResult>,
    active_episode: Option<Episode>,
    running: bool,
    steps_per_second: f32,
    last_step_time: f64,
    evaluation_agent: Option<QLearningAgent>,
    evaluation_episode: Option<Episode>,
    evaluation_mouse: Option<Position>,
    evaluation_visited: HashSet<Position>,
    evaluation_result: Option<EvaluationResult>,
    evaluation_snapshot_episode: Option<usize>,
    evaluation_running: bool,
    evaluation_cycle_detected: bool,
    evaluation_max_actions: usize,
}

impl TheseusApp {
    pub fn new() -> Self {
        let width = 10;
        let height = 10;
        let seed = 42;

        let start = Position::new(0, 0);
        let goal = Position::new(width - 1, height - 1);

        let mut maze = Maze::new(width, height);
        let mut maze_rng = StdRng::seed_from_u64(seed);
        maze.generate_with(&mut maze_rng);

        Self {
            environment: Environment::new(maze, goal),
            mouse: start,
            start,
            agent: QLearningAgent::new(width, height, 0.2, 0.5, 0.99),
            training_rng: StdRng::seed_from_u64(seed.wrapping_add(1)),
            seed,
            episode_count: 0,
            last_result: None,
            active_episode: None,
            running: false,
            steps_per_second: 10.0,
            last_step_time: 0.0,
            evaluation_agent: None,
            evaluation_episode: None,
            evaluation_mouse: None,
            evaluation_visited: HashSet::new(),
            evaluation_result: None,
            evaluation_snapshot_episode: None,
            evaluation_running: false,
            evaluation_cycle_detected: false,
            evaluation_max_actions: 500,
        }
    }

    fn regenerate_maze(&mut self) {
        let width = self.environment.maze().width();
        let height = self.environment.maze().height();

        let start = Position::new(0, 0);
        let goal = Position::new(width - 1, height - 1);

        let mut maze = Maze::new(width, height);
        let mut maze_rng = StdRng::seed_from_u64(self.seed);
        maze.generate_with(&mut maze_rng);

        self.environment = Environment::new(maze, goal);
        self.mouse = start;
        self.start = start;

        self.agent = QLearningAgent::new(width, height, 0.2, 0.5, 0.9);

        self.training_rng = StdRng::seed_from_u64(self.seed.wrapping_add(1));

        self.episode_count = 0;
        self.last_result = None;
        self.active_episode = None;
        self.running = false;
        self.evaluation_visited.clear();
        self.evaluation_result = None;
        self.evaluation_snapshot_episode = None;
        self.evaluation_running = false;
    }

    fn start_training_episode(&mut self) {
        self.active_episode = Some(Episode::new(self.start, &self.environment));
        self.mouse = self.start;
        self.start_evaluation();
    }

    fn training_step(&mut self) {
        let Some(episode) = self.active_episode.as_mut() else {
            return;
        };

        if episode.is_complete() {
            return;
        }

        let state = episode.position();

        let action = self.agent.choose_action(state, &mut self.training_rng);

        let result = self.environment.act(state, action);

        self.agent.learn(
            state,
            action,
            result.reward,
            result.position,
            result.terminated,
        );

        episode.apply_result(result);

        self.mouse = episode.position();

        if episode.is_complete() {
            self.episode_count += 1;

            self.last_result = Some(RunResult {
                completed: true,
                actions: episode.actions(),
                moves: episode.moves(),
            });

            self.active_episode = None;
        }
    }

    fn draw_maze(&self, ui: &mut egui::Ui) {
        let maze = self.environment.maze();

        let size = ui.available_width().min(600.0);

        let (response, painter) = ui.allocate_painter(egui::vec2(size, size), egui::Sense::hover());

        let rect = response.rect;

        let cell_width = rect.width() / maze.width() as f32;

        let cell_height = rect.height() / maze.height() as f32;

        let stroke = egui::Stroke::new(2.0, ui.visuals().text_color());

        for y in 0..maze.height() {
            for x in 0..maze.width() {
                let position = Position::new(x, y);

                let left = rect.left() + x as f32 * cell_width;

                let top = rect.top() + y as f32 * cell_height;

                let right = left + cell_width;
                let bottom = top + cell_height;

                if maze.has_wall(position, Direction::North) {
                    painter.line_segment([egui::pos2(left, top), egui::pos2(right, top)], stroke);
                }

                if maze.has_wall(position, Direction::West) {
                    painter.line_segment([egui::pos2(left, top), egui::pos2(left, bottom)], stroke);
                }

                if x == maze.width() - 1 && maze.has_wall(position, Direction::East) {
                    painter
                        .line_segment([egui::pos2(right, top), egui::pos2(right, bottom)], stroke);
                }

                if y == maze.height() - 1 && maze.has_wall(position, Direction::South) {
                    painter.line_segment(
                        [egui::pos2(left, bottom), egui::pos2(right, bottom)],
                        stroke,
                    );
                }
            }
        }

        for y in 0..maze.height() {
            for x in 0..maze.width() {
                let position = Position::new(x, y);

                if !self.agent.has_learned_value(position) {
                    continue;
                }

                let direction = self.agent.best_action_for(position);

                if let Some(evaluation_agent) = &self.evaluation_agent {
                    if evaluation_agent.has_learned_value(position) {
                        let direction = evaluation_agent.best_action_for(position);

                        let center = egui::pos2(
                            rect.left() + (x as f32 + 0.5) * cell_width,
                            rect.top() + (y as f32 + 0.5) * cell_height,
                        );

                        let length = cell_width.min(cell_height) * 0.14;

                        let offset = match direction {
                            Direction::North => egui::vec2(0.0, -length),
                            Direction::East => egui::vec2(length, 0.0),
                            Direction::South => egui::vec2(0.0, length),
                            Direction::West => egui::vec2(-length, 0.0),
                        };

                        painter.arrow(center, offset, egui::Stroke::new(2.0, egui::Color32::GREEN));
                    }
                }

                let center = egui::pos2(
                    rect.left() + (x as f32 + 0.5) * cell_width,
                    rect.top() + (y as f32 + 0.5) * cell_height,
                );

                let length = cell_width.min(cell_height) * 0.22;

                let offset = match direction {
                    Direction::North => egui::vec2(0.0, -length),
                    Direction::East => egui::vec2(length, 0.0),
                    Direction::South => egui::vec2(0.0, length),
                    Direction::West => egui::vec2(-length, 0.0),
                };

                painter.arrow(center, offset, egui::Stroke::new(1.0, egui::Color32::GRAY));
            }
        }

        let cell_center = |position: Position| {
            egui::pos2(
                rect.left() + (position.x as f32 + 0.5) * cell_width,
                rect.top() + (position.y as f32 + 0.5) * cell_height,
            )
        };

        let marker_radius = cell_width.min(cell_height) * 0.25;

        painter.circle_filled(
            cell_center(self.environment.goal()),
            marker_radius,
            egui::Color32::GOLD,
        );

        painter.circle_filled(
            cell_center(self.mouse),
            marker_radius,
            egui::Color32::from_rgb(120, 170, 255),
        );

        if let Some(position) = self.evaluation_mouse {
            painter.circle_filled(
                cell_center(position),
                marker_radius * 0.75,
                egui::Color32::GREEN,
            );
        }

        ui.horizontal(|ui| {
            ui.label("Training mouse: blue");
            ui.separator();
            ui.label("Evaluation mouse: green");
            ui.separator();
            ui.label("Cheese: gold");
        });
    }

    fn update_animation(&mut self, ctx: &egui::Context) {
        if !self.running && !self.evaluation_running {
            return;
        }

        if self.running && self.active_episode.is_none() {
            self.start_training_episode();
        }

        let now = ctx.input(|input| input.time);
        let interval = 1.0 / self.steps_per_second as f64;

        if now - self.last_step_time >= interval {
            if self.running {
                self.training_step();
            }

            if self.evaluation_running {
                self.evaluation_step();
            }

            self.last_step_time = now;
        }

        ctx.request_repaint();
    }

    fn start_evaluation(&mut self) {
        let mut evaluation_agent = self.agent.clone();
        evaluation_agent.set_epsilon(0.0);

        self.evaluation_agent = Some(evaluation_agent);
        self.evaluation_episode = Some(Episode::new(self.start, &self.environment));
        self.evaluation_mouse = Some(self.start);

        self.evaluation_visited.clear();
        self.evaluation_cycle_detected = false;
        self.evaluation_result = None;

        self.evaluation_snapshot_episode = Some(self.episode_count);
        self.evaluation_running = true;
    }

    fn evaluation_step(&mut self) {
        if !self.evaluation_running {
            return;
        }

        let Some(agent) = self.evaluation_agent.as_ref() else {
            self.evaluation_running = false;
            return;
        };

        let Some(episode) = self.evaluation_episode.as_mut() else {
            self.evaluation_running = false;
            return;
        };

        if episode.is_complete() {
            self.evaluation_result = Some(EvaluationResult {
                completed: true,
                cycle_detected: self.evaluation_cycle_detected,
                actions: episode.actions(),
                moves: episode.moves(),
            });

            self.evaluation_running = false;
            return;
        }

        if episode.actions() >= self.evaluation_max_actions {
            self.evaluation_result = Some(EvaluationResult {
                completed: false,
                cycle_detected: self.evaluation_cycle_detected,
                actions: episode.actions(),
                moves: episode.moves(),
            });

            self.evaluation_running = false;
            return;
        }

        let state = episode.position();

        if !self.evaluation_visited.insert(state) {
            self.evaluation_cycle_detected = true;
        }

        let action = agent.best_action_for(state);

        let result = self.environment.act(state, action);

        episode.apply_result(result);
        self.evaluation_mouse = Some(episode.position());

        if episode.is_complete() {
            self.evaluation_result = Some(EvaluationResult {
                completed: true,
                cycle_detected: self.evaluation_cycle_detected,
                actions: episode.actions(),
                moves: episode.moves(),
            });

            self.evaluation_running = false;
        }
    }
}

impl eframe::App for TheseusApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_animation(ui.ctx());

        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Theseus");

            ui.separator();

            // Maze controls.
            ui.horizontal(|ui| {
                ui.label("Seed:");

                let changed = ui.add(egui::DragValue::new(&mut self.seed)).changed();

                if ui.button("New Maze").clicked() || changed {
                    self.regenerate_maze();
                }
            });

            ui.separator();

            // Training and evaluation controls.
            ui.horizontal(|ui| {
                if ui.button("Start Episode").clicked() {
                    self.start_training_episode();
                }

                ui.add_enabled_ui(!self.running, |ui| {
                    if ui.button("Step").clicked() {
                        self.training_step();
                    }
                });

                if ui
                    .button(if self.running { "Pause" } else { "Run" })
                    .clicked()
                {
                    self.running = !self.running;
                    self.last_step_time = ui.input(|input| input.time);
                }

                if ui.button("Evaluate").clicked() {
                    self.start_evaluation();
                }

                ui.add_enabled_ui(!self.evaluation_running, |ui| {
                    if ui.button("Eval Step").clicked() {
                        self.evaluation_step();
                    }
                });
            });

            ui.horizontal(|ui| {
                ui.label("Speed:");

                ui.add(
                    egui::Slider::new(&mut self.steps_per_second, 1.0..=60.0)
                        .logarithmic(true)
                        .suffix(" steps/s"),
                );

                ui.separator();

                ui.label(format!(
                    "Completed training episodes: {}",
                    self.episode_count
                ));

                if let Some(snapshot_episode) = self.evaluation_snapshot_episode {
                    ui.label(format!(
                        "Green snapshot: after {} completed training episode{}",
                        snapshot_episode,
                        if snapshot_episode == 1 { "" } else { "s" }
                    ));
                }
            });

            // Live training status.
            if let Some(episode) = &self.active_episode {
                ui.label(format!(
                    "Current episode: actions={}, moves={}",
                    episode.actions(),
                    episode.moves(),
                ));
            }

            if let Some(result) = self.last_result {
                ui.label(format!(
                    "Last episode: completed={}, actions={}, moves={}",
                    result.completed, result.actions, result.moves,
                ));
            }

            ui.separator();

            // Maze visualization.
            self.draw_maze(ui);

            // Legend.
            ui.horizontal(|ui| {
                ui.label("Blue mouse: training agent");
                ui.label("Gray arrows: current live learned policy");
                ui.label("Green mouse: frozen evaluation snapshot");
                ui.label("Green arrows: frozen snapshot policy");
                ui.label("Gold marker: cheese");
            });

            // Evaluation status.
            if self.evaluation_agent.is_some() {
                ui.separator();

                ui.heading("Evaluation");

                if let Some(snapshot_episode) = self.evaluation_snapshot_episode {
                    ui.label(format!("Policy snapshot after episode {snapshot_episode}"));
                }

                if let Some(result) = &self.evaluation_result {
                    ui.separator();
                    ui.heading("Evaluation");

                    ui.label(format!("Completed: {}", result.completed));
                    ui.label(format!(
                        "Cycle detected: {}",
                        self.evaluation_cycle_detected
                    ));
                    ui.label(format!("Actions: {}", result.actions));
                    ui.label(format!("Moves: {}", result.moves));

                    if result.completed {
                        if let Some(optimal) = self
                            .environment
                            .maze()
                            .shortest_path_length(self.start, self.environment.goal())
                        {
                            ui.label(format!("Optimal path length: {}", optimal));

                            if optimal > 0 {
                                let overhead = result.moves as f32 / optimal as f32;
                                ui.label(format!("Move overhead: {:.2}×", overhead));
                            }
                        }
                    }
                } else if let Some(episode) = &self.evaluation_episode {
                    ui.separator();
                    ui.heading("Evaluation");

                    ui.label("Status: running");
                    ui.label(format!(
                        "Cycle detected: {}",
                        self.evaluation_cycle_detected
                    ));
                    ui.label(format!("Actions: {}", episode.actions()));
                    ui.label(format!("Moves: {}", episode.moves()));
                }
            }
        });
    }
}
