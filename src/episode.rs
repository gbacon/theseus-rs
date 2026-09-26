use crate::{
    environment::{Environment, StepResult},
    maze::{Direction, Position},
    mouse::Mouse,
};

#[derive(Debug)]
pub struct Episode {
    mouse: Mouse,
    actions: usize,
    moves: usize,
    complete: bool,
}

impl Episode {
    pub fn new(start: Position, environment: &Environment) -> Self {
        let complete = environment.is_goal(start);

        Self {
            mouse: Mouse::new(start),
            actions: 0,
            moves: 0,
            complete,
        }
    }

    pub fn actions(&self) -> usize {
        self.actions
    }

    pub fn position(&self) -> Position {
        self.mouse.position()
    }

    pub fn moves(&self) -> usize {
        self.moves
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn step(&mut self, environment: &Environment, direction: Direction) -> bool {
        if self.complete {
            return false;
        }

        let result = environment.act(self.mouse.position(), direction);

        self.apply_result(result)
    }

    pub fn apply_result(&mut self, result: StepResult) -> bool {
        if self.complete {
            return false;
        }

        self.actions += 1;

        if result.moved {
            self.mouse.set_position(result.position);
            self.moves += 1;
        }

        if result.terminated {
            self.complete = true;
        }

        result.moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::Maze;

    #[test]
    fn new_episode_starts_at_requested_position() {
        let maze = Maze::new(2, 2);
        let environment = Environment::new(maze, Position::new(1, 1));

        let episode = Episode::new(Position::new(0, 0), &environment);

        assert_eq!(episode.position(), Position::new(0, 0));
        assert_eq!(episode.moves(), 0);
        assert!(!episode.is_complete());
    }

    #[test]
    fn episode_is_complete_if_start_is_goal() {
        let maze = Maze::new(2, 2);
        let goal = Position::new(0, 0);

        let environment = Environment::new(maze, goal);

        let episode = Episode::new(goal, &environment);

        assert!(episode.is_complete());
        assert_eq!(episode.moves(), 0);
    }

    #[test]
    fn blocked_move_does_not_increment_move_count() {
        let maze = Maze::new(2, 2);
        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        assert!(!episode.step(&environment, Direction::East));

        assert_eq!(episode.actions(), 1);
        assert_eq!(episode.moves(), 0);
        assert_eq!(episode.position(), Position::new(0, 0));
    }

    #[test]
    fn successful_move_increments_move_count() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        assert!(episode.step(&environment, Direction::East));

        assert_eq!(episode.actions(), 1);
        assert_eq!(episode.moves(), 1);
        assert_eq!(episode.position(), Position::new(1, 0));
    }

    #[test]
    fn reaching_goal_completes_episode() {
        let mut maze = Maze::new(2, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 0));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        assert!(episode.step(&environment, Direction::East));

        assert!(episode.is_complete());
        assert_eq!(episode.actions(), 1);
        assert_eq!(episode.moves(), 1);
    }

    #[test]
    fn completed_episode_rejects_further_moves() {
        let mut maze = Maze::new(2, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 0));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        assert!(episode.step(&environment, Direction::East));

        assert!(!episode.step(&environment, Direction::West));

        assert_eq!(episode.moves(), 1);
        assert_eq!(episode.position(), Position::new(1, 0));
    }

    #[test]
    fn blocked_action_counts_as_action_but_not_move() {
        let maze = Maze::new(2, 2);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        assert!(!episode.step(&environment, Direction::East));

        assert_eq!(episode.actions(), 1);
        assert_eq!(episode.moves(), 0);
    }

    #[test]
    fn apply_result_updates_episode_state() {
        let maze = Maze::new(2, 2);
        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let result = StepResult {
            position: Position::new(1, 0),
            reward: -1.0,
            moved: true,
            terminated: false,
        };

        assert!(episode.apply_result(result));

        assert_eq!(episode.actions(), 1);
        assert_eq!(episode.moves(), 1);
        assert_eq!(episode.position(), Position::new(1, 0));
        assert!(!episode.is_complete());
    }
}
