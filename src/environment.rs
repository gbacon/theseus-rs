use crate::maze::{Direction, Maze, Position};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    pub position: Position,
    pub reward: f32,
    pub moved: bool,
    pub terminated: bool,
}

const STEP_REWARD: f32 = -1.0;
const WALL_REWARD: f32 = -5.0;
const GOAL_REWARD: f32 = 100.0;

#[derive(Debug)]
pub struct Environment {
    maze: Maze,
    goal: Position,
}

impl Environment {
    pub fn new(maze: Maze, goal: Position) -> Self {
        assert!(
            goal.x < maze.width() && goal.y < maze.height(),
            "goal must be inside the maze"
        );

        Self { maze, goal }
    }

    pub fn maze(&self) -> &Maze {
        &self.maze
    }

    pub fn goal(&self) -> Position {
        self.goal
    }

    pub fn is_goal(&self, position: Position) -> bool {
        position == self.goal
    }

    pub fn step(&self, position: Position, direction: Direction) -> Option<Position> {
        self.maze.step(position, direction)
    }

    pub fn act(&self, position: Position, direction: Direction) -> StepResult {
        let Some(next) = self.maze.step(position, direction) else {
            return StepResult {
                position,
                reward: WALL_REWARD,
                moved: false,
                terminated: false,
            };
        };

        let terminated = self.is_goal(next);

        StepResult {
            position: next,
            reward: if terminated { GOAL_REWARD } else { STEP_REWARD },
            moved: true,
            terminated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_stores_goal() {
        let maze = Maze::new(4, 3);
        let goal = Position::new(3, 2);

        let environment = Environment::new(maze, goal);

        assert_eq!(environment.goal(), goal);
    }

    #[test]
    fn recognizes_goal_position() {
        let maze = Maze::new(4, 3);
        let environment = Environment::new(maze, Position::new(3, 2));

        assert!(environment.is_goal(Position::new(3, 2)));
        assert!(!environment.is_goal(Position::new(2, 2)));
    }

    #[test]
    #[should_panic(expected = "goal must be inside the maze")]
    fn rejects_goal_outside_maze() {
        let maze = Maze::new(4, 3);

        Environment::new(maze, Position::new(4, 2));
    }

    #[test]
    fn environment_applies_valid_movement() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 1));

        assert_eq!(
            environment.step(Position::new(0, 0), Direction::East,),
            Some(Position::new(1, 0))
        );
    }

    #[test]
    fn hitting_wall_keeps_position_and_penalizes_action() {
        let maze = Maze::new(2, 2);

        let environment = Environment::new(maze, Position::new(1, 1));

        let result = environment.act(Position::new(0, 0), Direction::East);

        assert_eq!(result.position, Position::new(0, 0));
        assert!(!result.moved);
        assert_eq!(result.reward, WALL_REWARD);
        assert!(!result.terminated);
    }

    #[test]
    fn ordinary_move_has_step_reward() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 1));

        let result = environment.act(Position::new(0, 0), Direction::East);

        assert_eq!(result.position, Position::new(1, 0));
        assert!(result.moved);
        assert_eq!(result.reward, STEP_REWARD);
        assert!(!result.terminated);
    }

    #[test]
    fn reaching_goal_returns_goal_reward_and_terminates() {
        let mut maze = Maze::new(2, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 0));

        let result = environment.act(Position::new(0, 0), Direction::East);

        assert_eq!(result.position, Position::new(1, 0));
        assert!(result.moved);
        assert_eq!(result.reward, GOAL_REWARD);
        assert!(result.terminated);
    }
}
