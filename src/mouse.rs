use crate::{
    environment::Environment,
    maze::{Direction, Position},
};

#[derive(Debug)]
pub struct Mouse {
    position: Position,
}

impl Mouse {
    pub fn new(position: Position) -> Self {
        Self { position }
    }

    pub fn position(&self) -> Position {
        self.position
    }

    pub(crate) fn set_position(&mut self, position: Position) {
        self.position = position;
    }

    pub fn try_move(&mut self, environment: &Environment, direction: Direction) -> bool {
        let Some(next) = environment.step(self.position, direction) else {
            return false;
        };

        self.position = next;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maze::Maze;

    #[test]
    fn new_mouse_starts_at_requested_position() {
        let mouse = Mouse::new(Position::new(2, 3));

        assert_eq!(mouse.position(), Position::new(2, 3));
    }

    #[test]
    fn mouse_does_not_move_through_wall() {
        let maze = Maze::new(2, 2);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut mouse = Mouse::new(Position::new(0, 0));

        assert!(!mouse.try_move(&environment, Direction::East));

        assert_eq!(mouse.position(), Position::new(0, 0));
    }

    #[test]
    fn mouse_moves_through_open_passage() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut mouse = Mouse::new(Position::new(0, 0));

        assert!(mouse.try_move(&environment, Direction::East));

        assert_eq!(mouse.position(), Position::new(1, 0));
    }
}
