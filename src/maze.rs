use rand::Rng;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

pub const DIRECTIONS: [Direction; 4] = [
    Direction::North,
    Direction::East,
    Direction::South,
    Direction::West,
];

impl Direction {
    fn delta(self) -> (isize, isize) {
        match self {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        }
    }

    fn opposite(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

impl Position {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone)]
struct Cell {
    north: bool,
    east: bool,
    south: bool,
    west: bool,
    visited: bool,
}

impl Cell {
    fn new() -> Self {
        Self {
            north: true,
            east: true,
            south: true,
            west: true,
            visited: false,
        }
    }

    fn remove_wall(&mut self, direction: Direction) {
        match direction {
            Direction::North => self.north = false,
            Direction::East => self.east = false,
            Direction::South => self.south = false,
            Direction::West => self.west = false,
        }
    }

    fn has_wall(&self, direction: Direction) -> bool {
        match direction {
            Direction::North => self.north,
            Direction::East => self.east,
            Direction::South => self.south,
            Direction::West => self.west,
        }
    }
}

#[derive(Debug)]
pub struct Maze {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Maze {
    pub fn new(width: usize, height: usize) -> Self {
        let cells = vec![Cell::new(); width * height];

        Self {
            width,
            height,
            cells,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn has_wall(&self, position: Position, direction: Direction) -> bool {
        self.cell(position.x, position.y).has_wall(direction)
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn cell(&self, x: usize, y: usize) -> &Cell {
        &self.cells[self.index(x, y)]
    }

    fn neighbor(&self, x: usize, y: usize, direction: Direction) -> Option<(usize, usize)> {
        let (dx, dy) = direction.delta();

        let nx = x as isize + dx;
        let ny = y as isize + dy;

        if nx < 0 || ny < 0 || nx >= self.width as isize || ny >= self.height as isize {
            None
        } else {
            Some((nx as usize, ny as usize))
        }
    }

    fn cell_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        let index = self.index(x, y);
        &mut self.cells[index]
    }

    fn remove_wall(&mut self, x: usize, y: usize, direction: Direction) -> bool {
        let Some((nx, ny)) = self.neighbor(x, y, direction) else {
            return false;
        };

        self.cell_mut(x, y).remove_wall(direction);
        self.cell_mut(nx, ny).remove_wall(direction.opposite());

        true
    }

    pub fn generate_with<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        use rand::seq::SliceRandom;

        let mut stack = Vec::new();

        let start = (0, 0);
        self.cell_mut(start.0, start.1).visited = true;
        stack.push(start);

        while let Some(&(x, y)) = stack.last() {
            let mut directions = DIRECTIONS;
            directions.shuffle(rng);

            let mut moved = false;

            for direction in directions {
                let Some((nx, ny)) = self.neighbor(x, y, direction) else {
                    continue;
                };

                if self.cell(nx, ny).visited {
                    continue;
                }

                self.remove_wall(x, y, direction);
                self.cell_mut(nx, ny).visited = true;
                stack.push((nx, ny));

                moved = true;
                break;
            }

            if !moved {
                stack.pop();
            }
        }
    }

    pub fn generate(&mut self) {
        let mut rng = rand::rng();
        self.generate_with(&mut rng);
    }

    pub fn render_ascii(&self) -> String {
        let mut output = String::new();

        // Top boundary.
        for x in 0..self.width {
            output.push('+');

            if self.cell(x, 0).north {
                output.push_str("---");
            } else {
                output.push_str("   ");
            }
        }
        output.push_str("+\n");

        for y in 0..self.height {
            // Cell interiors and vertical walls.
            for x in 0..self.width {
                let cell = self.cell(x, y);

                if cell.west {
                    output.push('|');
                } else {
                    output.push(' ');
                }

                output.push_str("   ");
            }

            if self.cell(self.width - 1, y).east {
                output.push('|');
            } else {
                output.push(' ');
            }

            output.push('\n');

            // South walls for this row.
            for x in 0..self.width {
                output.push('+');

                if self.cell(x, y).south {
                    output.push_str("---");
                } else {
                    output.push_str("   ");
                }
            }

            output.push_str("+\n");
        }

        output
    }

    pub fn can_move(&self, position: Position, direction: Direction) -> bool {
        if position.x >= self.width || position.y >= self.height {
            return false;
        }

        if self.cell(position.x, position.y).has_wall(direction) {
            return false;
        }

        self.neighbor(position.x, position.y, direction).is_some()
    }

    pub fn step(&self, position: Position, direction: Direction) -> Option<Position> {
        if !self.can_move(position, direction) {
            return None;
        }

        self.neighbor(position.x, position.y, direction)
            .map(|(x, y)| Position::new(x, y))
    }

    #[cfg(test)]
    pub(crate) fn open_passage_for_test(&mut self, position: Position, direction: Direction) {
        assert!(
            self.remove_wall(position.x, position.y, direction),
            "test attempted to open an invalid passage"
        );
    }

    pub fn shortest_path_length(&self, start: Position, goal: Position) -> Option<usize> {
        if start.x >= self.width
            || start.y >= self.height
            || goal.x >= self.width
            || goal.y >= self.height
        {
            return None;
        }

        let mut distance = vec![None; self.cells.len()];
        let mut queue = VecDeque::new();

        distance[self.index(start.x, start.y)] = Some(0);
        queue.push_back(start);

        while let Some(position) = queue.pop_front() {
            if position == goal {
                return distance[self.index(position.x, position.y)];
            }

            let current_distance = distance[self.index(position.x, position.y)]
                .expect("queued position must have a distance");

            for direction in DIRECTIONS {
                let Some(next) = self.step(position, direction) else {
                    continue;
                };

                let next_index = self.index(next.x, next.y);

                if distance[next_index].is_some() {
                    continue;
                }

                distance[next_index] = Some(current_distance + 1);
                queue.push_back(next);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn same_seed_generates_same_maze() {
        let mut maze_a = Maze::new(10, 10);
        let mut maze_b = Maze::new(10, 10);

        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);

        maze_a.generate_with(&mut rng_a);
        maze_b.generate_with(&mut rng_b);

        for y in 0..maze_a.height {
            for x in 0..maze_a.width {
                let a = maze_a.cell(x, y);
                let b = maze_b.cell(x, y);

                assert_eq!(a.north, b.north);
                assert_eq!(a.east, b.east);
                assert_eq!(a.south, b.south);
                assert_eq!(a.west, b.west);
            }
        }
    }

    #[test]
    fn new_maze_has_all_walls_and_no_visited_cells() {
        let maze = Maze::new(4, 3);

        assert_eq!(maze.cells.len(), 12);

        for cell in &maze.cells {
            assert!(cell.north);
            assert!(cell.east);
            assert!(cell.south);
            assert!(cell.west);
            assert!(!cell.visited);
        }
    }

    #[test]
    fn neighbor_respects_boundaries() {
        let maze = Maze::new(3, 3);

        assert_eq!(maze.neighbor(1, 1, Direction::North), Some((1, 0)));
        assert_eq!(maze.neighbor(1, 1, Direction::East), Some((2, 1)));
        assert_eq!(maze.neighbor(1, 1, Direction::South), Some((1, 2)));
        assert_eq!(maze.neighbor(1, 1, Direction::West), Some((0, 1)));

        assert_eq!(maze.neighbor(0, 0, Direction::North), None);
        assert_eq!(maze.neighbor(0, 0, Direction::West), None);
        assert_eq!(maze.neighbor(2, 2, Direction::East), None);
        assert_eq!(maze.neighbor(2, 2, Direction::South), None);
    }

    #[test]
    fn remove_wall_updates_both_cells() {
        let mut maze = Maze::new(3, 3);

        assert!(maze.remove_wall(1, 1, Direction::East));

        assert!(!maze.cell(1, 1).east);
        assert!(!maze.cell(2, 1).west);
    }

    #[test]
    fn remove_wall_rejects_boundary() {
        let mut maze = Maze::new(3, 3);

        assert!(!maze.remove_wall(0, 0, Direction::North));
        assert!(!maze.remove_wall(0, 0, Direction::West));

        assert!(maze.cell(0, 0).north);
        assert!(maze.cell(0, 0).west);
    }

    #[test]
    fn generated_maze_visits_every_cell() {
        let mut maze = Maze::new(10, 10);

        maze.generate();

        assert!(maze.cells.iter().all(|cell| cell.visited));
    }

    #[test]
    fn generated_maze_has_symmetric_passages() {
        let mut maze = Maze::new(10, 10);

        maze.generate();

        for y in 0..maze.height {
            for x in 0..maze.width {
                let cell = maze.cell(x, y);

                if x + 1 < maze.width {
                    assert_eq!(
                        cell.east,
                        maze.cell(x + 1, y).west,
                        "east/west mismatch at ({x},{y})"
                    );
                }

                if y + 1 < maze.height {
                    assert_eq!(
                        cell.south,
                        maze.cell(x, y + 1).north,
                        "south/north mismatch at ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn generated_maze_keeps_boundary_walls_closed() {
        let mut maze = Maze::new(10, 10);

        maze.generate();

        for x in 0..maze.width {
            assert!(maze.cell(x, 0).north);
            assert!(maze.cell(x, maze.height - 1).south);
        }

        for y in 0..maze.height {
            assert!(maze.cell(0, y).west);
            assert!(maze.cell(maze.width - 1, y).east);
        }
    }

    #[test]
    fn generated_maze_has_exactly_n_minus_one_passages() {
        let mut maze = Maze::new(10, 10);

        maze.generate();

        let mut passages = 0;

        for y in 0..maze.height {
            for x in 0..maze.width {
                let cell = maze.cell(x, y);

                if x + 1 < maze.width && !cell.east {
                    passages += 1;
                }

                if y + 1 < maze.height && !cell.south {
                    passages += 1;
                }
            }
        }

        assert_eq!(passages, maze.cells.len() - 1);
    }

    #[test]
    fn generated_maze_is_connected() {
        let mut maze = Maze::new(10, 10);
        maze.generate();

        let mut seen = vec![false; maze.cells.len()];
        let mut queue = VecDeque::new();

        seen[maze.index(0, 0)] = true;
        queue.push_back((0, 0));

        while let Some((x, y)) = queue.pop_front() {
            let cell = maze.cell(x, y);

            for direction in DIRECTIONS {
                let open = match direction {
                    Direction::North => !cell.north,
                    Direction::East => !cell.east,
                    Direction::South => !cell.south,
                    Direction::West => !cell.west,
                };

                if !open {
                    continue;
                }

                let Some((nx, ny)) = maze.neighbor(x, y, direction) else {
                    panic!("open passage leads outside maze at ({x},{y})");
                };

                let index = maze.index(nx, ny);

                if !seen[index] {
                    seen[index] = true;
                    queue.push_back((nx, ny));
                }
            }
        }

        assert!(seen.iter().all(|visited| *visited));
    }

    #[test]
    fn ascii_renderer_draws_known_maze() {
        let mut maze = Maze::new(2, 2);

        maze.remove_wall(0, 0, Direction::East);
        maze.remove_wall(1, 0, Direction::South);

        let expected = concat!(
            "+---+---+\n",
            "|       |\n",
            "+---+   +\n",
            "|   |   |\n",
            "+---+---+\n",
        );

        assert_eq!(maze.render_ascii(), expected);
    }

    #[test]
    fn cannot_move_through_wall() {
        let maze = Maze::new(2, 2);
        let start = Position::new(0, 0);

        assert!(!maze.can_move(start, Direction::East));
        assert_eq!(maze.step(start, Direction::East), None);
    }

    #[test]
    fn can_move_through_open_passage() {
        let mut maze = Maze::new(2, 2);
        maze.remove_wall(0, 0, Direction::East);

        let start = Position::new(0, 0);

        assert!(maze.can_move(start, Direction::East));
        assert_eq!(maze.step(start, Direction::East), Some(Position::new(1, 0)));
    }

    #[test]
    fn movement_through_passage_is_bidirectional() {
        let mut maze = Maze::new(2, 2);
        maze.remove_wall(0, 0, Direction::East);

        assert_eq!(
            maze.step(Position::new(0, 0), Direction::East),
            Some(Position::new(1, 0))
        );

        assert_eq!(
            maze.step(Position::new(1, 0), Direction::West),
            Some(Position::new(0, 0))
        );
    }

    #[test]
    fn shortest_path_length_on_known_maze() {
        let mut maze = Maze::new(2, 2);

        maze.remove_wall(0, 0, Direction::East);
        maze.remove_wall(1, 0, Direction::South);

        assert_eq!(
            maze.shortest_path_length(Position::new(0, 0), Position::new(1, 1),),
            Some(2)
        );
    }

    #[test]
    fn shortest_path_from_position_to_itself_is_zero() {
        let maze = Maze::new(2, 2);

        let position = Position::new(1, 1);

        assert_eq!(maze.shortest_path_length(position, position), Some(0));
    }

    #[test]
    fn shortest_path_returns_none_for_unreachable_goal() {
        let maze = Maze::new(2, 2);

        assert_eq!(
            maze.shortest_path_length(Position::new(0, 0), Position::new(1, 1),),
            None
        );
    }

    #[test]
    fn public_has_wall_reports_cell_walls() {
        let mut maze = Maze::new(2, 1);

        assert!(maze.has_wall(Position::new(0, 0), Direction::East));

        maze.remove_wall(0, 0, Direction::East);

        assert!(!maze.has_wall(Position::new(0, 0), Direction::East));
    }
}
