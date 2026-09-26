use rand::{Rng, RngExt};

use crate::maze::{DIRECTIONS, Direction, Position};

pub trait Agent {
    fn choose_action<R: Rng + ?Sized>(&mut self, position: Position, rng: &mut R) -> Direction;
}

#[derive(Debug, Default)]
pub struct RandomAgent;

impl Agent for RandomAgent {
    fn choose_action<R: Rng + ?Sized>(&mut self, _position: Position, rng: &mut R) -> Direction {
        DIRECTIONS[rng.random_range(0..DIRECTIONS.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn random_agent_returns_valid_direction() {
        let mut agent = RandomAgent;
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let direction = agent.choose_action(Position::new(0, 0), &mut rng);

            assert!(matches!(
                direction,
                Direction::North | Direction::East | Direction::South | Direction::West
            ));
        }
    }

    #[test]
    fn random_agent_is_reproducible_with_same_seed() {
        let mut agent_a = RandomAgent;
        let mut agent_b = RandomAgent;

        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);

        let position = Position::new(0, 0);

        for _ in 0..100 {
            assert_eq!(
                agent_a.choose_action(position, &mut rng_a),
                agent_b.choose_action(position, &mut rng_b)
            );
        }
    }
}
