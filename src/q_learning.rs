use rand::{Rng, RngExt};

use crate::agent::Agent;
use crate::maze::{DIRECTIONS, Direction, Position};

#[derive(Debug, Clone)]
pub struct QTable {
    width: usize,
    values: Vec<[f32; 4]>,
}

impl QTable {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            values: vec![[0.0; 4]; width * height],
        }
    }

    pub fn get(&self, position: Position, direction: Direction) -> f32 {
        self.values[self.index(position)][Self::action_index(direction)]
    }

    pub fn set(&mut self, position: Position, direction: Direction, value: f32) {
        let state = self.index(position);
        let action = Self::action_index(direction);

        self.values[state][action] = value;
    }

    fn index(&self, position: Position) -> usize {
        position.y * self.width + position.x
    }

    fn action_index(direction: Direction) -> usize {
        match direction {
            Direction::North => 0,
            Direction::East => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }

    pub fn max_value(&self, position: Position) -> f32 {
        self.values[self.index(position)]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn update(
        &mut self,
        state: Position,
        action: Direction,
        reward: f32,
        next_state: Position,
        terminated: bool,
        alpha: f32,
        gamma: f32,
    ) {
        let current = self.get(state, action);

        let future = if terminated {
            0.0
        } else {
            self.max_value(next_state)
        };

        let target = reward + gamma * future;

        let updated = current + alpha * (target - current);

        self.set(state, action, updated);
    }
}

#[derive(Debug, Clone)]
pub struct QLearningAgent {
    table: QTable,
    epsilon: f32,
    alpha: f32,
    gamma: f32,
}

impl QLearningAgent {
    pub fn new(width: usize, height: usize, epsilon: f32, alpha: f32, gamma: f32) -> Self {
        assert!((0.0..=1.0).contains(&epsilon));
        assert!((0.0..=1.0).contains(&alpha));
        assert!((0.0..=1.0).contains(&gamma));

        Self {
            table: QTable::new(width, height),
            epsilon,
            alpha,
            gamma,
        }
    }

    pub fn learn(
        &mut self,
        state: Position,
        action: Direction,
        reward: f32,
        next_state: Position,
        terminated: bool,
    ) {
        self.table.update(
            state, action, reward, next_state, terminated, self.alpha, self.gamma,
        );
    }

    pub fn q_value(&self, position: Position, direction: Direction) -> f32 {
        self.table.get(position, direction)
    }

    fn best_action<R: Rng + ?Sized>(&self, position: Position, rng: &mut R) -> Direction {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_actions = Vec::new();

        for direction in DIRECTIONS {
            let value = self.table.get(position, direction);

            if value > best_value {
                best_value = value;
                best_actions.clear();
                best_actions.push(direction);
            } else if value == best_value {
                best_actions.push(direction);
            }
        }

        best_actions[rng.random_range(0..best_actions.len())]
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        assert!((0.0..=1.0).contains(&epsilon));
        self.epsilon = epsilon;
    }

    pub fn best_action_for(&self, position: Position) -> Direction {
        let mut best_direction = Direction::North;
        let mut best_value = self.q_value(position, best_direction);

        for direction in [Direction::East, Direction::South, Direction::West] {
            let value = self.q_value(position, direction);

            if value > best_value {
                best_value = value;
                best_direction = direction;
            }
        }

        best_direction
    }

    pub fn has_learned_value(&self, position: Position) -> bool {
        DIRECTIONS
            .iter()
            .any(|&direction| self.q_value(position, direction) != 0.0)
    }
}

impl Agent for QLearningAgent {
    fn choose_action<R: Rng + ?Sized>(&mut self, position: Position, rng: &mut R) -> Direction {
        if rng.random::<f32>() < self.epsilon {
            return DIRECTIONS[rng.random_range(0..DIRECTIONS.len())];
        }

        self.best_action(position, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn new_q_table_initializes_all_values_to_zero() {
        let table = QTable::new(3, 2);

        for y in 0..2 {
            for x in 0..3 {
                let position = Position::new(x, y);

                for direction in [
                    Direction::North,
                    Direction::East,
                    Direction::South,
                    Direction::West,
                ] {
                    assert_eq!(table.get(position, direction), 0.0);
                }
            }
        }
    }

    #[test]
    fn setting_one_q_value_does_not_change_others() {
        let mut table = QTable::new(3, 2);

        let position = Position::new(1, 0);

        table.set(position, Direction::East, 7.5);

        assert_eq!(table.get(position, Direction::East), 7.5);

        assert_eq!(table.get(position, Direction::North), 0.0);

        assert_eq!(table.get(Position::new(2, 0), Direction::East), 0.0);
    }

    #[test]
    fn max_value_returns_largest_action_value() {
        let mut table = QTable::new(2, 2);
        let position = Position::new(0, 0);

        table.set(position, Direction::North, -2.0);
        table.set(position, Direction::East, 4.0);
        table.set(position, Direction::South, 1.5);
        table.set(position, Direction::West, -1.0);

        assert_eq!(table.max_value(position), 4.0);
    }

    #[test]
    fn q_update_uses_reward_and_future_value() {
        let mut table = QTable::new(2, 1);

        let state = Position::new(0, 0);
        let next_state = Position::new(1, 0);

        table.set(state, Direction::East, 2.0);

        table.set(next_state, Direction::North, 4.0);
        table.set(next_state, Direction::East, 10.0);
        table.set(next_state, Direction::South, 3.0);
        table.set(next_state, Direction::West, 1.0);

        table.update(state, Direction::East, -1.0, next_state, false, 0.5, 0.9);

        assert_eq!(table.get(state, Direction::East), 5.0);
    }

    #[test]
    fn terminal_q_update_does_not_bootstrap_future_reward() {
        let mut table = QTable::new(2, 1);

        let state = Position::new(0, 0);
        let goal = Position::new(1, 0);

        table.set(state, Direction::East, 20.0);

        // Deliberately huge values at the goal. They must be ignored.
        table.set(goal, Direction::North, 1_000.0);
        table.set(goal, Direction::East, 1_000.0);
        table.set(goal, Direction::South, 1_000.0);
        table.set(goal, Direction::West, 1_000.0);

        table.update(state, Direction::East, 100.0, goal, true, 0.5, 0.9);

        assert_eq!(table.get(state, Direction::East), 60.0);
    }

    #[test]
    fn greedy_agent_selects_highest_q_value() {
        let mut agent = QLearningAgent::new(2, 2, 0.0, 0.5, 0.9);

        let position = Position::new(0, 0);

        agent.table.set(position, Direction::North, -1.0);
        agent.table.set(position, Direction::East, 8.0);
        agent.table.set(position, Direction::South, 3.0);
        agent.table.set(position, Direction::West, 2.0);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        assert_eq!(agent.choose_action(position, &mut rng), Direction::East);
    }

    #[test]
    fn learning_updates_agents_q_value() {
        let mut agent = QLearningAgent::new(2, 1, 0.1, 0.5, 0.9);

        let state = Position::new(0, 0);
        let next_state = Position::new(1, 0);

        agent.learn(state, Direction::East, 100.0, next_state, true);

        assert_eq!(agent.q_value(state, Direction::East), 50.0);
    }

    #[test]
    fn greedy_agent_can_choose_different_actions_when_q_values_tie() {
        let mut agent = QLearningAgent::new(1, 1, 0.0, 0.5, 0.9);

        let position = Position::new(0, 0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut seen = std::collections::HashSet::new();

        for _ in 0..100 {
            seen.insert(agent.choose_action(position, &mut rng));
        }

        assert!(
            seen.len() > 1,
            "tie-breaking should not always choose the same direction"
        );
    }

    #[test]
    fn epsilon_can_be_changed() {
        let mut agent = QLearningAgent::new(2, 2, 0.2, 0.5, 0.9);

        agent.set_epsilon(0.5);

        assert_eq!(agent.epsilon, 0.5);
    }

    #[test]
    fn best_action_for_returns_highest_q_value() {
        let mut agent = QLearningAgent::new(2, 2, 0.0, 0.5, 0.9);

        let position = Position::new(0, 0);

        agent.table.set(position, Direction::North, -2.0);
        agent.table.set(position, Direction::East, 7.0);
        agent.table.set(position, Direction::South, 3.0);
        agent.table.set(position, Direction::West, 1.0);

        assert_eq!(agent.best_action_for(position), Direction::East);
    }

    #[test]
    fn has_learned_value_is_false_for_new_state() {
        let agent = QLearningAgent::new(2, 2, 0.2, 0.5, 0.9);

        assert!(!agent.has_learned_value(Position::new(0, 0)));
    }

    #[test]
    fn has_learned_value_is_true_after_update() {
        let mut agent = QLearningAgent::new(2, 2, 0.2, 0.5, 0.9);

        agent.learn(
            Position::new(0, 0),
            Direction::East,
            -5.0,
            Position::new(0, 0),
            false,
        );

        assert!(agent.has_learned_value(Position::new(0, 0)));
    }
}
