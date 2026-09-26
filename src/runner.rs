use rand::Rng;
use std::collections::HashSet;

use crate::{
    agent::Agent, environment::Environment, episode::Episode, maze::Position,
    q_learning::QLearningAgent,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunResult {
    pub completed: bool,
    pub actions: usize,
    pub moves: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationResult {
    pub completed: bool,
    pub cycle_detected: bool,
    pub actions: usize,
    pub moves: usize,
}

pub fn run_episode<R, A>(
    episode: &mut Episode,
    environment: &Environment,
    agent: &mut A,
    rng: &mut R,
    max_actions: usize,
) -> RunResult
where
    R: Rng + ?Sized,
    A: Agent,
{
    while !episode.is_complete() && episode.actions() < max_actions {
        let action = agent.choose_action(episode.position(), rng);
        episode.step(environment, action);
    }

    RunResult {
        completed: episode.is_complete(),
        actions: episode.actions(),
        moves: episode.moves(),
    }
}

pub fn train<R>(
    environment: &Environment,
    start: Position,
    agent: &mut QLearningAgent,
    rng: &mut R,
    episodes: usize,
    max_actions_per_episode: usize,
) -> Vec<RunResult>
where
    R: Rng + ?Sized,
{
    let mut results = Vec::with_capacity(episodes);

    for _ in 0..episodes {
        let mut episode = Episode::new(start, environment);

        let result = train_episode(
            &mut episode,
            environment,
            agent,
            rng,
            max_actions_per_episode,
        );

        results.push(result);
    }

    results
}

pub fn train_episode<R>(
    episode: &mut Episode,
    environment: &Environment,
    agent: &mut QLearningAgent,
    rng: &mut R,
    max_actions: usize,
) -> RunResult
where
    R: Rng + ?Sized,
{
    while !episode.is_complete() && episode.actions() < max_actions {
        let state = episode.position();

        let action = agent.choose_action(state, rng);

        let result = environment.act(state, action);

        agent.learn(
            state,
            action,
            result.reward,
            result.position,
            result.terminated,
        );

        episode.apply_result(result);
    }

    RunResult {
        completed: episode.is_complete(),
        actions: episode.actions(),
        moves: episode.moves(),
    }
}

pub fn evaluate_episode<R, A>(
    episode: &mut Episode,
    environment: &Environment,
    agent: &mut A,
    rng: &mut R,
    max_actions: usize,
) -> EvaluationResult
where
    R: Rng + ?Sized,
    A: Agent,
{
    let mut visited = HashSet::new();

    while !episode.is_complete() && episode.actions() < max_actions {
        let position = episode.position();

        if !visited.insert(position) {
            return EvaluationResult {
                completed: false,
                cycle_detected: true,
                actions: episode.actions(),
                moves: episode.moves(),
            };
        }

        let action = agent.choose_action(position, rng);
        episode.step(environment, action);
    }

    EvaluationResult {
        completed: episode.is_complete(),
        cycle_detected: false,
        actions: episode.actions(),
        moves: episode.moves(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::{
        agent::{Agent, RandomAgent},
        maze::{Direction, Maze, Position},
        q_learning::QLearningAgent,
    };

    #[derive(Debug)]
    struct ScriptedAgent {
        actions: Vec<Direction>,
        next: usize,
    }

    impl ScriptedAgent {
        fn new(actions: Vec<Direction>) -> Self {
            Self { actions, next: 0 }
        }
    }

    #[derive(Debug)]
    struct BackAndForthAgent;

    impl Agent for BackAndForthAgent {
        fn choose_action<R: rand::Rng + ?Sized>(
            &mut self,
            position: Position,
            _rng: &mut R,
        ) -> Direction {
            if position.x == 0 {
                Direction::East
            } else {
                Direction::West
            }
        }
    }

    impl Agent for ScriptedAgent {
        fn choose_action<R: rand::Rng + ?Sized>(
            &mut self,
            _position: Position,
            _rng: &mut R,
        ) -> Direction {
            let action = self.actions[self.next];
            self.next += 1;
            action
        }
    }

    #[test]
    fn runner_stops_at_action_limit() {
        let maze = Maze::new(2, 2);
        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let mut agent = RandomAgent;
        let mut rng = StdRng::seed_from_u64(42);

        let result = run_episode(&mut episode, &environment, &mut agent, &mut rng, 10);

        assert!(!result.completed);
        assert_eq!(result.actions, 10);
        assert_eq!(result.moves, 0);
    }

    #[test]
    fn runner_reports_already_complete_episode() {
        let goal = Position::new(0, 0);

        let maze = Maze::new(1, 1);
        let environment = Environment::new(maze, goal);

        let mut episode = Episode::new(goal, &environment);

        let mut agent = RandomAgent;
        let mut rng = StdRng::seed_from_u64(42);

        let result = run_episode(&mut episode, &environment, &mut agent, &mut rng, 100);

        assert!(result.completed);
        assert_eq!(result.actions, 0);
        assert_eq!(result.moves, 0);
    }

    #[test]
    fn runner_completes_known_path_end_to_end() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        maze.open_passage_for_test(Position::new(1, 0), Direction::South);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let mut agent = ScriptedAgent::new(vec![Direction::East, Direction::South]);

        let mut rng = StdRng::seed_from_u64(42);

        let result = run_episode(&mut episode, &environment, &mut agent, &mut rng, 10);

        assert!(result.completed);
        assert_eq!(result.actions, 2);
        assert_eq!(result.moves, 2);

        assert_eq!(episode.position(), Position::new(1, 1));

        assert!(episode.is_complete());
    }

    #[test]
    fn training_episode_learns_from_goal_reward() {
        let mut maze = Maze::new(2, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(1, 0));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let mut agent = QLearningAgent::new(2, 1, 0.0, 0.5, 0.9);

        let mut rng = StdRng::seed_from_u64(42);

        // Bias the only useful action so the greedy agent takes East.
        agent.learn(
            Position::new(0, 0),
            Direction::East,
            1.0,
            Position::new(0, 0),
            true,
        );

        let result = train_episode(&mut episode, &environment, &mut agent, &mut rng, 10);

        assert!(result.completed);

        assert!(agent.q_value(Position::new(0, 0), Direction::East) > 1.0);
    }

    #[test]
    fn repeated_training_improves_on_simple_maze() {
        let mut maze = Maze::new(3, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        maze.open_passage_for_test(Position::new(1, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(2, 0));

        let mut agent = QLearningAgent::new(3, 1, 0.2, 0.5, 0.9);

        let mut rng = StdRng::seed_from_u64(42);

        let results = train(
            &environment,
            Position::new(0, 0),
            &mut agent,
            &mut rng,
            100,
            100,
        );

        assert!(results.iter().any(|result| result.completed));

        let last = results.last().unwrap();

        assert!(last.completed);
        assert!(last.actions <= 10);
    }

    #[test]
    fn evaluation_completes_known_path_without_cycle() {
        let mut maze = Maze::new(2, 2);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        maze.open_passage_for_test(Position::new(1, 0), Direction::South);

        let environment = Environment::new(maze, Position::new(1, 1));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let mut agent = ScriptedAgent::new(vec![Direction::East, Direction::South]);

        let mut rng = StdRng::seed_from_u64(42);

        let result = evaluate_episode(&mut episode, &environment, &mut agent, &mut rng, 10);

        assert!(result.completed);
        assert!(!result.cycle_detected);
        assert_eq!(result.actions, 2);
        assert_eq!(result.moves, 2);
    }

    #[test]
    fn evaluation_detects_policy_cycle() {
        let mut maze = Maze::new(3, 1);

        maze.open_passage_for_test(Position::new(0, 0), Direction::East);

        maze.open_passage_for_test(Position::new(1, 0), Direction::East);

        let environment = Environment::new(maze, Position::new(2, 0));

        let mut episode = Episode::new(Position::new(0, 0), &environment);

        let mut agent = BackAndForthAgent;

        let mut rng = StdRng::seed_from_u64(42);

        let result = evaluate_episode(&mut episode, &environment, &mut agent, &mut rng, 100);

        assert!(!result.completed);
        assert!(result.cycle_detected);
        assert_eq!(result.actions, 2);
        assert_eq!(result.moves, 2);
    }
}
