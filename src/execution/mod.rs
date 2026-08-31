//! Order execution simulation for RaptorBT.

pub mod algos;
pub mod fees;
pub mod fill;
pub mod indian_costs;
pub mod orders;
pub mod queue;
pub mod slippage;

pub use algos::{AlgoEngine, AlgoError, AlgoSchedule, ExecAlgorithm, PendingSlice};
pub use fees::FeeModel;
pub use fill::{
    BarLiquidity, BarTape, FillDepth, FillModel, FillPrice, NextPrint, StepKind, Tail, Tape,
};
pub use indian_costs::{calculate_side, CostSchedule, FeeBreakdown, Segment};
pub use queue::{QueueTracker, QueueVerdict};
pub use slippage::SlippageModel;
