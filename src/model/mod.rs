//! The model kernels, one module per stage of SPEC sections 2 to 8.
//!
//! Every kernel here is per gene. Genes are independent under this model, so
//! there is no cross-gene coupling and no global iteration anywhere below.

pub(crate) mod empty_cells;
pub(crate) mod fractions;
pub(crate) mod gene;
pub(crate) mod likelihood;
pub(crate) mod variance;
