pub mod generated;
pub mod kafka;
pub mod metrics;
pub mod observability;
pub mod processor;
pub mod retry_consumer;
pub mod trace_context;
mod wire;

pub use jagua_utils::{PageResult, PlacedPartInfo};
pub use kafka::{KafkaSettings, OffsetWatermark};
// The DTO names keep their `Sqs` prefix deliberately. They are wire-contract types
// generated against the shared AsyncAPI spec and used across ~3k lines of tests;
// cutl-backend kept `SqsService` through its own Kafka port for the same reason.
// Only the transport itself was renamed.
pub use processor::{
    NestingProcessor, SqsNestingRequest, SqsNestingResponse, SvgDownloader, SvgPartSpec,
};
