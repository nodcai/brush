//! Per-pose, per-splat (position, rotation) deltas applied on top of the
//! canonical (rest-pose) splat parameters during posed training.
//!
//! File format (little-endian, packed):
//!
//! ```text
//! [24-byte header]
//!   magic        : char[8]  = "MMPDELTA"
//!   version      : u32      = 1
//!   num_poses    : u32
//!   num_splats   : u32
//!   world_scale  : f32      (informational)
//! [body]
//!   num_poses × num_splats × 7 f32 = (dx, dy, dz, qw, qx, qy, qz)
//! ```
//!
//! Semantics:
//!
//! ```text
//! posed_pos = canonical_pos + (dx, dy, dz)
//! posed_rot = quat_mul((qw, qx, qy, qz), canonical_rot)
//! ```
//!
//! Pose 0 is expected to be neutral (zero translation, identity quaternion
//! up to FP noise), so the canonical splat renders unchanged for that
//! frame block. Higher poses define the deformation deltas applied during
//! posed training.

use std::path::Path;
use std::sync::Arc;

use brush_vfs::BrushVfs;
use tokio::io::AsyncReadExt;

#[derive(Clone, Debug)]
pub struct PoseDeltas {
    pub num_poses: u32,
    pub num_splats: u32,
    /// Flat buffer of `num_poses * num_splats * 7` floats. Row-major:
    /// `pose × num_splats × 7`, splat-major within a pose, lane order
    /// `(dx, dy, dz, qw, qx, qy, qz)`.
    pub deltas: Vec<f32>,
}

impl PoseDeltas {
    /// Load a pose deltas file (MMPDELTA) into memory. Errors are returned
    /// as `String` to avoid pulling in an error crate; the dataset loader
    /// wraps them in `FormatError::InvalidFormat`.
    pub async fn load(vfs: Arc<BrushVfs>, path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        vfs.reader_at_path(path)
            .await
            .map_err(|e| format!("opening pose deltas {path:?}: {e}"))?
            .read_to_end(&mut bytes)
            .await
            .map_err(|e| format!("reading pose deltas {path:?}: {e}"))?;

        if bytes.len() < 24 {
            return Err(format!(
                "pose deltas file too small ({} bytes)",
                bytes.len()
            ));
        }
        if &bytes[0..8] != b"MMPDELTA" {
            return Err("pose deltas magic mismatch (expected MMPDELTA)".to_owned());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported pose deltas version: {version}"));
        }
        let num_poses = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let num_splats = u32::from_le_bytes(bytes[16..20].try_into().unwrap());

        let expected_body = num_poses as usize * num_splats as usize * 7 * 4;
        let body = &bytes[24..];
        if body.len() != expected_body {
            return Err(format!(
                "pose deltas body size mismatch: got {}, expected {}",
                body.len(),
                expected_body
            ));
        }
        let deltas = body
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Self {
            num_poses,
            num_splats,
            deltas,
        })
    }
}
