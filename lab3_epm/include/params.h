#pragma once
// Fixed constants. Read-only. Do not change!
constexpr float EPM_GRID_SPACING = 0.1f;
constexpr float EPM_SLICE_Z = 0.0f;
// Generator keeps atoms at least this far from the slice (no singularities).
constexpr float EPM_MIN_Z_DISTANCE = 0.25f;
// Relative L2 tolerance against the double-precision CPU reference.
constexpr double EPM_TOL_REL = 1e-4;
