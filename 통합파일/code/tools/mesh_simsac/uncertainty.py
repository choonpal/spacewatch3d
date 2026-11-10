"""Combine whole-cloud residuals and held-out regional structural error."""
import math


def object_uncertainty(registration, regions, voxel):
    errors = [voxel/2]
    errors.extend(v for v in registration.get('median_inlier_distances_u', []) if v is not None)
    errors.extend(v for v in registration.get('pointcloud_residual_p90_u', []) if v is not None)
    supported = registration.get('state') == 'passed_geometric_gate'
    structural = registration.get('fresh_structural')
    if structural:
        for region in regions or []:
            record = structural['regions'][int(region)-1]
            supported &= bool(record['supported'])
            value = record['validation'].get('tolerance_u')
            if value is not None:
                errors.append(value)
            else:
                supported = False
    return max(v for v in errors if math.isfinite(v)), bool(supported)
