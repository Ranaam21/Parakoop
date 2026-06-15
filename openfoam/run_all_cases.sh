#!/bin/bash
# Run all 5 ParaKoop Ahmed body OpenFOAM validation cases via Docker.
# Requires: docker with opencfd/openfoam-default:latest image pulled.
#
# Usage:
#   chmod +x openfoam/run_all_cases.sh
#   ./openfoam/run_all_cases.sh
#
# Logs go to openfoam/cases/<case>/log.*
# Results summary written to results/openfoam_cfd_results.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="${SCRIPT_DIR}/cases"
RESULTS_DIR="${SCRIPT_DIR}/../results"
IMAGE="opencfd/openfoam-default:latest"

mkdir -p "${RESULTS_DIR}"

CASES=(slant10 slant12 slant15 slant17 slant19)

echo "============================================================"
echo "  ParaKoop — OpenFOAM Ahmed Body Validation"
echo "  Image : ${IMAGE}"
echo "  Cases : ${#CASES[@]}"
echo "============================================================"

for CASE in "${CASES[@]}"; do
    CASE_PATH="${CASES_DIR}/${CASE}"
    echo ""
    echo "── Running: ${CASE} ─────────────────────────────────────"
    echo "   Path: ${CASE_PATH}"

    docker run --rm \
        -v "${CASE_PATH}:/case" \
        "${IMAGE}" \
        bash -c "
            cd /case
            echo '[clean]'
            find constant/polyMesh -maxdepth 1 -type f ! -name 'blockMeshDict' -delete 2>/dev/null; echo '  done'
            echo '[blockMesh]'
            blockMesh > log.blockMesh 2>&1 && echo '  done' || { echo '  FAILED'; cat log.blockMesh | tail -5; exit 1; }

            echo '[surfaceFeatureExtract]'
            surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 && echo '  done' || { echo '  FAILED'; cat log.surfaceFeatureExtract | tail -5; exit 1; }

            echo '[snappyHexMesh]'
            snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && echo '  done' || { echo '  FAILED'; cat log.snappyHexMesh | tail -5; exit 1; }

            echo '[simpleFoam]'
            simpleFoam > log.simpleFoam 2>&1 && echo '  done' || { echo '  FAILED'; cat log.simpleFoam | tail -10; exit 1; }

            echo '[extract Cd/Cl]'
            COEFF_FILE=\$(find postProcessing -name 'forceCoeffs.dat' 2>/dev/null | head -1)
            if [ -f \"\${COEFF_FILE}\" ]; then
                echo '  Found:' \"\${COEFF_FILE}\"
                grep -v '^#' \"\${COEFF_FILE}\" | tail -5
            else
                echo '  WARNING: forceCoeffs.dat not found'
            fi
        "

    echo "   Case ${CASE} complete."
done

echo ""
echo "============================================================"
echo "  All cases done. Parsing results..."
echo "============================================================"

python3 "${SCRIPT_DIR}/../scripts/parse_foam_results.py" \
    --cases-dir "${CASES_DIR}" \
    --out "${RESULTS_DIR}/openfoam_cfd_results.csv"
