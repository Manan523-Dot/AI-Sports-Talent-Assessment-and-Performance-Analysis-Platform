import os
import json
import pandas as pd


def analyze_vertical_jump(input_csv, output_json):
    """
    Analyze vertical jump using smoothed knee-angle movement patterns.

    Detects candidate movement cycles instead of relying exclusively on
    the single deepest knee bend in the entire video.

    Knee-angle-based flight detection is approximate. It cannot directly
    confirm that the feet leave or contact the ground.
    """

    print("\n========== VERTICAL JUMP ANALYSIS ==========")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = [
        "frame",
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing from CSV."
            )

    df = df.copy()

    for column in [
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(
            subset=[
                "time_seconds",
                "left_knee_angle",
                "right_knee_angle",
            ]
        )
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            "Not enough valid pose frames for vertical jump analysis."
        )

    # -----------------------------------------
    # 1. Average knee angle
    # -----------------------------------------

    df["knee_angle"] = (
        df["left_knee_angle"] + df["right_knee_angle"]
    ) / 2.0

    # Median smoothing reduces sudden pose-estimation noise.
    df["knee_angle_smooth"] = (
        df["knee_angle"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    times = df["time_seconds"].tolist()
    angles = df["knee_angle_smooth"].tolist()

    max_knee_angle = float(max(angles))
    min_knee_angle = float(min(angles))

    knee_range_of_motion = (
        max_knee_angle - min_knee_angle
    )

    # -----------------------------------------
    # 2. Identify frame-time gaps
    # -----------------------------------------

    time_diffs = df["time_seconds"].diff().dropna()

    median_frame_interval = (
        float(time_diffs.median())
        if len(time_diffs) > 0
        else 0.038
    )

    # A large gap should not be treated as continuous movement.
    max_continuous_gap = max(
        0.12,
        median_frame_interval * 2.5
    )

    def has_large_gap(start_index, end_index):
        """Check whether movement crosses missing video frames."""

        if end_index <= start_index:
            return False

        for index in range(start_index + 1, end_index + 1):
            gap = times[index] - times[index - 1]

            if gap > max_continuous_gap:
                return True

        return False

    # -----------------------------------------
    # 3. Find local knee-bend candidates
    # -----------------------------------------

    bottom_candidates = []

    for index in range(1, len(angles) - 1):

        previous_angle = angles[index - 1]
        current_angle = angles[index]
        next_angle = angles[index + 1]

        # A local minimum indicates a possible crouch/bend.
        if (
            current_angle <= previous_angle
            and current_angle <= next_angle
        ):
            bottom_candidates.append(index)

    # Include the global minimum as a fallback.
    global_bottom_index = angles.index(min(angles))

    if global_bottom_index not in bottom_candidates:
        bottom_candidates.append(global_bottom_index)

    bottom_candidates = sorted(
        set(bottom_candidates),
        key=lambda index: angles[index]
    )

    bottom_index = global_bottom_index
    bottom_time = times[bottom_index]
    bottom_angle = angles[bottom_index]

    # -----------------------------------------
    # 4. Detect movement cycles
    # -----------------------------------------

    standing_threshold = 160.0
    sustain_frames = 3

    flexion_drop_threshold = 15.0
    landing_sustain_frames = 2

    min_valid_flight_time = 0.12
    max_valid_flight_time = 1.20

    best_candidate = None

    for candidate_bottom in bottom_candidates:

        candidate_bottom_angle = angles[candidate_bottom]
        candidate_bottom_time = times[candidate_bottom]

        # Only consider plausible preceding bends.
        if candidate_bottom_angle > 155:
            continue

        # Search for sustained extension after this bend.
        takeoff_index = None

        for index in range(
            candidate_bottom + 1,
            len(angles) - sustain_frames + 1
        ):

            window = angles[index:index + sustain_frames]

            if all(
                angle >= standing_threshold
                for angle in window
            ):

                # Do not accept extension across missing frames.
                if has_large_gap(
                    candidate_bottom,
                    index + sustain_frames - 1
                ):
                    continue

                takeoff_index = index
                break

        if takeoff_index is None:
            continue

        takeoff_time_candidate = times[takeoff_index]

        # -------------------------------------
        # Find maximum extension after takeoff
        # -------------------------------------

        peak_index = takeoff_index

        for index in range(takeoff_index, len(angles)):

            if has_large_gap(peak_index, index):
                break

            if angles[index] > angles[peak_index]:
                peak_index = index

        peak_angle = angles[peak_index]

        # -------------------------------------
        # Find post-extension flexion
        # -------------------------------------

        landing_index = None

        for index in range(
            peak_index + 1,
            len(angles) - landing_sustain_frames + 1
        ):

            if has_large_gap(peak_index, index):
                break

            if (
                peak_angle - angles[index]
                < flexion_drop_threshold
            ):
                continue

            window = angles[
                index:index + landing_sustain_frames
            ]

            if all(
                peak_angle - angle
                >= flexion_drop_threshold
                for angle in window
            ):

                if has_large_gap(
                    index,
                    index + landing_sustain_frames - 1
                ):
                    continue

                landing_index = index
                break

        if landing_index is None:
            continue

        candidate_landing_time = times[landing_index]

        candidate_flight_time = (
            candidate_landing_time
            - takeoff_time_candidate
        )

        if not (
            min_valid_flight_time
            <= candidate_flight_time
            <= max_valid_flight_time
        ):
            continue

        # Prefer the candidate with the largest knee bend.
        candidate = {
            "bottom_index": candidate_bottom,
            "bottom_time": candidate_bottom_time,
            "bottom_angle": candidate_bottom_angle,
            "takeoff_index": takeoff_index,
            "takeoff_time": takeoff_time_candidate,
            "peak_index": peak_index,
            "peak_angle": peak_angle,
            "landing_index": landing_index,
            "landing_time": candidate_landing_time,
            "flight_time": candidate_flight_time,
        }

        if (
            best_candidate is None
            or candidate_bottom_angle
            < best_candidate["bottom_angle"]
        ):
            best_candidate = candidate

    # -----------------------------------------
    # 5. Use validated candidate if available
    # -----------------------------------------

    takeoff_detected = False
    landing_detected = False
    jump_detected = False

    takeoff_index = None
    landing_index = None

    candidate_flight_time = 0.0
    flight_time = 0.0

    takeoff_time = 0.0
    landing_time = 0.0

    takeoff_knee_angle = 0.0
    landing_knee_angle = 0.0

    estimated_jump_height_cm = 0.0

    if best_candidate is not None:

        bottom_index = best_candidate["bottom_index"]
        bottom_time = best_candidate["bottom_time"]
        bottom_angle = best_candidate["bottom_angle"]

        takeoff_index = best_candidate["takeoff_index"]
        landing_index = best_candidate["landing_index"]

        takeoff_time = best_candidate["takeoff_time"]
        landing_time = best_candidate["landing_time"]

        candidate_flight_time = (
            best_candidate["flight_time"]
        )

        takeoff_detected = True
        landing_detected = True
        jump_detected = True

        flight_time = candidate_flight_time

        takeoff_knee_angle = angles[takeoff_index]
        landing_knee_angle = angles[landing_index]

        # Physics-based estimate assuming symmetrical flight.
        gravity = 9.81

        estimated_jump_height_cm = (
            gravity
            * flight_time ** 2
            / 8.0
            * 100.0
        )

    # -----------------------------------------
    # 6. Symmetry measurements
    # -----------------------------------------

    left_knee_rom = (
        float(df["left_knee_angle"].max())
        - float(df["left_knee_angle"].min())
    )

    right_knee_rom = (
        float(df["right_knee_angle"].max())
        - float(df["right_knee_angle"].min())
    )

    knee_rom_difference = abs(
        left_knee_rom - right_knee_rom
    )

    # -----------------------------------------
    # 7. Graph data
    # -----------------------------------------

    knee_angle_data = [
        {
            "time": float(row["time_seconds"]),
            "knee_angle": float(row["knee_angle_smooth"]),
        }
        for _, row in df.iterrows()
    ]

    # -----------------------------------------
    # 8. Detection status
    # -----------------------------------------

    if jump_detected:

        detection_status = "candidate_events_detected"

        detection_note = (
            "A candidate jump movement cycle was detected "
            "using sustained knee extension and subsequent "
            "flexion. Flight events are approximate and "
            "feet-off/feet-on are not directly confirmed."
        )

    else:

        detection_status = "unvalidated"

        # Give a useful reason when pose frames are missing.
        largest_gap = (
            float(time_diffs.max())
            if len(time_diffs) > 0
            else 0.0
        )

        if largest_gap > max_continuous_gap:

            detection_note = (
                "Movement could not be validated because "
                "the pose data contains large time gaps. "
                "A reliable takeoff-to-landing interval "
                "could not be confirmed."
            )

        else:

            detection_note = (
                "A complete crouch-extension-flexion cycle "
                "could not be confirmed from knee angles. "
                "Jump height is withheld."
            )

    # -----------------------------------------
    # 9. Output features
    # -----------------------------------------

    features = {
        "activity": "vertical_jump",
        "jump_detected": bool(jump_detected),

        "detection_status": detection_status,
        "detection_note": detection_note,

        "takeoff_detected": bool(takeoff_detected),
        "landing_detected": bool(landing_detected),

        "candidate_flight_time_seconds": round(
            candidate_flight_time, 3
        ),

        "max_knee_angle": round(
            max_knee_angle, 2
        ),

        "min_knee_angle": round(
            min_knee_angle, 2
        ),

        "knee_range_of_motion": round(
            knee_range_of_motion, 2
        ),

        "bottom_time": round(
            bottom_time, 3
        ),

        "bottom_knee_angle": round(
            bottom_angle, 2
        ),

        "takeoff_time": round(
            takeoff_time, 3
        ),

        "landing_time": round(
            landing_time, 3
        ),

        "flight_time": round(
            flight_time, 3
        ),

        "estimated_jump_height_cm": round(
            estimated_jump_height_cm, 2
        ),

        "takeoff_knee_angle": round(
            takeoff_knee_angle, 2
        ),

        "landing_knee_angle": round(
            landing_knee_angle, 2
        ),

        "left_knee_rom": round(
            left_knee_rom, 2
        ),

        "right_knee_rom": round(
            right_knee_rom, 2
        ),

        "knee_rom_difference": round(
            knee_rom_difference, 2
        ),

        "knee_angle_data": knee_angle_data,
    }

    # -----------------------------------------
    # 10. Save JSON
    # -----------------------------------------

    output_directory = os.path.dirname(output_json)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            features,
            file,
            indent=4
        )

    # -----------------------------------------
    # 11. Console output
    # -----------------------------------------

    print(
        f"Jump detected/validated: "
        f"{features['jump_detected']}"
    )

    print(
        f"Takeoff detected: "
        f"{features['takeoff_detected']}"
    )

    print(
        f"Landing detected: "
        f"{features['landing_detected']}"
    )

    print(
        f"Candidate flight time: "
        f"{features['candidate_flight_time_seconds']} s"
    )

    print(
        f"Validated flight time: "
        f"{features['flight_time']} s"
    )

    print(
        f"Estimated jump height: "
        f"{features['estimated_jump_height_cm']} cm"
    )

    print(
        f"Detection status: "
        f"{features['detection_status']}"
    )

    print(
        f"Detection note: "
        f"{features['detection_note']}"
    )

    print(
        f"Feature file saved to: "
        f"{output_json}"
    )

    print("============================================\n")

    return features