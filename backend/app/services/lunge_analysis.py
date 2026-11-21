
import json
import os

import numpy as np
import pandas as pd


def smooth_series(series, window=5):
    return series.rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()


def detect_lunge_repetitions(df, knee_column):
    angles = smooth_series(df[knee_column]).to_numpy()
    times = df["time_seconds"].to_numpy()

    repetitions = []
    active = False
    start_index = None
    bottom_index = None

    for i, angle in enumerate(angles):

        if np.isnan(angle):
            continue

        if not active and angle < 140:
            active = True
            start_index = i
            bottom_index = i

        elif active:

            if angle < angles[bottom_index]:
                bottom_index = i

            if angle > 145:

                duration = times[i] - times[start_index]

                if duration >= 0.5:

                    repetitions.append({
                        "rep": len(repetitions) + 1,
                        "start_time": round(
                            float(times[start_index]), 3
                        ),
                        "bottom_time": round(
                            float(times[bottom_index]), 3
                        ),
                        "end_time": round(float(times[i]), 3),
                        "duration": round(float(duration), 3),
                        "bottom_knee_angle": round(
                            float(angles[bottom_index]), 2
                        ),
                        "knee_rom": round(
                            float(angles[start_index] -
                                  angles[bottom_index]), 2
                        )
                    })

                active = False
                start_index = None
                bottom_index = None

    return repetitions, angles


def analyze_lunge(input_csv, output_json):

    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Lunge CSV is empty.")

    left_coverage = (
        df["left_knee_visibility"].mean() * 100
    )

    right_coverage = (
        df["right_knee_visibility"].mean() * 100
    )

    # Choose the side with better visibility.
    knee_column = (
        "left_knee_angle"
        if left_coverage >= right_coverage
        else "right_knee_angle"
    )

    selected_side = (
        "Left"
        if knee_column == "left_knee_angle"
        else "Right"
    )

    repetitions, smoothed_angles = detect_lunge_repetitions(
        df,
        knee_column
    )

    valid_angles = pd.Series(
        smoothed_angles
    ).dropna()

    if valid_angles.empty:
        raise ValueError(
            "Knee angles could not be detected. "
            "Try a clearer side-view video."
        )

    average_knee_angle = float(valid_angles.mean())
    minimum_knee_angle = float(valid_angles.min())
    maximum_knee_angle = float(valid_angles.max())

    average_rom = (
        float(np.mean([
            rep["knee_rom"]
            for rep in repetitions
        ]))
        if repetitions else 0
    )

    average_rep_duration = (
        float(np.mean([
            rep["duration"]
            for rep in repetitions
        ]))
        if repetitions else 0
    )

    # Rep consistency based on variation in duration.
    durations = [
        rep["duration"]
        for rep in repetitions
    ]

    if len(durations) > 1:
        duration_std = float(np.std(durations))
        duration_mean = float(np.mean(durations))

        consistency = max(
            0,
            min(
                100,
                100 - (duration_std / duration_mean * 100)
            )
        )
    else:
        consistency = 100 if repetitions else 0

    # Visibility coverage.
    coverage = max(
        left_coverage,
        right_coverage
    )

    # Available torso measurements.
    torso_column = (
        "left_torso_angle"
        if selected_side == "Left"
        else "right_torso_angle"
    )

    torso_angles = df[torso_column].dropna()

    average_torso_angle = (
        float(torso_angles.mean())
        if not torso_angles.empty
        else None
    )

    # Knee-angle graph data.
    graph_data = []

    for i, row in df.iterrows():
        angle = smoothed_angles[i]

        if not np.isnan(angle):
            graph_data.append({
                "time": round(
                    float(row["time_seconds"]), 3
                ),
                "knee_angle": round(float(angle), 2)
            })

    result = {
        "activity": "Lunge",
        "selected_side": selected_side,
        "detected_repetitions": len(repetitions),
        "average_knee_angle": round(
            average_knee_angle, 2
        ),
        "minimum_knee_angle": round(
            minimum_knee_angle, 2
        ),
        "maximum_knee_angle": round(
            maximum_knee_angle, 2
        ),
        "average_knee_rom": round(
            average_rom, 2
        ),
        "average_rep_duration": round(
            average_rep_duration, 2
        ),
        "rep_consistency": round(
            consistency, 2
        ),
        "average_torso_angle": (
            round(average_torso_angle, 2)
            if average_torso_angle is not None
            else None
        ),
        "left_visibility": round(
            float(left_coverage), 2
        ),
        "right_visibility": round(
            float(right_coverage), 2
        ),
        "coverage": round(float(coverage), 2),
        "repetitions": repetitions,
        "graph_data": graph_data
    }

    os.makedirs(
        os.path.dirname(output_json),
        exist_ok=True
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4
        )

    print("\nLUNGE ANALYSIS COMPLETED")
    print("Repetitions:", len(repetitions))
    print("Selected side:", selected_side)
    print("Average knee angle:", average_knee_angle)
    print("Average ROM:", average_rom)
    print("Coverage:", coverage)

    return result
