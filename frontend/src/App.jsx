import { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const EXERCISES = [
  { key: "squat", label: "Squat", short: "SQ" },
  { key: "sprint", label: "Sprint", short: "SP" },
  { key: "vertical-jump", label: "Vertical Jump", short: "VJ" },
  { key: "push-up", label: "Push-up", short: "PU" },
  { key: "lunge", label: "Lunge", short: "LU" },
];

function normalizeActivity(value = "") {
  const text = String(value).toLowerCase().replace(/[ _]/g, "-");
  if (text.includes("vertical") || text.includes("jump")) return "vertical-jump";
  if (text.includes("sprint") || text.includes("run")) return "sprint";
  if (text.includes("push")) return "push-up";
  if (text.includes("lunge")) return "lunge";
  if (text.includes("squat")) return "squat";
  return text;
}

function getRecordScore(record) {
  const score = record?.performance_score ?? record?.assessment?.performance_score ?? record?.assessment?.score;
  const numeric = Number(score);
  return Number.isFinite(numeric) ? numeric : null;
}

function buildInsights(assessment = {}, activity = "squat") {
  const metrics = assessment.metrics || assessment;
  const componentSource = assessment.component_scores || assessment.assessment_components || {};
  const components = Object.entries(componentSource)
    .map(([key, value]) => ({
      key,
      name: key.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      value: value == null ? NaN : Number(value),
    }))
    .filter((item) => Number.isFinite(item.value));
  const sorted = [...components].sort((a, b) => b.value - a.value);
  const existing = Array.isArray(assessment.recommendations)
    ? assessment.recommendations.filter(Boolean)
    : assessment.recommendation ? [assessment.recommendation] : [];
  const score = getRecordScore({ assessment, performance_score: assessment.performance_score ?? assessment.overall_score ?? assessment.score });
  const activityLabels = { squat: "Squat", sprint: "Sprint", "vertical-jump": "Vertical Jump", "push-up": "Push-up", lunge: "Lunge" };
  const activityLabel = activityLabels[activity] || "Exercise";
  const insights = [];

  if (sorted.length) {
    const best = sorted[0];
    insights.push({ type: "strength", title: `Strongest component: ${best.name}`, text: `${best.name} scored ${best.value.toFixed(1)}/100 in this ${activityLabel.toLowerCase()} assessment. This is the strongest available component score for this recording.` });
    const lowest = sorted[sorted.length - 1];
    if (lowest.key !== best.key && lowest.value < 80) {
      insights.push({ type: "focus", title: `Review ${lowest.name}`, text: `${lowest.name} received ${lowest.value.toFixed(1)}/100, the lowest available component score. Review the corresponding movement in the processed video and follow the assessment feedback.` });
    }
  }

  const metricRules = {
    squat: [
      ["knee_symmetry_difference", "Knee symmetry difference", "This is the measured left–right knee movement difference. Compare it with the symmetry component and the pose video; the value alone does not determine whether movement is safe."],
      ["hip_symmetry_difference", "Hip symmetry difference", "This is the measured left–right hip movement difference. Use the pose overlay to check visibility and compare sides; it is not a diagnosis."],
      ["trunk_angle_change", "Trunk angle change", "The system measured trunk-angle change during the squat. Review the video alongside this value because camera angle can affect measurements."],
      ["rep_time_difference", "Rep timing variation", "This represents the measured difference in repetition timing. Review whether the repetitions look consistent in the video."],
    ],
    sprint: [
      ["knee_angle_variability", "Knee-angle variability", "The knee angle varied across the analyzed running sequence. Review the knee-angle graph and video together; this value can also be affected by visibility and camera setup."],
    ],
    "vertical-jump": [
      ["knee_rom_difference", "Left–right knee ROM difference", "This is the measured difference between sides. Check that both knees were clearly visible during the jump; the measurement is not a standalone quality or safety judgement."],
      ["flight_time", "Detected flight time", "The system estimated the airborne interval from the video. Jump-height estimation depends on accurate takeoff and landing detection."],
    ],
    "push-up": [
      ["body_alignment_deviation", "Body-alignment deviation", "This is the measured alignment deviation during the analyzed movement. Review the pose overlay and ensure your whole body was visible."],
      ["elbow_symmetry_difference", "Elbow symmetry difference", "The system measured a left–right elbow difference. Confirm both arms were visible; camera angle or occlusion can affect this result."],
      ["rep_time_difference", "Rep timing variation", "This reflects repetition timing variation. Check the repetition sequence and use a comfortable, controlled pace."],
    ],
    lunge: [
      ["rep_consistency", "Repetition consistency", "This is the backend's repetition-consistency metric. Review the detected repetitions to understand which movements contributed to the result."],
      ["average_torso_angle", "Average torso angle", "The system measured average torso angle during the lunges. Camera position and visibility influence pose measurements."],
    ],
  };
  const rules = metricRules[activity] || [];
  for (const [key, title, explanation] of rules) {
    const value = metrics?.[key];
    if (value !== undefined && value !== null && Number.isFinite(Number(value))) {
      const unit = key.includes("time") || key.includes("flight") ? " s" : key.includes("consistency") ? "%" : "°";
      insights.push({ type: "metric", title: `${title}: ${Number(value).toFixed(1)}${unit}`, text: explanation });
      if (insights.length >= 4) break;
    }
  }

  const coverage = assessment.score_coverage_percent ?? assessment.score_coverage ?? assessment.coverage;
  if (coverage !== undefined && coverage !== null && Number.isFinite(Number(coverage))) {
    insights.push({ type: "neutral", title: `Score coverage: ${Number(coverage).toFixed(0)}%`, text: "Coverage indicates how much of the assessment's scoring information was available. Missing measurements may limit how complete the score is." });
  }
  if (score !== null && !sorted.length) {
    insights.push({ type: score >= 80 ? "strength" : "focus", title: "Overall score context", text: `This ${activityLabel.toLowerCase()} assessment scored ${score.toFixed(1)}/100. Use the displayed metrics and feedback to understand the result; the prototype score is not a validated athletic rating.` });
  }
  existing.slice(0, 2).forEach((text, i) => insights.push({ type: "recommendation", title: `Assessment feedback ${i + 1}`, text: String(text) }));
  if (!insights.length) insights.push({ type: "neutral", title: "More data needed", text: "This report has limited component or metric data. Try a clear recording with the movement and relevant joints visible to provide more information." });
  return insights.slice(0, 5);
}

function App() {
  const [activity, setActivity] = useState("squat");
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState("library");
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [history, setHistory] = useState([]);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] = useState(null);
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false);
  const [comparisonMode, setComparisonMode] = useState("progress");
  const [comparisonExercise, setComparisonExercise] = useState("squat");
  const [comparisonVideoA, setComparisonVideoA] = useState(null);
  const [comparisonVideoB, setComparisonVideoB] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);

  const assessment = result?.assessment || {};
  const metrics = assessment.metrics || {};
  const componentScores = assessment.component_scores || {};
  const repDetails = assessment.rep_details || [];
  const feedback = Array.isArray(assessment.recommendations)
    ? assessment.recommendations
    : assessment.recommendation
    ? [assessment.recommendation]
    : [];
  const kneeAngleData = result?.knee_angle_data || [];

  const isVerticalJump = activity === "vertical-jump";
  const isSprint = activity === "sprint";
  const isSquat = activity === "squat";
  const isPushup = activity === "push-up";
  const isLunge = activity === "lunge";
  const isNewExercise = isPushup || isLunge;
  const newExerciseMetrics = isLunge ? assessment : (assessment.metrics || {});
  const newExerciseComponents = isLunge
    ? (assessment.assessment_components || assessment.component_scores || {})
    : (assessment.component_scores || {});
  const newExerciseCount =
    assessment.detected_repetitions ??
    assessment.repetitions_count ??
    (Array.isArray(assessment.repetitions)
      ? assessment.repetitions.length
      : assessment.repetitions);
  const newExerciseCoverage =
    assessment.coverage ??
    assessment.score_coverage_percent ??
    assessment.score_coverage;

  // Squat API sends time_seconds + average_knee_angle.
  // Convert these to the time + knee_angle fields used by the chart.
  const chartKneeAngleData = useMemo(() => {
    if (!isSquat) {
      return kneeAngleData;
    }

    return kneeAngleData
      .map((item) => {
        const angle =
          item.average_knee_angle ??
          item.right_knee_angle ??
          item.left_knee_angle;

        return {
          ...item,
          time: Number(item.time_seconds),
          knee_angle:
            angle === null || angle === undefined || angle === ""
              ? null
              : Number(angle),
        };
      })
      .filter(
        (item) =>
          Number.isFinite(item.time) &&
          Number.isFinite(item.knee_angle)
      );
  }, [isSquat, kneeAngleData]);

  // Vertical Jump event times are returned inside assessment.metrics.
  // Keep the top-level fallback for compatibility with older API responses.
  const takeoffTime = metrics.takeoff_time ?? result?.takeoff_time;
  const landingTime = metrics.landing_time ?? result?.landing_time;

  const formatNumber = (value, decimals = 2) => {
    if (
      value === undefined ||
      value === null ||
      Number.isNaN(Number(value))
    ) {
      return "—";
    }

    return Number(value).toFixed(decimals);
  };

  const getVideoUrl = (path) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    return `${API_BASE}${path}`;
  };

  const loadAnalysisHistory = async () => {
    setActiveView("history");
    setSelectedHistoryRecord(null);
    setHistoryLoading(true);
    setHistoryError(null);

    try {
      const response = await fetch(`${API_BASE}/api/analyses/?limit=100&offset=0`);
      if (!response.ok) {
        throw new Error("Could not load analysis history. Please check that the backend is running.");
      }
      const data = await response.json();
      setHistory(Array.isArray(data.analyses) ? data.analyses : []);
    } catch (err) {
      console.error(err);
      setHistoryError(err.message || "Unable to load analysis history.");
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadDashboard = async () => {
    setActiveView("dashboard");
    setSelectedHistoryRecord(null);
    setDashboardLoading(true);
    setHistoryError(null);
    try {
      const response = await fetch(`${API_BASE}/api/analyses/?limit=100&offset=0`);
      if (!response.ok) throw new Error("Could not load dashboard data. Check that the backend is running.");
      const data = await response.json();
      setHistory(Array.isArray(data.analyses) ? data.analyses : []);
    } catch (err) {
      setHistoryError(err.message || "Unable to load dashboard data.");
    } finally {
      setDashboardLoading(false);
    }
  };

  const openHistoryRecord = async (analysisId) => {
    setHistoryDetailLoading(true);
    setHistoryError(null);

    try {
      const response = await fetch(`${API_BASE}/api/analyses/${encodeURIComponent(analysisId)}`);
      if (!response.ok) {
        throw new Error(response.status === 404 ? "This analysis record could not be found." : "Could not load this analysis report.");
      }
      const data = await response.json();
      setSelectedHistoryRecord(data);
    } catch (err) {
      console.error(err);
      setHistoryError(err.message || "Unable to load this analysis.");
    } finally {
      setHistoryDetailLoading(false);
    }
  };

  const returnToAnalysis = () => {
    setActiveView("library");
    setSelectedExercise(null);
    setVideo(null);
    setResult(null);
    setError(null);
    setSelectedHistoryRecord(null);
    setHistoryError(null);
  };

  const formatHistoryDate = (value) => {
    if (!value) return "Date unavailable";
    const date = new Date(value);
    return Number.isNaN(date.getTime())
      ? "Date unavailable"
      : date.toLocaleString();
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0] || null;

    setVideo(selectedFile);
    setError(null);
    setResult(null);
  };

  const handleActivityChange = (event) => {
    setActivity(event.target.value);
    setVideo(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!video) {
      setError("Please select a video first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("video", video);

      const endpoint = {
        "vertical-jump": "/api/analyze/vertical-jump",
        sprint: "/api/analyze/sprint",
        squat: "/api/analyze/squat",
        "push-up": "/api/analyze/pushup",
        lunge: "/api/analyze/lunge",
      }[activity] || "/api/analyze/squat";

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          "Analysis failed. Please make sure the FastAPI backend is running."
        );
      }

      const data = await response.json();
      setResult(data);
      setActiveView("analysis");
    } catch (err) {
      console.error(err);

      setError(
        err instanceof TypeError && err.message.toLowerCase().includes("fetch")
          ? "Cannot connect to the backend. Start FastAPI at http://127.0.0.1:8000 and try again."
          : err.message || "Something went wrong while analyzing the video."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCompareVideos = async (event) => {
    event?.preventDefault?.();
    if (!comparisonVideoA || !comparisonVideoB) {
      setComparisonError("Please select both videos before starting the comparison.");
      return;
    }
    setComparisonLoading(true);
    setComparisonError(null);
    setComparisonResult(null);
    try {
      const formData = new FormData();
      formData.append("video_a", comparisonVideoA);
      formData.append("video_b", comparisonVideoB);
      formData.append("comparison_mode", comparisonMode);
      const comparisonEndpoint = {
        squat: "/api/compare/squat",
        sprint: "/api/compare/sprint",
        "vertical-jump": "/api/compare/vertical-jump",
        pushup: "/api/compare/pushup",
        lunge: "/api/compare/lunge",
      }[comparisonExercise] || "/api/compare/squat";
      const response = await fetch(`${API_BASE}${comparisonEndpoint}`, { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || "Comparison failed. Check the backend and try again.");
      }
      setComparisonResult(data);
      setActiveView("comparison");
    } catch (err) {
      console.error(err);
      setComparisonError(err.message || "Unable to compare these videos.");
    } finally {
      setComparisonLoading(false);
    }
  };

  const openComparison = () => {
    setActiveView("comparison");
    setComparisonError(null);
    setSelectedHistoryRecord(null);
  };

  const scoreCards = useMemo(
    () => {
      if (isVerticalJump) {
        return [
          {
            label: "Performance Score",
            value:
              assessment.performance_score !== undefined
                ? `${formatNumber(assessment.performance_score, 0)}/100`
                : "—",
            detail: "No validated score available",
            accent: "blue",
          },
          {
            label: "Performance Level",
            value: assessment.performance_level || "—",
            detail: "No performance classification available",
            accent: "green",
          },
          {
            label: "Jump Height",
            value:
              metrics.estimated_jump_height_cm !== undefined
                ? `${formatNumber(
                    metrics.estimated_jump_height_cm
                  )} cm`
                : "—",
            detail: "Estimated from flight time",
            accent: "purple",
          },
          {
            label: "Flight Time",
            value:
              metrics.flight_time !== undefined
                ? `${formatNumber(metrics.flight_time, 3)} s`
                : "—",
            detail: "Detected flight phase",
            accent: "orange",
          },
        ];
      }

      if (isSprint) {
        const kneeMetrics = metrics.knee_movement || {};
        const duration =
          result?.sprint_metrics?.duration_seconds ??
          result?.sprint_metrics?.duration ??
          result?.duration_seconds ??
          metrics.duration_seconds;

        return [
          {
            label: "Performance Score",
            value:
              assessment.overall_score !== undefined
                ? `${formatNumber(assessment.overall_score, 0)}/100`
                : "—",
            detail: "Experimental prototype score",
            accent: "blue",
          },
          {
            label: "Performance Band",
            value: assessment.performance_band || "—",
            detail: assessment.performance_level || "Assessment level unavailable",
            accent: "green",
          },
          {
            label: "Average Knee ROM",
            value:
              kneeMetrics.average_rom !== undefined
                ? `${formatNumber(kneeMetrics.average_rom)}°`
                : "—",
            detail: "Average left and right knee movement",
            accent: "purple",
          },
          {
            label: "Duration",
            value:
              duration !== undefined && duration !== null
                ? `${formatNumber(duration, 2)} s`
                : "—",
            detail: "Analyzed running sequence",
            accent: "orange",
          },
        ];
      }

      if (isNewExercise) {
        const primaryMetric = isPushup
          ? (newExerciseMetrics.average_elbow_rom ??
             newExerciseMetrics.elbow_range_of_motion ??
             newExerciseMetrics.average_bottom_elbow_angle)
          : (newExerciseMetrics.average_knee_rom ??
             newExerciseMetrics.average_knee_range_of_motion);
        return [
          {
            label: "Performance Score",
            value: assessment.performance_score !== undefined
              ? `${formatNumber(assessment.performance_score, 0)}/100`
              : "—",
            detail: "Prototype assessment",
            accent: "blue",
          },
          {
            label: "Performance Level",
            value: assessment.performance_status || assessment.performance_band || assessment.movement_level || "—",
            detail: assessment.assessment_status || "Movement assessment",
            accent: "green",
          },
          {
            label: "Repetitions",
            value: newExerciseCount ?? "—",
            detail: "Detected complete repetitions",
            accent: "purple",
          },
          {
            label: isPushup ? "Average Elbow ROM" : "Average Knee ROM",
            value: primaryMetric != null ? `${formatNumber(primaryMetric)}°` : "—",
            detail: "Movement range",
            accent: "orange",
          },
        ];
      }

      return [
        {
          label: "Performance Score",
          value:
            assessment.performance_score !== undefined
              ? `${formatNumber(assessment.performance_score, 0)}/100`
              : "—",
          detail: "Prototype assessment",
          accent: "blue",
        },
        {
          label: "Performance Band",
          value: assessment.performance_band || assessment.movement_level || "—",
          detail: assessment.assessment_status
            ? `${assessment.assessment_status} assessment`
            : "Based on detected movement",
          accent: "green",
        },
        {
          label: "Repetitions",
          value: assessment.repetitions ?? "—",
          detail: "Valid squat repetitions",
          accent: "purple",
        },
        {
          label: "Knee ROM",
          value:
            metrics.knee_range_of_motion !== undefined
              ? `${formatNumber(
                  metrics.knee_range_of_motion
                )}°`
              : "—",
          detail: "Range of motion",
          accent: "orange",
        },
      ];
    },
    [
      assessment,
      metrics,
      result,
      isVerticalJump,
      isSprint,
      isPushup,
      isLunge,
      isNewExercise,
      newExerciseMetrics,
      newExerciseCount,
    ]
  );

  const metricCards = isVerticalJump
    ? [
        [
          "Estimated Jump Height",
          metrics.estimated_jump_height_cm,
          " cm",
        ],
        ["Flight Time", metrics.flight_time, " s"],
        ["Knee ROM", metrics.knee_range_of_motion, "°"],
        [
          "Knee ROM Difference",
          metrics.knee_rom_difference,
          "°",
        ],
        [
          "Takeoff Knee Angle",
          metrics.takeoff_knee_angle,
          "°",
        ],
        [
          "Landing Knee Angle",
          metrics.landing_knee_angle,
          "°",
        ],
      ]
    : isSprint
    ? [
        ["Left Knee ROM", metrics.knee_movement?.left_rom, "°"],
        ["Right Knee ROM", metrics.knee_movement?.right_rom, "°"],
        ["Average Knee ROM", metrics.knee_movement?.average_rom, "°"],
        ["Knee Symmetry Difference", metrics.knee_movement?.symmetry_difference, "°"],
        ["Left Hip ROM", metrics.hip_movement?.left_rom, "°"],
        ["Right Hip ROM", metrics.hip_movement?.right_rom, "°"],
        ["Average Hip ROM", metrics.hip_movement?.average_rom, "°"],
        ["Hip Symmetry Difference", metrics.hip_movement?.symmetry_difference, "°"],
        ["Average Trunk ROM", metrics.trunk_movement?.average_rom, "°"],
        ["Knee Angle Variability", metrics.knee_angle_variability, "°"],
      ]
    : isPushup
    ? [
        ["Elbow Range of Motion", metrics.elbow_range_of_motion, "°"],
        ["Average Bottom Elbow Angle", metrics.average_bottom_elbow_angle, "°"],
        ["Body Alignment Deviation", metrics.body_alignment_deviation, "°"],
        ["Elbow Symmetry Difference", metrics.elbow_symmetry_available === false ? null : metrics.elbow_symmetry_difference, "°"],
        ["Average Rep Time", metrics.average_rep_time, " s"],
        ["Rep Time Difference", metrics.rep_time_difference, " s"],
      ]
    : isLunge
    ? [
        ["Average Knee Angle", newExerciseMetrics.average_knee_angle, "°"],
        ["Minimum Knee Angle", newExerciseMetrics.minimum_knee_angle, "°"],
        ["Maximum Knee Angle", newExerciseMetrics.maximum_knee_angle, "°"],
        ["Average Knee ROM", newExerciseMetrics.average_knee_rom, "°"],
        ["Average Rep Duration", newExerciseMetrics.average_rep_duration, " s"],
        ["Rep Consistency", newExerciseMetrics.rep_consistency, "%"],
        ["Average Torso Angle", newExerciseMetrics.average_torso_angle, "°"],
        ["Left Visibility", newExerciseMetrics.left_visibility, "%"],
        ["Right Visibility", newExerciseMetrics.right_visibility, "%"],
      ]
    : [
        [
          "Average Bottom Angle",
          metrics.average_bottom_angle,
          "°",
        ],
        [
          "Deepest Rep Angle",
          metrics.deepest_rep_angle,
          "°",
        ],
        [
          "Hip ROM",
          metrics.hip_range_of_motion,
          "°",
        ],
        [
          "Knee Symmetry Difference",
          metrics.knee_symmetry_difference,
          "°",
        ],
        [
          "Hip Symmetry Difference",
          metrics.hip_symmetry_difference,
          "°",
        ],
        ["Descent Time", metrics.descent_time, " s"],
        ["Ascent Time", metrics.ascent_time, " s"],
        [
          "Average Rep Time",
          metrics.average_rep_time,
          " s",
        ],
        [
          "Fastest Rep",
          metrics.fastest_rep_time,
          " s",
        ],
        [
          "Slowest Rep",
          metrics.slowest_rep_time,
          " s",
        ],
        [
          "Rep Time Difference",
          metrics.rep_time_difference,
          " s",
        ],
        [
          "Trunk Angle Change",
          metrics.trunk_angle_change,
          "°",
        ],
      ];

  const noValidReps = Boolean(
    (isSquat && result && (assessment.repetitions || 0) === 0) ||
    (isNewExercise && result && Number(newExerciseCount || 0) === 0)
  );

  /*
    Sprint backend returns:
      time_seconds
      left_knee_angle
      right_knee_angle

    Convert those values into chart-friendly data.
  */
  const sprintChartData = useMemo(() => {
    if (!isSprint || !kneeAngleData.length) {
      return [];
    }

    return kneeAngleData
      .map((item) => ({
        time: Number(item.time_seconds),
        left_knee_angle:
          item.left_knee_angle === null || item.left_knee_angle === undefined
            ? null
            : Number(item.left_knee_angle),
        right_knee_angle:
          item.right_knee_angle === null || item.right_knee_angle === undefined
            ? null
            : Number(item.right_knee_angle),
      }))
      .filter(
        (item) =>
          Number.isFinite(item.time) &&
          (Number.isFinite(item.left_knee_angle) ||
            Number.isFinite(item.right_knee_angle))
      );
  }, [isSprint, kneeAngleData]);

  const activityChartData = useMemo(() => {
    if (isPushup) {
      return (result?.elbow_angle_data || result?.angle_data || [])
        .map((item) => ({
          time: Number(item.time_seconds ?? item.time),
          angle: Number(item.elbow_angle ?? item.average_elbow_angle ?? item.angle),
        }))
        .filter((item) => Number.isFinite(item.time) && Number.isFinite(item.angle));
    }
    if (isLunge) {
      return (assessment.graph_data || result?.graph_data || result?.knee_angle_data || [])
        .map((item) => ({
          time: Number(item.time_seconds ?? item.time),
          angle: Number(item.knee_angle ?? item.average_knee_angle ?? item.angle),
        }))
        .filter((item) => Number.isFinite(item.time) && Number.isFinite(item.angle));
    }
    return [];
  }, [isPushup, isLunge, result, assessment]);

  const newExerciseReps = Array.isArray(assessment.rep_details)
    ? assessment.rep_details
    : Array.isArray(assessment.reps)
    ? assessment.reps
    : Array.isArray(assessment.repetitions)
    ? assessment.repetitions
    : Array.isArray(result?.rep_details)
    ? result.rep_details
    : Array.isArray(result?.repetitions)
    ? result.repetitions
    : [];

  return (
    <div className="app-shell">
      <style>{styles}</style>

      <header className="topbar">
        <div className="container topbar-inner">
          <div className="brand">
            <div className="brand-mark">S</div>

            <div>
              <div className="brand-name">SportsAI</div>
              <div className="brand-subtitle">
                Talent Assessment Platform
              </div>
            </div>
          </div>

          <div className="topbar-actions">
            <button
              type="button"
              className={`nav-button ${activeView === "library" || activeView === "analysis" ? "nav-button-active" : ""}`}
              onClick={returnToAnalysis}
            >
              New Analysis
            </button>
            <button
              type="button"
              className={`nav-button ${activeView === "dashboard" ? "nav-button-active" : ""}`}
              onClick={loadDashboard}
            >
              Athlete Dashboard
            </button>
            <button
              type="button"
              className={`nav-button ${activeView === "comparison" ? "nav-button-active" : ""}`}
              onClick={openComparison}
            >
              Compare Videos
            </button>
            <button
              type="button"
              className={`nav-button ${activeView === "history" ? "nav-button-active" : ""}`}
              onClick={loadAnalysisHistory}
            >
              Analysis History
            </button>
            <div className="topbar-badge">
              <span className="status-dot" />
              AI / ML Performance Analysis
            </div>
          </div>
        </div>
      </header>

      <main>
        {activeView === "library" ? (
          <ExerciseLibrary
            selectedExercise={selectedExercise}
            onSelectExercise={(key) => { setActivity(key); setSelectedExercise(key); setVideo(null); setResult(null); setError(null); }}
            onBack={() => setSelectedExercise(null)}
            video={video}
            onFileChange={handleFileChange}
            onAnalyze={handleAnalyze}
            loading={loading}
            error={error}
          />
        ) : activeView === "dashboard" ? (
          <AthleteDashboard
            history={history}
            loading={dashboardLoading}
            error={historyError}
            onRefresh={loadDashboard}
            onNewAnalysis={returnToAnalysis}
            onOpenHistory={loadAnalysisHistory}
            formatDate={formatHistoryDate}
          />
        ) : activeView === "comparison" ? (
          <ComparisonWorkspace
            exercise={comparisonExercise}
            setExercise={(value) => { setComparisonExercise(value); setComparisonResult(null); setComparisonError(null); setComparisonVideoA(null); setComparisonVideoB(null); }}
            mode={comparisonMode}
            setMode={setComparisonMode}
            videoA={comparisonVideoA}
            setVideoA={setComparisonVideoA}
            videoB={comparisonVideoB}
            setVideoB={setComparisonVideoB}
            loading={comparisonLoading}
            error={comparisonError}
            result={comparisonResult}
            onCompare={handleCompareVideos}
            getVideoUrl={getVideoUrl}
            formatNumber={formatNumber}
            onNewComparison={() => { setComparisonResult(null); setComparisonError(null); setComparisonVideoA(null); setComparisonVideoB(null); }}
          />
        ) : activeView === "history" ? (
          <section className="history-section">
            <div className="container">
              <div className="history-heading">
                <div>
                  <div className="section-kicker">SAVED RESULTS</div>
                  <h1>Analysis History</h1>
                  <p>Review previous Squat, Vertical Jump and Sprint assessments.</p>
                </div>
                <button type="button" className="history-refresh" onClick={loadAnalysisHistory} disabled={historyLoading}>
                  {historyLoading ? "Refreshing…" : "Refresh History"}
                </button>
              </div>

              {historyError && <div className="history-error">{historyError}</div>}

              {selectedHistoryRecord ? (
                (selectedHistoryRecord.comparison || selectedHistoryRecord.result?.comparison || selectedHistoryRecord.comparison_result || selectedHistoryRecord.record_type === "comparison" || selectedHistoryRecord.analysis_type === "comparison" || String(selectedHistoryRecord.activity || "").toLowerCase().includes("comparison")) ? (
                  <ComparisonReport
                    data={selectedHistoryRecord.assessment?.comparison ? selectedHistoryRecord.assessment : (selectedHistoryRecord.comparison || selectedHistoryRecord.result?.comparison || selectedHistoryRecord.comparison_result || selectedHistoryRecord.assessment || selectedHistoryRecord)}
                    getVideoUrl={getVideoUrl}
                    formatNumber={formatNumber}
                    historical
                    createdAt={formatHistoryDate(selectedHistoryRecord.created_at)}
                    onBack={() => setSelectedHistoryRecord(null)}
                  />
                ) : (
                <div className="history-detail-card">
                  <div className="history-detail-top">
                    <button type="button" className="history-back-button" onClick={() => setSelectedHistoryRecord(null)}>← Back to history</button>
                    <span className="history-date">{formatHistoryDate(selectedHistoryRecord.created_at)}</span>
                  </div>
                  <div className="history-detail-title">
                    <div>
                      <div className="section-kicker">FULL ASSESSMENT REPORT</div>
                      <h2>{selectedHistoryRecord.activity}</h2>
                      <p>{selectedHistoryRecord.filename || "Uploaded video"}</p>
                    </div>
                    <div className="history-score-large">
                      <strong>{selectedHistoryRecord.assessment?.performance_score ?? selectedHistoryRecord.assessment?.score ?? "—"}</strong>
                      <span>/ 100</span>
                      <small>{selectedHistoryRecord.assessment?.performance_level || selectedHistoryRecord.assessment?.movement_level || selectedHistoryRecord.assessment?.rating || "Assessment"}</small>
                    </div>
                  </div>

                  {selectedHistoryRecord.processed_video_url && (
                    <div className="history-video-panel">
                      <h3>AI Processed Video</h3>
                      <video src={getVideoUrl(selectedHistoryRecord.processed_video_url)} controls playsInline />
                    </div>
                  )}

                  {selectedHistoryRecord.assessment?.assessment_status && (
                    <div className="squat-assessment-summary">
                      <div className="squat-summary-item">
                        <span>Assessment Status</span>
                        <strong className={selectedHistoryRecord.assessment.assessment_status === "Provisional" ? "status-provisional" : ""}>
                          {selectedHistoryRecord.assessment.assessment_status}
                        </strong>
                      </div>
                      <div className="squat-summary-item">
                        <span>Score Coverage</span>
                        <strong>
                          {selectedHistoryRecord.assessment.score_coverage_percent !== undefined &&
                          selectedHistoryRecord.assessment.score_coverage_percent !== null
                            ? `${formatNumber(selectedHistoryRecord.assessment.score_coverage_percent, 0)}%`
                            : "—"}
                        </strong>
                      </div>
                      {selectedHistoryRecord.assessment.score_interpretation && (
                        <p>{selectedHistoryRecord.assessment.score_interpretation}</p>
                      )}
                    </div>
                  )}

                  <div className="history-subheading">Movement Metrics</div>
                  <div className="metrics-grid history-metrics-grid">
                    {Object.entries(selectedHistoryRecord.assessment?.metrics || {}).map(([key, value]) => (
                      <MetricCard key={key} label={key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())} value={value === null && ["knee_symmetry_difference", "hip_symmetry_difference"].includes(key)
                        ? "Unavailable"
                        : typeof value === "number"
                        ? formatNumber(value)
                        : String(value ?? "—")} />
                    ))}
                    {Object.keys(selectedHistoryRecord.assessment?.metrics || {}).length === 0 && (
                      <div className="empty-state">No detailed metrics are available for this record.</div>
                    )}
                  </div>

                  {Object.keys(selectedHistoryRecord.assessment?.component_scores || {}).length > 0 && (
                    <div className="history-subsection">
                      <div className="history-subheading">Component Scores</div>
                      <div className="score-bars">
                        {Object.entries(selectedHistoryRecord.assessment.component_scores).map(([key, value]) => (
                          <ScoreBar key={key} label={key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())} value={value} />
                        ))}
                      </div>
                    </div>
                  )}

                  {Array.isArray(selectedHistoryRecord.assessment?.recommendations) && selectedHistoryRecord.assessment.recommendations.length > 0 && (
                    <div className="history-subsection">
                      <div className="history-subheading">Recommendations</div>
                      <div className="feedback-list">
                        {selectedHistoryRecord.assessment.recommendations.map((item, index) => (
                          <div className="feedback-item" key={`${item}-${index}`}><span>✓</span><p>{item}</p></div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="history-id">Analysis ID: {selectedHistoryRecord.analysis_id}</div>
                </div>
                )
              ) : historyLoading ? (
                <div className="history-empty">Loading your saved analyses…</div>
              ) : history.length === 0 ? (
                <div className="history-empty">No saved analyses found yet. Complete an activity analysis to see it here.</div>
              ) : (
                <div className="history-list">
                  {history.map((record) => (
                    <article className="history-record" key={record.analysis_id}>
                      <div className="history-record-main">
                        <div className="history-activity-icon">{record.activity?.toLowerCase().includes("sprint") ? "SP" : record.activity?.toLowerCase().includes("jump") ? "VJ" : record.activity?.toLowerCase().includes("push") ? "PU" : record.activity?.toLowerCase().includes("lunge") ? "LU" : "SQ"}</div>
                        <div className="history-record-info">
                          <h3>{record.activity}</h3>
                          <p>{record.filename || "Uploaded video"}</p>
                          <span>{formatHistoryDate(record.created_at)}</span>
                        </div>
                      </div>
                      <div className="history-record-score">
                        <strong>{record.performance_score ?? "—"}</strong>
                        <span>{record.performance_level || "Assessment"}</span>
                      </div>
                      <button type="button" className="history-view-button" onClick={() => openHistoryRecord(record.analysis_id)} disabled={historyDetailLoading}>View Report <span>→</span></button>
                    </article>
                  ))}
                </div>
              )}
              {historyDetailLoading && <div className="history-loading-note">Loading selected report…</div>}
            </div>
          </section>
        ) : (
          <>
        <section className="hero">
          <div className="container hero-grid">
            <div className="hero-copy">
              <div className="eyebrow">
                SPORTS PERFORMANCE ANALYTICS
              </div>

              <h1>
                Analyze movement.
                <br />
                <span>Understand performance.</span>
              </h1>

              <p>
                Upload a short sports video and let our
                computer-vision pipeline measure movement
                patterns, repetitions, range of motion and
                performance indicators.
              </p>

              <div className="hero-points">
                <span>✓ Video-based analysis</span>
                <span>✓ Pose tracking</span>
                <span>✓ Explainable metrics</span>
              </div>
            </div>

            <div className="hero-mini-card">
              <div className="mini-icon">01</div>

              <div>
                <strong>
                  {isVerticalJump
                    ? "Start with Vertical Jump Analysis"
                    : isSprint
                    ? "Start with Sprint Analysis"
                    : isPushup
                    ? "Start with Push-up Analysis"
                    : isLunge
                    ? "Start with Lunge Analysis"
                    : "Start with Squat Analysis"}
                </strong>

                <span>
                  {isVerticalJump
                    ? "Record one clear vertical jump from the side."
                    : isSprint
                    ? "Record one clear running sequence from the side."
                    : isPushup
                    ? "Record several clear push-ups from the side."
                    : isLunge
                    ? "Record several clear lunges with your body visible."
                    : "Record 5–10 clear repetitions from the side."}
                </span>
              </div>
            </div>
          </div>
        </section>

        <section
          className="analysis-section"
          id="analysis"
        >
          <div className="container">
            <div className="section-heading">
              <div>
                <div className="section-kicker">
                  VIDEO ANALYSIS
                </div>

                <h2>Start Your Analysis</h2>

                <p>
                  Follow the simple recording guide, then
                  upload your video.
                </p>
              </div>

              <div className="step-indicator">
                <span className="active">1</span>
                <i />
                <span>2</span>
                <i />
                <span>3</span>
              </div>
            </div>

            <div className="analysis-layout">
              <section className="guidelines-card">
                {isVerticalJump ? (
                  <>
                    <div className="card-heading">
                      <div className="heading-icon">
                        REC
                      </div>

                      <div>
                        <h3>
                          How to record your vertical jump
                        </h3>

                        <p>
                          A few simple things make the AI
                          jump analysis much clearer.
                        </p>
                      </div>
                    </div>

                    <div className="guide-grid">
                      <GuideStep
                        number="01"
                        title="Side view"
                        text="Place the camera to your side so your whole jump can be seen clearly."
                      />

                      <GuideStep
                        number="02"
                        title="Full body"
                        text="Keep your head, hips, knees and feet visible throughout the video."
                      />

                      <GuideStep
                        number="03"
                        title="Camera still"
                        text="Keep your phone on a stable surface while recording."
                      />

                      <GuideStep
                        number="04"
                        title="Good lighting"
                        text="Record in a bright area so your body is easy to see."
                      />

                      <GuideStep
                        number="05"
                        title="Clear jump"
                        text="Start standing, jump vertically with both feet and land clearly."
                      />

                      <GuideStep
                        number="06"
                        title="Space above"
                        text="Leave enough space above your head so you are not cropped during the jump."
                      />
                    </div>

                    <div className="guide-good">
                      <div className="guide-good-title">
                        ✓ For better analysis
                      </div>

                      <div className="guide-good-list">
                        <span>
                          Whole body inside the frame
                        </span>

                        <span>Camera stays still</span>

                        <span>
                          Takeoff and landing visible
                        </span>

                        <span>Good lighting</span>
                      </div>
                    </div>

                    <div className="guide-warning">
                      <div className="warning-icon">
                        !
                      </div>

                      <div>
                        <strong>
                          Avoid these problems
                        </strong>

                        <p>
                          Cropped feet or head, a moving
                          camera, very dark lighting or
                          another person blocking the
                          athlete can reduce detection
                          accuracy.
                        </p>
                      </div>
                    </div>
                  </>
                ) : isNewExercise ? (
                  <>
                    <div className="card-heading">
                      <div className="heading-icon">REC</div>
                      <div>
                        <h3>{isPushup ? "How to record your push-up" : "How to record your lunge"}</h3>
                        <p>Use a clear, stable video so the movement and body joints remain visible.</p>
                      </div>
                    </div>
                    <div className="guide-grid">
                      <GuideStep number="01" title="Side view" text="Place the camera to your side so the movement can be seen clearly." />
                      <GuideStep number="02" title="Full body" text="Keep your body and the moving joints visible throughout the recording." />
                      <GuideStep number="03" title="Camera still" text="Keep the phone on a stable surface while recording." />
                      <GuideStep number="04" title="Good lighting" text="Record in a bright area without shadows hiding your joints." />
                      <GuideStep number="05" title="Complete reps" text={isPushup ? "Capture several complete push-ups, including the lowering and pushing-up phases." : "Capture several complete lunges, including lowering and returning to the starting position."} />
                      <GuideStep number="06" title="Clear space" text="Avoid objects or people blocking the athlete from the camera." />
                    </div>
                    <div className="guide-good">
                      <div className="guide-good-title">✓ For better analysis</div>
                      <div className="guide-good-list"><span>Body visible in frame</span><span>Camera stays still</span><span>Side-view recording</span><span>Clear movement repetitions</span></div>
                    </div>
                    <div className="guide-warning">
                      <div className="warning-icon">!</div>
                      <div><strong>Avoid these problems</strong><p>Cropped body parts, a moving camera, poor lighting or blocked joints can reduce detection accuracy.</p></div>
                    </div>
                  </>
                ) : isSprint ? (
                  <>
                    <div className="card-heading">
                      <div className="heading-icon">
                        REC
                      </div>

                      <div>
                        <h3>
                          How to record your sprint
                        </h3>

                        <p>
                          A few simple things make the AI
                          running analysis much clearer.
                        </p>
                      </div>
                    </div>

                    <div className="guide-grid">
                      <GuideStep
                        number="01"
                        title="Side view"
                        text="Place the camera to your side so your running movement can be seen clearly."
                      />

                      <GuideStep
                        number="02"
                        title="Full body"
                        text="Keep your head, hips, knees and feet visible throughout the run."
                      />

                      <GuideStep
                        number="03"
                        title="Camera still"
                        text="Keep your phone on a stable surface while recording."
                      />

                      <GuideStep
                        number="04"
                        title="Good lighting"
                        text="Record in a bright area so your body is easy to see."
                      />

                      <GuideStep
                        number="05"
                        title="Clear running space"
                        text="Make sure there is enough space for the runner to move through the frame."
                      />

                      <GuideStep
                        number="06"
                        title="One runner"
                        text="Keep the athlete being analyzed clearly visible without another person blocking the view."
                      />
                    </div>

                    <div className="guide-good">
                      <div className="guide-good-title">
                        ✓ For better analysis
                      </div>

                      <div className="guide-good-list">
                        <span>
                          Whole body inside the frame
                        </span>

                        <span>Camera stays still</span>

                        <span>Side-view recording</span>

                        <span>Good lighting</span>
                      </div>
                    </div>

                    <div className="guide-warning">
                      <div className="warning-icon">
                        !
                      </div>

                      <div>
                        <strong>
                          Avoid these problems
                        </strong>

                        <p>
                          Cropped feet or head, a moving
                          camera, very dark lighting,
                          another person blocking the
                          runner or the runner leaving the
                          frame can reduce detection
                          accuracy.
                        </p>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="card-heading">
                      <div className="heading-icon">
                        REC
                      </div>

                      <div>
                        <h3>
                          How to record your squat
                        </h3>

                        <p>
                          A few simple things make the AI
                          analysis much clearer.
                        </p>
                      </div>
                    </div>

                    <div className="guide-grid">
                      <GuideStep
                        number="01"
                        title="Side view"
                        text="Place the camera to your side, not directly in front."
                      />

                      <GuideStep
                        number="02"
                        title="Full body"
                        text="Keep your head, hips, knees and feet visible."
                      />

                      <GuideStep
                        number="03"
                        title="Camera still"
                        text="Keep your phone on a stable surface while recording."
                      />

                      <GuideStep
                        number="04"
                        title="Good lighting"
                        text="Record in a bright area so your body is easy to see."
                      />

                      <GuideStep
                        number="05"
                        title="Complete reps"
                        text="Start standing, squat down, stand up and repeat 5–10 times."
                      />

                      <GuideStep
                        number="06"
                        title="Clear space"
                        text="Keep people and objects from blocking your body."
                      />
                    </div>

                    <div className="guide-good">
                      <div className="guide-good-title">
                        ✓ For better analysis
                      </div>

                      <div className="guide-good-list">
                        <span>
                          Whole body inside the frame
                        </span>

                        <span>
                          Camera around hip height
                        </span>

                        <span>Legs clearly visible</span>

                        <span>5–10 complete squats</span>
                      </div>
                    </div>

                    <div className="guide-warning">
                      <div className="warning-icon">
                        !
                      </div>

                      <div>
                        <strong>
                          Avoid these problems
                        </strong>

                        <p>
                          Cropped legs, a moving camera,
                          very dark lighting or another
                          person blocking the athlete can
                          reduce detection accuracy.
                        </p>
                      </div>
                    </div>
                  </>
                )}
              </section>

              <section className="upload-card">
                <div className="upload-top">
                  <div>
                    <div className="section-kicker">
                      STEP 1
                    </div>

                    <h3>Choose your activity</h3>

                    <p>
                      {isVerticalJump
                        ? "Vertical Jump analysis is currently available."
                        : isSprint
                        ? "Sprint / Running analysis is currently available."
                        : isPushup
                        ? "Push-up analysis is currently available."
                        : isLunge
                        ? "Lunge analysis is currently available."
                        : "Squat analysis is currently available."}
                    </p>
                  </div>

                  <div className="available-pill">
                    ● Available
                  </div>
                </div>

                <label
                  className="field-label"
                  htmlFor="activity"
                >
                  Activity
                </label>

                <select
                  id="activity"
                  value={activity}
                  onChange={handleActivityChange}
                  className="activity-select"
                >
                  <option value="squat">
                    Squat
                  </option>

                  <option value="vertical-jump">
                    Vertical Jump
                  </option>

                  <option value="sprint">
                    Sprint / Running
                  </option>
                  <option value="push-up">Push-up</option>
                  <option value="lunge">Lunge</option>
                </select>

                <div className="upload-divider">
                  <span>STEP 2</span>
                </div>

                <div className="upload-zone">
                  <input
                    id="video-input"
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                  />

                  <label
                    htmlFor="video-input"
                    className="upload-zone-content"
                  >
                    <div className="upload-icon">
                      ↑
                    </div>

                    <strong>
                      {video
                        ? "Video selected"
                        : "Choose a video"}
                    </strong>

                    <span>
                      {video
                        ? video.name
                        : "MP4, MOV or another browser-supported video"}
                    </span>

                    <em>
                      Click here to browse files
                    </em>
                  </label>
                </div>

                {video && (
                  <div className="selected-file">
                    <div className="file-check">
                      ✓
                    </div>

                    <div className="selected-file-text">
                      <strong>{video.name}</strong>

                      <span>
                        {(video.size / 1024 / 1024).toFixed(
                          2
                        )}{" "}
                        MB
                      </span>
                    </div>

                    <button
                      type="button"
                      className="remove-file"
                      onClick={() => setVideo(null)}
                    >
                      Remove
                    </button>
                  </div>
                )}

                {error && (
                  <div className="error-box">
                    {error}
                  </div>
                )}

                <button
                  type="button"
                  className="analyze-button"
                  onClick={handleAnalyze}
                  disabled={loading || !video}
                >
                  {loading ? (
                    <>
                      <span className="spinner" />
                      Analyzing video...
                    </>
                  ) : (
                    <>
                      Analyze Performance{" "}
                      <span>→</span>
                    </>
                  )}
                </button>

                <p className="privacy-note">
                  Your video is processed by the local
                  project backend for this analysis.
                </p>
              </section>
            </div>
          </div>
        </section>

        {result && (
          <section
            className="results-section"
            id="results"
          >
            <div className="container">
              <div className="results-header">
                <div>
                  <div className="section-kicker">
                    ANALYSIS COMPLETE
                  </div>

                  <h2>Performance Dashboard</h2>

                  <p>
                    Movement measurements extracted from
                    your uploaded video.
                  </p>
                </div>

                <div className="analysis-id">
                  <span>Analysis ID</span>

                  <strong>
                    {result.analysis_id || "—"}
                  </strong>
                </div>
              </div>

              {noValidReps && (
                <div className="no-reps-banner">
                  <div className="no-reps-icon">
                    i
                  </div>

                  <div>
                    <strong>
                      {isLunge ? "No valid lunge repetitions detected" : isPushup ? "No valid push-up repetitions detected" : "No valid squat repetitions detected"}
                    </strong>

                    <p>
                      The video was processed, but the movement did not match the current detection pattern. Try the recording guide above with a clear view and complete repetitions.
                    </p>
                  </div>
                </div>
              )}

              <div className="score-grid">
                {scoreCards.map((card) => (
                  <StatCard
                    key={card.label}
                    {...card}
                  />
                ))}
              </div>

              <section className="insights-panel">
                <div className="insights-heading">
                  <div>
                    <div className="section-kicker">PERSONALIZED ANALYSIS</div>
                    <h3>Smart Performance Insights</h3>
                    <p>Highlights are derived from this assessment's available scores and feedback.</p>
                  </div>
                  <span className="insights-badge">Metric-driven</span>
                </div>
                <div className="insights-grid">
                  {buildInsights(assessment, activity).map((insight, index) => (
                    <article className={`insight-card insight-${insight.type}`} key={`${insight.title}-${index}`}>
                      <span className="insight-icon">{insight.type === "strength" ? "✓" : insight.type === "focus" ? "↗" : "✦"}</span>
                      <div><strong>{insight.title}</strong><p>{insight.text}</p></div>
                    </article>
                  ))}
                </div>
              </section>

              {isSquat && (
                <div className="squat-assessment-summary">
                  <div className="squat-summary-item">
                    <span>Assessment Status</span>
                    <strong className="status-provisional">
                      {assessment.assessment_status || "Not available"}
                    </strong>
                  </div>
                  <div className="squat-summary-item">
                    <span>Score Coverage</span>
                    <strong>
                      {assessment.score_coverage_percent !== undefined &&
                      assessment.score_coverage_percent !== null
                        ? `${formatNumber(assessment.score_coverage_percent, 0)}%`
                        : "—"}
                    </strong>
                  </div>
                  <p>
                    {assessment.score_interpretation ||
                      "The assessment uses the component scores available from this video. Unavailable measurements are excluded."}
                  </p>
                </div>
              )}

              {isNewExercise && (
                <div className="squat-assessment-summary">
                  <div className="squat-summary-item"><span>Assessment Status</span><strong>{assessment.assessment_status || "—"}</strong></div>
                  <div className="squat-summary-item"><span>Score Coverage</span><strong>{newExerciseCoverage != null ? `${formatNumber(newExerciseCoverage, 0)}%` : "—"}</strong></div>
                  <div className="squat-summary-item"><span>Detected Repetitions</span><strong>{newExerciseCount ?? "—"}</strong></div>
                  {isPushup && <div className="squat-summary-item"><span>Components Available</span><strong>{assessment.available_component_count ?? Object.values(newExerciseComponents).filter((v) => v != null).length}/{assessment.total_component_count ?? 5}</strong></div>}
                  {isLunge && <div className="squat-summary-item"><span>Selected Side</span><strong>{assessment.selected_side || "—"}</strong></div>}
                  <p>{assessment.score_interpretation || assessment.interpretation || (isPushup ? "Push-up scoring reflects available depth, alignment, symmetry, tempo and consistency measurements." : "Lunge assessment uses movement measurements and video visibility.")}</p>
                </div>
              )}

              {isSprint && (
                <div className="squat-assessment-summary">
                  <div className="squat-summary-item">
                    <span>Assessment Status</span>
                    <strong>{assessment.assessment_status || "—"}</strong>
                  </div>
                  <div className="squat-summary-item">
                    <span>Score Coverage</span>
                    <strong>
                      {assessment.score_coverage !== undefined &&
                      assessment.score_coverage !== null
                        ? `${formatNumber(assessment.score_coverage, 0)}%`
                        : "—"}
                    </strong>
                  </div>
                  <div className="squat-summary-item">
                    <span>Available Components</span>
                    <strong>{assessment.available_component_count ?? "—"}</strong>
                  </div>
                  <div className="squat-summary-item">
                    <span>Unavailable Components</span>
                    <strong>{assessment.unavailable_component_count ?? "—"}</strong>
                  </div>
                  <p>
                    {assessment.interpretation ||
                      assessment.ml_status ||
                      "Sprint assessment uses availability-aware, rule-based scoring."}
                  </p>
                </div>
              )}

              <div className="results-two-column">
                <div className="result-card video-card">
                  <CardTitle
                    title="Video Analysis"
                    subtitle={
                      isVerticalJump
                        ? "Original and AI-processed jump footage"
                        : isSprint
                        ? "Original and AI-processed sprint footage"
                        : "Original and AI-processed footage"
                    }
                  />

                  <div className="video-grid">
                    <VideoPanel
                      title="Original Video"
                      src={
                        video
                          ? URL.createObjectURL(video)
                          : ""
                      }
                    />

                    <VideoPanel
                      title="AI Pose Analysis"
                      src={getVideoUrl(
                        result.processed_video_url || result.video_url
                      )}
                    />
                  </div>
                </div>

                <div className="result-card breakdown-card">
                  <CardTitle
                    title={
                      isVerticalJump
                        ? "Vertical Jump Movement Breakdown"
                        : "Performance Breakdown"
                    }
                    subtitle={
                      isVerticalJump
                        ? "Measured movement values extracted from the uploaded video"
                        : "Component scores from the prototype assessment"
                    }
                  />

                  {isVerticalJump ? (
                    <>
                      <div className="metrics-grid">
                        <div className="metric-card">
                          <span>Knee Range of Motion</span>
                          <strong>{formatNumber(metrics.knee_range_of_motion)}°</strong>
                        </div>
                        <div className="metric-card">
                          <span>Left–Right Knee ROM Difference</span>
                          <strong>{formatNumber(metrics.knee_rom_difference)}°</strong>
                        </div>
                        <div className="metric-card">
                          <span>Takeoff Knee Angle</span>
                          <strong>{formatNumber(metrics.takeoff_knee_angle)}°</strong>
                        </div>
                        <div className="metric-card">
                          <span>Landing Knee Angle</span>
                          <strong>{formatNumber(metrics.landing_knee_angle)}°</strong>
                        </div>
                      </div>
                      <p className="movement-breakdown-note">
                        These are measured movement values, not component scores or performance ratings.
                      </p>
                    </>
                  ) : (
                    <div className="score-bars">
                      {isNewExercise ? (
                        <>
                          {Object.entries(newExerciseComponents).map(([key, value]) => (
                            <ScoreBar key={key} label={key.replaceAll("_", " ").replace(/\\b\\w/g, (c) => c.toUpperCase())} value={value} />
                          ))}
                        </>
                      ) : isSprint ? (
                        <>
                          <ScoreBar label="Knee Movement" value={componentScores.knee_movement} />
                          <ScoreBar label="Knee Symmetry" value={componentScores.knee_symmetry} />
                          <ScoreBar label="Hip Movement" value={componentScores.hip_movement} />
                          <ScoreBar label="Hip Symmetry" value={componentScores.hip_symmetry} />
                          <ScoreBar label="Trunk Movement" value={componentScores.trunk_movement} />
                        </>
                      ) : (
                        <>
                          <ScoreBar label="Depth" value={componentScores.depth_score} />
                          <ScoreBar label="Symmetry" value={componentScores.symmetry_score} />
                          <ScoreBar label="Movement Control" value={componentScores.movement_control_score} />
                          <ScoreBar label="Tempo" value={componentScores.tempo_score} />
                          <ScoreBar label="Consistency" value={componentScores.consistency_score} />
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {isNewExercise ? (
                <div className="result-card chart-card">
                  <CardTitle title={isPushup ? "Elbow Angle vs Time" : "Knee Angle vs Time"} subtitle={isPushup ? "Elbow movement throughout the analyzed push-ups" : "Knee movement during the analyzed lunges"} />
                  <div className="chart-wrap">
                    {activityChartData.length ? (
                      <ResponsiveContainer width="100%" height={330}>
                        <LineChart data={activityChartData} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e7ebf2" />
                          <XAxis dataKey="time" tickFormatter={(v) => `${Number(v).toFixed(1)}s`} stroke="#8a94a6" fontSize={12} />
                          <YAxis domain={[0, 190]} tickFormatter={(v) => `${v}°`} stroke="#8a94a6" fontSize={12} />
                          <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}°`, isPushup ? "Elbow angle" : "Knee angle"]} labelFormatter={(v) => `Time: ${Number(v).toFixed(2)}s`} />
                          <Line type="monotone" dataKey="angle" stroke="#2563eb" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : <div className="empty-chart">No angle data available for this activity.</div>}
                  </div>
                </div>
              ) : isSprint ? (
                <div className="result-card chart-card">
                  <CardTitle
                    title="Knee Angle vs Time"
                    subtitle="Left and right knee movement during the analyzed running sequence"
                  />

                  <div className="chart-wrap">
                    {sprintChartData.length > 0 ? (
                      <ResponsiveContainer
                        width="100%"
                        height={330}
                      >
                        <LineChart
                          data={sprintChartData}
                          margin={{
                            top: 12,
                            right: 20,
                            left: 0,
                            bottom: 8,
                          }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#e7ebf2"
                          />

                          <XAxis
                            dataKey="time"
                            tickFormatter={(value) =>
                              `${Number(value).toFixed(
                                1
                              )}s`
                            }
                            stroke="#8a94a6"
                            fontSize={12}
                          />

                          <YAxis
                            domain={[0, 190]}
                            tickFormatter={(value) =>
                              `${value}°`
                            }
                            stroke="#8a94a6"
                            fontSize={12}
                          />

                          <Tooltip
                            formatter={(
                              value,
                              name
                            ) => [
                              `${Number(value).toFixed(
                                1
                              )}°`,
                              name ===
                              "left_knee_angle"
                                ? "Left knee"
                                : "Right knee",
                            ]}
                            labelFormatter={(label) =>
                              `Time: ${Number(
                                label
                              ).toFixed(2)}s`
                            }
                          />

                          <Line
                            type="monotone"
                            dataKey="left_knee_angle"
                            stroke="#2563eb"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5 }}
                          />

                          <Line
                            type="monotone"
                            dataKey="right_knee_angle"
                            stroke="#7c3aed"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty-chart">
                        No sprint angle data available.
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="result-card chart-card">
                  <CardTitle
                    title="Knee Angle vs Time"
                    subtitle={
                      isVerticalJump
                        ? "Smoothed knee angle during the analyzed jump"
                        : "Smoothed knee angle across the analyzed video"
                    }
                  />

                  <div className="chart-wrap">
                    {chartKneeAngleData.length > 0 ? (
                      <ResponsiveContainer
                        width="100%"
                        height={330}
                      >
                        <LineChart
                          data={chartKneeAngleData}
                          margin={{
                            top: 12,
                            right: 20,
                            left: 0,
                            bottom: 8,
                          }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#e7ebf2"
                          />

                          <XAxis
                            dataKey="time"
                            tickFormatter={(value) =>
                              `${Number(value).toFixed(
                                1
                              )}s`
                            }
                            stroke="#8a94a6"
                            fontSize={12}
                          />

                          <YAxis
                            domain={[0, 190]}
                            tickFormatter={(value) =>
                              `${value}°`
                            }
                            stroke="#8a94a6"
                            fontSize={12}
                          />

                          <Tooltip
                            formatter={(value) => [
                              `${Number(value).toFixed(
                                1
                              )}°`,
                              "Knee angle",
                            ]}
                            labelFormatter={(label) =>
                              `Time: ${Number(
                                label
                              ).toFixed(2)}s`
                            }
                          />

                          {!isVerticalJump && (
                            <>
                              <ReferenceLine
                                y={160}
                                stroke="#94a3b8"
                                strokeDasharray="6 5"
                                label="Standing"
                              />

                              <ReferenceLine
                                y={90}
                                stroke="#94a3b8"
                                strokeDasharray="6 5"
                                label="Deep"
                              />
                            </>
                          )}

                          <Line
                            type="monotone"
                            dataKey="knee_angle"
                            stroke="#2563eb"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty-chart">
                        No angle data available.
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="result-card">
                <CardTitle
                  title="Movement Metrics"
                  subtitle={
                    isVerticalJump
                      ? "Detailed measurements extracted from the jump"
                      : isSprint
                      ? "Detailed measurements extracted from the running sequence"
                      : "Detailed measurements extracted from the video"
                  }
                />

                <div className="metrics-grid">
                  {metricCards.map(
                    ([label, value, suffix]) => (
                      <MetricCard
                        key={label}
                        label={label}
                        value={
                          noValidReps &&
                          [
                            "Average Bottom Angle",
                            "Deepest Rep Angle",
                            "Descent Time",
                            "Ascent Time",
                            "Average Rep Time",
                            "Fastest Rep",
                            "Slowest Rep",
                            "Rep Time Difference",
                          ].includes(label)
                            ? "—"
                            : isSquat &&
                              value === null &&
                              ["Knee Symmetry Difference", "Hip Symmetry Difference"].includes(label)
                            ? "Unavailable"
                            : `${formatNumber(
                                value
                              )}${suffix}`
                        }
                      />
                    )
                  )}
                </div>
              </div>

              {isSquat ? (
                <div className="result-card">
                  <CardTitle
                    title="Rep-by-Rep Analysis"
                    subtitle="Timing and bottom position for each detected repetition"
                  />

                  {repDetails.length > 0 ? (
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Rep</th>
                            <th>Start</th>
                            <th>Bottom</th>
                            <th>End</th>
                            <th>Descent</th>
                            <th>Ascent</th>
                            <th>Total Time</th>
                            <th>Bottom Angle</th>
                          </tr>
                        </thead>

                        <tbody>
                          {repDetails.map((rep) => (
                            <tr
                              key={rep.rep_number}
                            >
                              <td>
                                <strong>
                                  #{rep.rep_number}
                                </strong>
                              </td>

                              <td>
                                {formatNumber(
                                  rep.start_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.bottom_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.end_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.descent_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.ascent_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.rep_time
                                )}
                                s
                              </td>

                              <td>
                                {formatNumber(
                                  rep.bottom_angle
                                )}
                                °
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty-state">
                      No valid repetitions were detected
                      in this video.
                    </div>
                  )}
                </div>
              ) : isNewExercise ? (
                <div className="result-card">
                  <CardTitle title="Rep-by-Rep Analysis" subtitle="Detected repetition measurements" />
                  {newExerciseReps.length > 0 ? (
                    <div className="table-scroll"><table><thead><tr><th>Rep</th><th>Start</th><th>Bottom</th><th>End</th><th>Duration</th><th>Angle</th><th>ROM</th></tr></thead><tbody>
                      {newExerciseReps.map((rep, idx) => <tr key={rep.rep_number ?? rep.rep ?? idx}><td><strong>#{rep.rep_number ?? rep.rep ?? idx + 1}</strong></td><td>{formatNumber(rep.start_time ?? rep.start)}s</td><td>{formatNumber(rep.bottom_time ?? rep.bottom)}s</td><td>{formatNumber(rep.end_time ?? rep.end)}s</td><td>{formatNumber(rep.duration ?? rep.rep_duration ?? rep.rep_time)}s</td><td>{formatNumber(rep.bottom_angle ?? rep.bottom_knee_angle ?? rep.minimum_angle ?? rep.bottom_elbow_angle ?? rep.angle)}°</td><td>{formatNumber(rep.rom ?? rep.knee_rom ?? rep.elbow_rom ?? rep.range_of_motion)}°</td></tr>)}
                    </tbody></table></div>
                  ) : <div className="empty-state">No repetition details were returned for this analysis.</div>}
                </div>
              ) : isSprint ? (
                <div className="result-card">
                  <CardTitle
                    title="Sprint / Running Summary"
                    subtitle="Detailed movement measurements from the running sequence"
                  />

                  <div className="metrics-grid">
                    <MetricCard
                      label="Knee Movement"
                      value={`${formatNumber(metrics.knee_movement?.average_rom)}°`}
                    />
                    <MetricCard
                      label="Knee Symmetry Difference"
                      value={
                        metrics.knee_movement?.symmetry_available === false
                          ? "Unavailable"
                          : `${formatNumber(metrics.knee_movement?.symmetry_difference)}°`
                      }
                    />
                    <MetricCard
                      label="Hip Movement"
                      value={`${formatNumber(metrics.hip_movement?.average_rom)}°`}
                    />
                    <MetricCard
                      label="Hip Symmetry Difference"
                      value={
                        metrics.hip_movement?.symmetry_available === false
                          ? "Unavailable"
                          : `${formatNumber(metrics.hip_movement?.symmetry_difference)}°`
                      }
                    />
                    <MetricCard
                      label="Trunk Movement"
                      value={`${formatNumber(metrics.trunk_movement?.average_rom)}°`}
                    />
                    <MetricCard
                      label="Knee Angle Variability"
                      value={`${formatNumber(metrics.knee_angle_variability)}°`}
                    />
                  </div>
                </div>
              ) : (
                <div className="result-card">
                  <CardTitle
                    title="Vertical Jump Summary"
                    subtitle="Key events detected during the jump"
                  />

                  <div className="metrics-grid">
                    <MetricCard
                      label="Takeoff Time"
                      value={`${formatNumber(
                        takeoffTime
                      )} s`}
                    />

                    <MetricCard
                      label="Landing Time"
                      value={`${formatNumber(
                        landingTime
                      )} s`}
                    />

                    <MetricCard
                      label="Takeoff Knee Angle"
                      value={`${formatNumber(
                        metrics.takeoff_knee_angle
                      )}°`}
                    />

                    <MetricCard
                      label="Landing Knee Angle"
                      value={`${formatNumber(
                        metrics.landing_knee_angle
                      )}°`}
                    />
                  </div>
                </div>
              )}

              <div className="result-card feedback-card">
                <CardTitle
                  title="AI Performance Feedback"
                  subtitle="Rule-based observations generated from the current prototype"
                />

                {feedback.length > 0 ? (
                  <div className="feedback-list">
                    {feedback.map(
                      (item, index) => (
                        <div
                          className="feedback-item"
                          key={`${item}-${index}`}
                        >
                          <span>✓</span>
                          <p>{item}</p>
                        </div>
                      )
                    )}
                  </div>
                ) : (
                  <div className="empty-state">
                    No feedback was generated.
                  </div>
                )}
              </div>

              <div className="prototype-note">
                <strong>
                  {isVerticalJump ? "Assessment Status" : "Prototype assessment notice"}
                </strong>

                <p>
                  {isVerticalJump
                    ? "Vertical Jump analysis uses computer-vision-based pose estimation to extract movement measurements. Jump height is calculated using detected flight time and a physics-based estimation method. Machine-learning-based prediction and validated performance scoring are planned for future development."
                    : `The current ${isSprint ? "sprint / running" : isPushup ? "push-up" : isLunge ? "lunge" : "squat"} score is generated from predefined movement rules and extracted video features. It is a project prototype, not a scientifically validated measure of athletic ability.`}
                </p>
              </div>
            </div>
          </section>
        )}

        <section className="how-section">
          <div className="container">
            <div className="center-heading">
              <div className="section-kicker">
                THE PIPELINE
              </div>

              <h2>How the AI analysis works</h2>

              <p>
                From your video to measurable movement
                information.
              </p>
            </div>

            <div className="pipeline">
              <PipelineStep
                number="01"
                title="Upload video"
                text="Provide a clear sports movement video."
              />

              <PipelineStep
                number="02"
                title="Process frames"
                text="OpenCV reads and prepares the video frames."
              />

              <PipelineStep
                number="03"
                title="Track pose"
                text="MediaPipe identifies body landmarks and joints."
              />

              <PipelineStep
                number="04"
                title="Extract features"
                text="Angles, timing, movement features and symmetry are calculated."
              />

              <PipelineStep
                number="05"
                title="Assess movement"
                text="The current prototype applies activity-specific assessment rules."
              />
            </div>
          </div>
        </section>
          </>
        )}
      </main>

      <footer className="footer">
        <div className="container footer-inner">
          <div>
            <strong>SportsAI</strong>

            <span>
              AI Sports Talent Assessment &
              Performance Analysis Platform
            </span>
          </div>

          <span>
            Final-Year B.Tech CSE AI/ML Project
          </span>
        </div>
      </footer>
    </div>
  );
}

const EXERCISE_GUIDES = {
  squat: {
    title: "Squat", category: "Lower-body strength", icon: "🏋️", color: "blue",
    summary: "Review squat movement, repetition consistency and lower-body joint angles from a side-view video.",
    steps: ["Stand with feet around shoulder width and keep a comfortable stance.", "Begin the movement by bending your hips and knees together.", "Lower only as far as you can while keeping balance and control.", "Return to standing smoothly without rushing the movement."],
    checkpoints: ["Keep the full body and feet in frame", "Use a steady side-view camera", "Capture 5–10 complete repetitions", "Use clear lighting and an uncluttered background"],
    mistakes: ["Feet or head cropped out", "Camera moving during recording", "Another person blocking the athlete", "Very dark or blurred footage"],
    recording: "Record 5–10 complete squats from the side with your full body visible."
  },
  sprint: {
    title: "Sprint", category: "Running mechanics", icon: "🏃", color: "orange",
    summary: "Analyze a running sequence and review available knee and hip movement indicators.",
    steps: ["Choose a clear, open running area.", "Position the camera to the side of the running path.", "Keep the runner visible throughout the recorded sequence.", "Record a continuous sequence without panning the camera."],
    checkpoints: ["One clearly visible runner", "Head, hips, knees and feet in frame", "Stable side-view recording", "Enough space to capture the running sequence"],
    mistakes: ["Runner leaving the frame", "Camera following or shaking", "Other people obscuring the runner", "Body joints too small or blurry to detect"],
    recording: "Record one continuous running sequence from the side. Only use a suitable open area and normal, comfortable activity."
  },
  "vertical-jump": {
    title: "Vertical Jump", category: "Explosive movement", icon: "⬆️", color: "purple",
    summary: "Estimate jump timing and available movement measurements from a clear side-view recording.",
    steps: ["Position the camera to the side so the whole movement is visible.", "Start with the athlete fully inside the frame.", "Capture one complete jump, including the takeoff and landing phases.", "Keep the camera still until the movement is complete."],
    checkpoints: ["Both feet visible before and after the jump", "Full body stays inside the frame", "Side view with good lighting", "Uninterrupted recording"],
    mistakes: ["Feet hidden at takeoff or landing", "Camera moving during the jump", "Jump partly outside the frame", "Blurred or low-light footage"],
    recording: "Record one complete vertical jump from the side, keeping the feet and full body visible."
  },
  "push-up": {
    title: "Push-up", category: "Upper-body strength", icon: "💪", color: "green",
    summary: "Review push-up repetitions, elbow movement and body-alignment measurements.",
    steps: ["Place the camera at a side angle with the full body visible.", "Begin in a stable starting position.", "Capture the lowering and return phases of each repetition.", "Keep the recording continuous and the camera still."],
    checkpoints: ["Head, shoulders, hips and feet visible", "Both arms visible where possible", "Several complete repetitions", "Stable side-view camera"],
    mistakes: ["Hands or feet cropped out", "Only part of each repetition recorded", "Camera placed too close", "Arms hidden behind the body"],
    recording: "Record several complete push-ups from the side, keeping the whole body and arms visible."
  },
  lunge: {
    title: "Lunge", category: "Lower-body movement", icon: "🦵", color: "pink",
    summary: "Review lunge repetitions and available knee, torso and consistency measurements.",
    steps: ["Stand in a clear area with enough room to move.", "Keep the camera to the side or at a slight angle so both legs can be seen.", "Capture the lowering and return phases of each repetition.", "Keep the camera fixed throughout the recording."],
    checkpoints: ["Both legs and feet visible", "Hips and torso remain in frame", "Several complete repetitions", "Good lighting and stable camera"],
    mistakes: ["Front or rear foot cropped out", "Legs obscured by objects", "Camera moving during the exercise", "Only partial repetitions recorded"],
    recording: "Record several complete lunges with both legs and your full body visible."
  }
};

function ComparisonWorkspace({ exercise, setExercise, mode, setMode, videoA, setVideoA, videoB, setVideoB, loading, error, result, onCompare, getVideoUrl, formatNumber, onNewComparison }) {
  return (
    <section className="comparison-page">
      <div className="container">
        <div className="comparison-heading">
          <div><div className="section-kicker">VIDEO COMPARISON TOOL</div><h1>Compare Two Videos</h1><p>Compare two recordings using the selected exercise's existing analysis pipeline. Results are limited to movement visible in both recordings.</p></div>
          <span className="comparison-exercise-tag">{exercise.toUpperCase()}</span>
        </div>
        <div className="comparison-exercise-picker">
          <label htmlFor="comparison-exercise">Exercise</label>
          <select id="comparison-exercise" value={exercise} onChange={(event) => setExercise(event.target.value)}>
            <option value="squat">Squat</option>
            <option value="sprint">Sprint</option>
            <option value="vertical-jump">Vertical Jump</option>
            <option value="pushup">Push-up</option>
            <option value="lunge">Lunge</option>
          </select>
          <span>
            {exercise === "vertical-jump"
              ? "Compares jump measurements and normalized knee-angle curves."
              : exercise === "pushup"
              ? "Compares Push-up assessment metrics and individually aligned repetition curves."
              : exercise === "lunge"
              ? "Compares Lunge assessment metrics and normalized knee-angle repetition curves."
              : "Choose an exercise to compare two recordings."}
          </span>
        </div>
        {exercise === "vertical-jump" && <div className="comparison-exercise-note">Vertical Jump curves are aligned by normalized recording time (0–100%), not by confirmed jump phases. Estimated jump height and flight time are experimental measurements.</div>}
        {exercise === "pushup" && <div className="comparison-exercise-note">Push-up movement curves are built by aligning detected repetitions individually from start to end. Scores should be interpreted alongside assessment coverage and measurement availability.</div>}
        {exercise === "lunge" && <div className="comparison-exercise-note comparison-lunge-note">Lunge curves compare average knee-angle movement across individually normalized repetitions. Check selected-side consistency, pose coverage and assessment status when interpreting the result.</div>}
        <div className="comparison-mode-switch" role="group" aria-label="Comparison mode">
          <button type="button" className={mode === "progress" ? "comparison-mode-active" : ""} onClick={() => setMode("progress")}><strong>Progress Comparison</strong><span>Compare an earlier and a newer recording from the same athlete</span></button>
          <button type="button" className={mode === "reference" ? "comparison-mode-active" : ""} onClick={() => setMode("reference")}><strong>Reference Comparison</strong><span>Compare your recording with a coach or reference video</span></button>
        </div>
        <form className="comparison-upload-grid" onSubmit={onCompare}>
          <ComparisonUpload title={mode === "progress" ? "Earlier video" : "Your video"} subtitle="Video A" file={videoA} onChange={setVideoA} />
          <ComparisonUpload title={mode === "progress" ? "Newer video" : "Reference video"} subtitle="Video B" file={videoB} onChange={setVideoB} />
          <div className="comparison-submit-row">
            {error && <div className="comparison-error">{error}</div>}
            <button type="submit" className="comparison-primary-button" disabled={loading || !videoA || !videoB}>{loading ? "Analyzing both videos…" : "Compare Videos"}</button>
          </div>
        </form>
        {result && <ComparisonReport data={result} getVideoUrl={getVideoUrl} formatNumber={formatNumber} />}
        {result && <div className="comparison-new-row"><button type="button" className="comparison-secondary-button" onClick={onNewComparison}>Start a new comparison</button></div>}
      </div>
    </section>
  );
}

function ComparisonUpload({ title, subtitle, file, onChange }) {
  return (
    <div className="comparison-upload-card">
      <div className="comparison-upload-top"><span>{subtitle}</span><strong>{title}</strong></div>
      <label className="comparison-file-drop">
        <span className="comparison-upload-icon">↑</span>
        <strong>{file ? file.name : "Choose a video"}</strong>
        <small>MP4, MOV, AVI or other supported video format</small>
        <input type="file" accept="video/*" onChange={(event) => onChange(event.target.files?.[0] || null)} />
      </label>
      {file && <div className="comparison-file-meta">Selected · {(file.size / (1024 * 1024)).toFixed(1)} MB</div>}
    </div>
  );
}

function ComparisonReport({ data = {}, getVideoUrl, formatNumber, historical = false, createdAt, onBack }) {
  const root = data?.comparison || data?.result?.comparison || data;
  const videoA = root.video_a || root.videoA || data.video_a || {};
  const videoB = root.video_b || root.videoB || data.video_b || {};
  const phase = root.phase_alignment || root.comparison?.phase_alignment || {};
  const movementCurve = root.movement_curve_comparison || {};
  const metricSource = root.metrics || root.metric_comparisons || root.report?.metrics || root.comparison?.metrics || [];
  const metricRows = Array.isArray(metricSource)
    ? metricSource
    : Object.entries(metricSource).map(([key, value]) => ({ metric: key, ...(typeof value === "object" ? value : { value }) }));
  const observations = root.observations || root.report?.observations || root.insights || root.report?.insights || [];
  const limitations = root.limitations || root.report?.limitations || [];
  const toTextList = (items) => Array.isArray(items) ? items : items ? [items] : [];
  const aVideoUrl = getVideoUrl(videoA.processed_video_url || videoA.processed_video || videoA.output_video_url || videoA.video_url || root.video_a_processed_url);
  const bVideoUrl = getVideoUrl(videoB.processed_video_url || videoB.processed_video || videoB.output_video_url || videoB.video_url || root.video_b_processed_url);

  const curveSource = phase.aligned_curve || phase.aligned_curves || phase.aligned_comparison || phase.curve || root.curve_comparison?.aligned_curve || root.curve_comparison?.points || [];
  let curveRows = Array.isArray(curveSource) ? curveSource : (Array.isArray(curveSource?.points) ? curveSource.points : []);

  // Push-up returns point objects; Lunge returns numeric arrays plus
  // a separate normalized 0–100% progress array. Convert both formats
  // into the common chart structure.
  if (!curveRows.length && movementCurve.available) {
    const aCurve = movementCurve.video_a || [];
    const bCurve = movementCurve.video_b || [];
    const progressValues = movementCurve.progress || [];
    if (Array.isArray(aCurve) && Array.isArray(bCurve)) {
      curveRows = Array.from({ length: Math.max(aCurve.length, bCurve.length) }, (_, index) => {
        const pointA = aCurve[index];
        const pointB = bCurve[index];
        const isObjectA = pointA !== null && typeof pointA === "object";
        const isObjectB = pointB !== null && typeof pointB === "object";
        return {
          phase_percent: progressValues[index] ?? pointA?.phase_percent ?? pointB?.phase_percent ?? index,
          video_a: isObjectA
            ? (pointA.elbow_angle ?? pointA.knee_angle ?? pointA.value ?? null)
            : pointA ?? null,
          video_b: isObjectB
            ? (pointB.elbow_angle ?? pointB.knee_angle ?? pointB.value ?? null)
            : pointB ?? null,
        };
      });
    }
  }

  if (!curveRows.length) {
    const aCurve = phase.video_a_curve || phase.curve_a || phase.video_a?.aligned_curve || [];
    const bCurve = phase.video_b_curve || phase.curve_b || phase.video_b?.aligned_curve || [];
    if (Array.isArray(aCurve) && Array.isArray(bCurve)) {
      curveRows = Array.from({ length: Math.max(aCurve.length, bCurve.length) }, (_, index) => ({ progress: index, video_a: aCurve[index], video_b: bCurve[index] }));
    }
  }

  const chartRows = curveRows.map((point, index) => ({
    progress: Number(point.phase_percent ?? point.progress_percent ?? point.progress ?? point.percent ?? point.x ?? index),
    video_a: numberOrNull(point.video_a ?? point.a ?? point.value_a ?? point.knee_angle_a ?? point.curve_a),
    video_b: numberOrNull(point.video_b ?? point.b ?? point.value_b ?? point.knee_angle_b ?? point.curve_b),
    difference: numberOrNull(point.difference ?? point.delta),
  })).filter((point) => Number.isFinite(point.progress) && (point.video_a !== null || point.video_b !== null));

  const assessmentA = videoA.assessment || data.video_a?.assessment || root.assessment_a || {};
  const assessmentB = videoB.assessment || data.video_b?.assessment || root.assessment_b || {};
  const nameA = videoA.filename || videoA.file_name || data.video_a?.filename || root.video_a_filename || "Video A";
  const nameB = videoB.filename || videoB.file_name || data.video_b?.filename || root.video_b_filename || "Video B";
  const scoreA = videoA.performance_score ?? assessmentA.performance_score ?? assessmentA.overall_score ?? assessmentA.score;
  const scoreB = videoB.performance_score ?? assessmentB.performance_score ?? assessmentB.overall_score ?? assessmentB.score;
  const exerciseName = String(root.exercise || data.exercise || "").toLowerCase();
  const isLungeComparison = exerciseName.includes("lunge");
  const isPushupComparison = exerciseName.includes("push");
  const coverageA = videoA.coverage ?? videoA.score_coverage_percent ?? assessmentA.coverage ?? assessmentA.score_coverage_percent;
  const coverageB = videoB.coverage ?? videoB.score_coverage_percent ?? assessmentB.coverage ?? assessmentB.score_coverage_percent;
  const repsA = videoA.detected_repetitions ?? videoA.repetitions ?? assessmentA.detected_repetitions ?? assessmentA.repetitions;
  const repsB = videoB.detected_repetitions ?? videoB.repetitions ?? assessmentB.detected_repetitions ?? assessmentB.repetitions;
  const curveAvailable = Boolean(movementCurve.available || phase.available);
  const curveTitle = isLungeComparison
    ? "Lunge Knee-Angle Curve Comparison"
    : isPushupComparison
    ? "Push-up Repetition Curve Comparison"
    : "Movement Phase Alignment";
  const curveDescription = movementCurve.description || movementCurve.explanation || phase.alignment_method || "Movement curves normalized to movement progress";

  return (
    <div className="comparison-report">
      <div className="comparison-report-heading"><div><span className="section-kicker">{historical ? "SAVED COMPARISON" : "COMPARISON RESULTS"}</span><h2>{root.exercise || data.exercise || "Squat"} {root.comparison_mode === "reference" || data.comparison_mode === "reference" ? "Reference Comparison" : "Progress Comparison"}</h2><p>{createdAt || "Both videos were processed independently using the selected exercise pipeline."}</p></div>{onBack && <button type="button" className="comparison-secondary-button" onClick={onBack}>← Back to history</button>}</div>
      <div className="comparison-video-grid">
        <ComparisonVideo title={videoA.label || "Video A"} name={nameA} src={aVideoUrl} score={scoreA} coverage={coverageA} repetitions={repsA} />
        <ComparisonVideo title={videoB.label || "Video B"} name={nameB} src={bVideoUrl} score={scoreB} coverage={coverageB} repetitions={repsB} />
      </div>
      <div className="comparison-section-card">
        <div className="comparison-section-title"><div><h3>Metric-by-Metric Comparison</h3><p>Interpretations are metric-aware; an increase is not automatically an improvement.</p></div></div>
        {metricRows.length ? <div className="comparison-metrics-table-wrap"><table className="comparison-metrics-table"><thead><tr><th>Metric</th><th>Video A</th><th>Video B</th><th>Difference</th><th>Interpretation</th></tr></thead><tbody>{metricRows.map((row, index) => <tr key={`${row.metric || row.key || row.name || index}`}><td><strong>{row.label || row.metric_label || (row.metric || row.name || row.key || `Metric ${index + 1}`).replaceAll("_", " ")}</strong></td><td>{displayMetric(row.video_a ?? row.value_a ?? row.a ?? row.baseline, row.unit)}</td><td>{displayMetric(row.video_b ?? row.value_b ?? row.b ?? row.current, row.unit)}</td><td>{displayMetric(row.difference ?? row.delta ?? row.change, row.unit)}</td><td>{row.interpretation || row.meaning || row.summary || "Compare the values alongside the processed videos."}</td></tr>)}</tbody></table></div> : <div className="comparison-empty">Metric details were not included in this response.</div>}
      </div>
      <div className="comparison-section-card">
        <div className="comparison-section-title"><div><h3>{curveTitle}</h3><p>{curveDescription}</p></div><span className={`comparison-status ${curveAvailable ? "is-available" : ""}`}>{curveAvailable ? "Available" : "Not available"}</span></div>
        {chartRows.length ? <div className="comparison-chart-wrap"><ResponsiveContainer width="100%" height={300}><LineChart data={chartRows}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="progress" type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} /><YAxis domain={["auto", "auto"]} /><Tooltip labelFormatter={(value) => `${value}% movement progress`} /><Line type="monotone" dataKey="video_a" name="Video A" stroke="#2563eb" strokeWidth={2.5} dot={false} connectNulls={false} /><Line type="monotone" dataKey="video_b" name="Video B" stroke="#ea580c" strokeWidth={2.5} dot={false} connectNulls={false} /></LineChart></ResponsiveContainer></div> : <div className="comparison-empty">No comparable movement curve was available for both videos.</div>}
        {movementCurve.explanation && <p className="comparison-chart-note">{movementCurve.explanation}</p>}
        {movementCurve.mean_absolute_difference !== null && movementCurve.mean_absolute_difference !== undefined && <div className="comparison-attempt-summary"><span>Mean absolute {isLungeComparison ? "knee-angle" : isPushupComparison ? "elbow-angle" : "movement-curve"} difference: <strong>{Number(movementCurve.mean_absolute_difference).toFixed(2)}°</strong></span>{movementCurve.repetitions_used && <span>Repetitions used — Video A: <strong>{movementCurve.repetitions_used.video_a ?? "—"}</strong>, Video B: <strong>{movementCurve.repetitions_used.video_b ?? "—"}</strong></span>}</div>}
        {phase.message && <p className="comparison-chart-note">{phase.message}</p>}
        {phase.attempt_counts && <div className="comparison-attempt-summary"><span>Detected attempts — Video A: <strong>{phase.attempt_counts.video_a ?? "—"}</strong></span><span>Video B: <strong>{phase.attempt_counts.video_b ?? "—"}</strong></span><span>Complete attempts — A: <strong>{phase.complete_attempts?.video_a ?? "—"}</strong></span><span>B: <strong>{phase.complete_attempts?.video_b ?? "—"}</strong></span></div>}
        {phase.alignment_method && <p className="comparison-chart-note">Alignment method: {phase.alignment_method}. Missing portions indicate movement that was not observed in the source video.</p>}
      </div>
      <div className="comparison-insights-grid">
        <div className="comparison-section-card"><h3>Observations</h3>{toTextList(observations).length ? <ul>{toTextList(observations).map((item, index) => <li key={index}>{typeof item === "string" ? item : item.text || item.description || JSON.stringify(item)}</li>)}</ul> : <p className="comparison-muted">No additional observations were returned.</p>}</div>
        <div className="comparison-section-card"><h3>Limitations</h3>{toTextList(limitations).length ? <ul>{toTextList(limitations).map((item, index) => <li key={index}>{typeof item === "string" ? item : item.text || item.description || JSON.stringify(item)}</li>)}</ul> : <p className="comparison-muted">Review video quality, camera position and visible movement when interpreting results.</p>}</div>
      </div>
      {(root.comparison_id || data.comparison_id || data.analysis_id) && <div className="comparison-record-id">Comparison ID: {root.comparison_id || data.comparison_id || data.analysis_id}</div>}
    </div>
  );
}
function ComparisonVideo({ title, name, src, score, coverage, repetitions }) {
  return (
    <article className="comparison-video-card">
      <div className="comparison-video-heading">
        <div><span>{title}</span><strong title={name}>{name}</strong></div>
        <small>{score === null || score === undefined ? "Score unavailable" : `${Number(score).toFixed(2)}/100`}</small>
      </div>
      {(coverage !== null && coverage !== undefined || repetitions !== null && repetitions !== undefined) && (
        <div className="comparison-video-meta">
          {repetitions !== null && repetitions !== undefined && <span>Repetitions: <strong>{repetitions}</strong></span>}
          {coverage !== null && coverage !== undefined && <span>Assessment coverage: <strong>{coverage}%</strong></span>}
        </div>
      )}
      {src ? <video src={src} controls playsInline preload="metadata" /> : <div className="comparison-video-unavailable">Processed video unavailable</div>}
    </article>
  );
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function displayMetric(value, unit = "") {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return `${Number(value).toFixed(2)}${unit ? ` ${unit}` : ""}`;
  if (typeof value === "object") { const shown = value.value ?? value.display ?? "—"; return `${shown}${unit ? ` ${unit}` : ""}`; }
  return `${String(value)}${unit ? ` ${unit}` : ""}`;
}

function ExerciseLibrary({ selectedExercise, onSelectExercise, onBack, video, onFileChange, onAnalyze, loading, error }) {
  const guide = selectedExercise ? EXERCISE_GUIDES[selectedExercise] : null;
  if (guide) {
    return (
      <section className="exercise-library-page">
        <div className="container">
          <button className="library-back" type="button" onClick={onBack}>← All exercises</button>
          <div className={`exercise-detail-hero exercise-tone-${guide.color}`}>
            <div className="exercise-detail-icon">{guide.icon}</div>
            <div><span className="library-eyebrow">{guide.category}</span><h1>{guide.title} Analysis</h1><p>{guide.summary}</p></div>
            <span className="library-ready">● Analysis available</span>
          </div>
          <div className="exercise-detail-grid">
            <article className="library-panel">
              <div className="library-panel-heading"><span>01</span><div><h2>How to perform</h2><p>General movement guidance</p></div></div>
              <ol className="exercise-instruction-list">{guide.steps.map((step, index) => <li key={step}><span>{String(index + 1).padStart(2, "0")}</span><p>{step}</p></li>)}</ol>
              <div className="library-subsection"><h3>Recording checklist</h3><ul className="library-check-list">{guide.checkpoints.map((item) => <li key={item}>✓ {item}</li>)}</ul></div>
              <div className="library-caution"><strong>Recording tips</strong><ul>{guide.mistakes.map((item) => <li key={item}>{item}</li>)}</ul></div>
            </article>
            <article className="library-panel library-upload-panel">
              <div className="library-panel-heading"><span>02</span><div><h2>Upload your video</h2><p>Start an analysis for {guide.title}</p></div></div>
              <div className="library-upload-drop">
                <div className="library-upload-icon">↑</div>
                <strong>{video ? video.name : "Choose a video to analyze"}</strong>
                <p>{video ? `${(video.size / (1024 * 1024)).toFixed(1)} MB · Ready to upload` : guide.recording}</p>
                <label className="library-file-button">Browse video<input type="file" accept="video/*" onChange={onFileChange} /></label>
              </div>
              {error && <div className="library-error">{error}</div>}
              <button type="button" className="library-analyze-button" onClick={onAnalyze} disabled={!video || loading}>{loading ? "Analyzing video…" : "Analyze " + guide.title + " video →"}</button>
              <p className="library-privacy-note">Use a clear recording where the movement and relevant joints remain visible. Results depend on video quality and are intended as informational feedback.</p>
              <button type="button" className="library-change-exercise" onClick={onBack}>Choose a different exercise</button>
            </article>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="exercise-library-page">
      <div className="container">
        <div className="library-heading"><div><span className="library-eyebrow">TRAINING & PERFORMANCE</span><h1>Start a New Analysis</h1><p>Choose an exercise to view movement guidance and upload your video.</p></div><div className="library-count"><strong>05</strong><span>ACTIVITIES</span></div></div>
        <div className="exercise-library-grid">
          {Object.entries(EXERCISE_GUIDES).map(([key, item], index) => (
            <article className={`exercise-library-card exercise-tone-${item.color}`} key={key}>
              <div className="exercise-card-top"><span className="exercise-card-number">0{index + 1}</span><span className="exercise-card-icon">{item.icon}</span></div>
              <span className="exercise-card-category">{item.category}</span>
              <h2>{item.title}</h2><p>{item.summary}</p>
              <div className="exercise-card-footer"><span>Video analysis available</span><button type="button" onClick={() => onSelectExercise(key)}>View exercise <span>→</span></button></div>
            </article>
          ))}
        </div>
        <div className="library-bottom-note"><span>ⓘ</span><p>For useful analysis, record the movement clearly with a steady camera and the relevant body joints visible. AI-generated measurements may be affected by camera angle, lighting and occlusion.</p></div>
      </div>
    </section>
  );
}

function AthleteDashboard({ history, loading, error, onRefresh, onNewAnalysis, onOpenHistory, formatDate }) {
  const scoresByExercise = EXERCISES.map((exercise) => {
    const matching = history.filter((record) => normalizeActivity(record.activity) === exercise.key && getRecordScore(record) !== null)
      .sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));
    return { ...exercise, score: matching.length ? getRecordScore(matching[0]) : null, record: matching[0] || null, attempts: matching.length };
  });
  const available = scoresByExercise.filter((item) => item.score !== null);
  const scoredRecords = history.map((record) => ({ record, score: getRecordScore(record) })).filter((item) => item.score !== null);
  const average = scoredRecords.length ? scoredRecords.reduce((sum, item) => sum + item.score, 0) / scoredRecords.length : null;
  const best = available.length ? [...available].sort((a, b) => b.score - a.score)[0] : null;
  const needsFocus = available.length ? [...available].sort((a, b) => a.score - b.score)[0] : null;
  const recent = [...history].sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)).slice(0, 5);
  const radarData = scoresByExercise.filter((item) => item.score !== null).map((item) => ({ exercise: item.label, score: item.score }));
  return (
    <section className="athlete-dashboard-section">
      <div className="container">
        <div className="dashboard-heading">
          <div><div className="section-kicker">ATHLETE OVERVIEW</div><h1>Performance Dashboard</h1><p>Your activity scores, performance snapshot and recent analysis.</p></div>
          <div className="dashboard-actions"><button className="history-refresh" onClick={onRefresh} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button><button className="dashboard-primary" onClick={onNewAnalysis}>＋ New Analysis</button></div>
        </div>
        {error && <div className="history-error">{error}</div>}
        {loading ? <div className="history-empty">Loading athlete profile…</div> : (
          <>
            <div className="dashboard-stat-grid">
              <article className="dashboard-stat"><span>Saved Assessments</span><strong>{history.length}</strong><small>Across all exercises</small></article>
              <article className="dashboard-stat"><span>Average Score</span><strong>{average === null ? "—" : average.toFixed(1)}</strong><small>{scoredRecords.length ? "Based on saved scores" : "Complete an analysis first"}</small></article>
              <article className="dashboard-stat"><span>Top Exercise</span><strong>{best ? best.label : "—"}</strong><small>{best ? `${best.score.toFixed(1)} / 100 latest score` : "No scores yet"}</small></article>
              <article className="dashboard-stat dashboard-stat-focus"><span>Focus Exercise</span><strong>{needsFocus ? needsFocus.label : "—"}</strong><small>{needsFocus ? `${needsFocus.score.toFixed(1)} / 100 latest score` : "No scores yet"}</small></article>
            </div>
            <div className="dashboard-content-grid">
              <article className="dashboard-card radar-card">
                <div className="dashboard-card-heading"><div><h2>Exercise Performance Profile</h2><p>Latest score available for each exercise</p></div></div>
                {radarData.length ? <div className="dashboard-radar-wrap"><ResponsiveContainer width="100%" height={310}><RadarChart data={radarData} outerRadius="72%"><PolarGrid /><PolarAngleAxis dataKey="exercise" tick={{ fontSize: 11, fill: "#64748b" }} /><PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} /><Radar name="Score" dataKey="score" stroke="#2563eb" fill="#60a5fa" fillOpacity={0.3} /></RadarChart></ResponsiveContainer></div> : <div className="dashboard-empty-chart">Your radar chart appears after you complete and save an exercise analysis.</div>}
              </article>
              <article className="dashboard-card exercise-scores-card">
                <div className="dashboard-card-heading"><div><h2>Exercise Scores</h2><p>Most recent result per activity</p></div></div>
                <div className="exercise-score-list">{scoresByExercise.map((item) => <div className="exercise-score-row" key={item.key}><div className="exercise-score-label"><span className="exercise-code">{item.short}</span><strong>{item.label}</strong></div><div className="exercise-score-track"><div style={{ width: `${Math.max(0, Math.min(100, item.score ?? 0))}%` }} /></div><strong className="exercise-score-number">{item.score === null ? "—" : item.score.toFixed(1)}</strong></div>)}</div>
              </article>
            </div>
            <div className="dashboard-content-grid dashboard-bottom-grid">
              <article className="dashboard-card dashboard-insights-card"><div className="dashboard-card-heading"><div><h2>Performance Insights</h2><p>Quick summary from your saved assessment scores</p></div><span className="insights-badge">Auto-generated</span></div>
                {best && needsFocus && available.length > 1 ? <div className="dashboard-summary-insights"><div className="dashboard-summary-item summary-positive"><span>Strength</span><strong>{best.label}</strong><p>Your highest latest exercise score is {best.score.toFixed(1)}.</p></div><div className="dashboard-summary-item summary-focus"><span>Review next</span><strong>{needsFocus.label}</strong><p>This is currently your lowest latest exercise score ({needsFocus.score.toFixed(1)}). Consider repeating its analysis with a consistent recording setup.</p></div></div> : <div className="dashboard-empty-chart">Complete at least two exercise types to unlock cross-exercise strengths and focus insights.</div>}
              </article>
              <article className="dashboard-card recent-card"><div className="dashboard-card-heading"><div><h2>Recent Activity</h2><p>Your latest saved reports</p></div><button className="text-link-button" onClick={onOpenHistory}>View history →</button></div>
                {recent.length ? <div className="recent-list">{recent.map((record) => <div className="recent-row" key={record.analysis_id}><div><strong>{record.activity || "Exercise"}</strong><span>{formatDate(record.created_at)}</span></div><strong className="recent-score">{getRecordScore(record) === null ? "—" : getRecordScore(record).toFixed(1)}</strong></div>)}</div> : <div className="dashboard-empty-chart">No saved analyses yet. Start by uploading an exercise video.</div>}
              </article>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function GuideStep({ number, title, text }) {
  return (
    <div className="guide-step">
      <div className="guide-number">{number}</div>

      <div>
        <h4>{title}</h4>
        <p>{text}</p>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  detail,
  accent,
}) {
  return (
    <div className={`stat-card stat-${accent}`}>
      <div className="stat-top">
        <span>{label}</span>
        <i />
      </div>

      <strong>{value}</strong>

      <small>{detail}</small>
    </div>
  );
}

function CardTitle({ title, subtitle }) {
  return (
    <div className="card-title">
      <div>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function VideoPanel({ title, src }) {
  return (
    <div className="video-panel">
      <div className="video-panel-title">
        {title}
      </div>

      {src ? (
        <video
          src={src}
          controls
          playsInline
        />
      ) : (
        <div className="video-placeholder">
          Video preview unavailable
        </div>
      )}
    </div>
  );
}

function ScoreBar({ label, value }) {
  const unavailable = value === null || value === undefined || value === "";
  const numericValue = Number(value);
  const safeValue = unavailable || !Number.isFinite(numericValue)
    ? 0
    : Math.max(0, Math.min(100, numericValue));

  return (
    <div className={`score-row ${unavailable ? "score-row-unavailable" : ""}`}>
      <div className="score-row-top">
        <span>{label}</span>
        <strong>
          {unavailable ? "Unavailable" : `${Math.round(safeValue)}/100`}
        </strong>
      </div>
      <div className="score-track">
        {!unavailable && (
          <div
            className="score-fill"
            style={{ width: `${safeValue}%` }}
          />
        )}
        {unavailable && <span className="score-unavailable-track" />}
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PipelineStep({
  number,
  title,
  text,
}) {
  return (
    <div className="pipeline-step">
      <div className="pipeline-number">
        {number}
      </div>

      <h3>{title}</h3>

      <p>{text}</p>
    </div>
  );
}

const styles = `
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: #f6f8fc; color: #111827; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  button, input, select { font: inherit; }
  .app-shell { min-height: 100vh; background: #f6f8fc; }
  .container { width: min(1180px, calc(100% - 40px)); margin: 0 auto; }

  .topbar { position: sticky; top: 0; z-index: 20; background: rgba(255,255,255,.94); backdrop-filter: blur(14px); border-bottom: 1px solid #e6eaf0; }
  .topbar-inner { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
  .brand { display: flex; align-items: center; gap: 11px; }
  .brand-mark { width: 40px; height: 40px; display: grid; place-items: center; border-radius: 12px; background: #111827; color: white; font-weight: 900; font-size: 18px; box-shadow: 0 7px 20px rgba(17,24,39,.16); }
  .brand-name { font-size: 17px; font-weight: 850; letter-spacing: -.02em; }
  .brand-subtitle { color: #7b8494; font-size: 11px; margin-top: 2px; }
  .topbar-badge { color: #667085; font-size: 12px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .topbar-actions { display: flex; align-items: center; gap: 10px; }
  .nav-button { border: 1px solid #e2e8f0; background: white; color: #475467; padding: 10px 14px; border-radius: 10px; font-size: 11px; font-weight: 800; cursor: pointer; }
  .nav-button:hover, .nav-button-active { border-color: #bfdbfe; background: #eff6ff; color: #1d4ed8; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #16a34a; box-shadow: 0 0 0 4px #dcfce7; }

  .hero { background: white; border-bottom: 1px solid #e9edf3; padding: 72px 0 76px; }
  .hero-grid { display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 70px; align-items: end; }
  .eyebrow, .section-kicker { color: #2563eb; font-size: 11px; letter-spacing: .14em; font-weight: 850; }
  .hero h1 { margin: 13px 0 18px; font-size: clamp(42px, 5vw, 68px); line-height: .98; letter-spacing: -.055em; color: #0f172a; }
  .hero h1 span { color: #64748b; }
  .hero-copy > p { max-width: 690px; margin: 0; color: #667085; font-size: 17px; line-height: 1.7; }
  .hero-points { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 25px; }
  .hero-points span { padding: 9px 13px; border: 1px solid #e5eaf1; background: #f8fafc; border-radius: 999px; color: #475467; font-size: 12px; font-weight: 700; }
  .hero-mini-card { border: 1px solid #e3e8f0; border-radius: 20px; padding: 20px; background: #fbfcfe; display: flex; gap: 14px; box-shadow: 0 12px 35px rgba(15,23,42,.05); }
  .mini-icon { width: 38px; height: 38px; flex: 0 0 auto; border-radius: 11px; background: #eff6ff; color: #2563eb; display: grid; place-items: center; font-size: 11px; font-weight: 900; }
  .hero-mini-card strong { display: block; font-size: 14px; color: #111827; }
  .hero-mini-card span { display: block; margin-top: 6px; color: #7b8494; font-size: 12px; line-height: 1.5; }

  .analysis-section { padding: 68px 0 78px; }
  .section-heading, .results-header { display: flex; justify-content: space-between; align-items: end; gap: 30px; margin-bottom: 28px; }
  .section-heading h2, .results-header h2, .center-heading h2 { margin: 7px 0 7px; font-size: 30px; letter-spacing: -.035em; color: #101828; }
  .section-heading p, .results-header p, .center-heading p { margin: 0; color: #667085; font-size: 14px; }
  .step-indicator { display: flex; align-items: center; gap: 9px; padding-bottom: 4px; }
  .step-indicator span { width: 29px; height: 29px; border-radius: 50%; display: grid; place-items: center; background: #eef2f7; color: #8a94a6; font-size: 11px; font-weight: 850; }
  .step-indicator span.active { background: #111827; color: white; }
  .step-indicator i { width: 34px; height: 1px; background: #dfe4eb; }

  .analysis-layout { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(380px, .92fr); gap: 22px; align-items: stretch; }
  .guidelines-card, .upload-card, .result-card, .stat-card { background: white; border: 1px solid #e2e7ee; box-shadow: 0 15px 40px rgba(15,23,42,.045); }
  .guidelines-card { border-radius: 22px; padding: 26px; }
  .card-heading { display: flex; align-items: center; gap: 13px; padding-bottom: 20px; border-bottom: 1px solid #edf0f4; }
  .heading-icon { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; background: #eef6ff; color: #2563eb; font-size: 10px; font-weight: 900; letter-spacing: .06em; }
  .card-heading h3, .upload-top h3 { margin: 0; font-size: 18px; letter-spacing: -.02em; }
  .card-heading p, .upload-top p { margin: 5px 0 0; color: #7b8494; font-size: 12px; line-height: 1.5; }
  .guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 20px; }
  .guide-step { display: flex; gap: 11px; padding: 15px; border: 1px solid #e8ecf2; background: #fbfcfe; border-radius: 15px; min-height: 94px; }
  .guide-number { width: 27px; height: 27px; flex: 0 0 auto; display: grid; place-items: center; border-radius: 8px; background: #f0f4f9; color: #475467; font-size: 9px; font-weight: 900; }
  .guide-step h4 { margin: 1px 0 5px; font-size: 13px; color: #172033; }
  .guide-step p { margin: 0; color: #667085; font-size: 11px; line-height: 1.55; }
  .guide-good { margin-top: 15px; padding: 16px 17px; border-radius: 15px; background: #f0fdf4; border: 1px solid #bbf7d0; }
  .guide-good-title { color: #166534; font-weight: 850; font-size: 12px; }
  .guide-good-list { display: grid; grid-template-columns: 1fr 1fr; gap: 7px 18px; margin-top: 10px; color: #287447; font-size: 11px; }
  .guide-good-list span::before { content: "•"; margin-right: 7px; }
  .guide-warning { display: flex; gap: 12px; margin-top: 12px; padding: 14px 16px; border-radius: 15px; background: #fff8ef; border: 1px solid #fed7aa; }
  .warning-icon { width: 25px; height: 25px; flex: 0 0 auto; border-radius: 50%; background: #ffedd5; color: #c2410c; display: grid; place-items: center; font-weight: 900; font-size: 12px; }
  .guide-warning strong { color: #9a3412; font-size: 12px; }
  .guide-warning p { margin: 4px 0 0; color: #b45309; font-size: 11px; line-height: 1.5; }

  .upload-card { border-radius: 22px; padding: 28px; display: flex; flex-direction: column; }
  .upload-top { display: flex; justify-content: space-between; align-items: start; gap: 15px; }
  .available-pill { white-space: nowrap; border: 1px solid #bbf7d0; background: #f0fdf4; color: #15803d; border-radius: 999px; padding: 6px 9px; font-size: 10px; font-weight: 850; }
  .field-label { display: block; margin: 24px 0 8px; color: #344054; font-size: 12px; font-weight: 800; }
  .activity-select { width: 100%; height: 48px; padding: 0 14px; border: 1px solid #d8dee7; border-radius: 12px; background: white; color: #1d2939; outline: none; cursor: pointer; }
  .activity-select:focus { border-color: #93c5fd; box-shadow: 0 0 0 3px #eff6ff; }
  .upload-divider { display: flex; align-items: center; gap: 10px; margin: 23px 0 12px; color: #98a2b3; font-size: 10px; font-weight: 850; letter-spacing: .08em; }
  .upload-divider::after { content: ""; height: 1px; background: #edf0f4; flex: 1; }
  .upload-zone { position: relative; border: 1.5px dashed #b9c4d4; border-radius: 16px; background: #fbfcfe; min-height: 155px; overflow: hidden; transition: .2s ease; }
  .upload-zone:hover { border-color: #60a5fa; background: #f8fbff; }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; z-index: 2; }
  .upload-zone-content { min-height: 155px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px; cursor: pointer; }
  .upload-icon { width: 42px; height: 42px; border-radius: 12px; background: #eff6ff; color: #2563eb; display: grid; place-items: center; font-size: 22px; font-weight: 500; margin-bottom: 9px; }
  .upload-zone strong { font-size: 14px; color: #172033; }
  .upload-zone span { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #667085; font-size: 11px; margin-top: 4px; }
  .upload-zone em { color: #2563eb; font-size: 11px; font-style: normal; font-weight: 750; margin-top: 8px; }
  .selected-file { display: flex; align-items: center; gap: 10px; margin-top: 11px; padding: 10px 11px; border: 1px solid #dbeafe; background: #f8fbff; border-radius: 12px; }
  .file-check { width: 26px; height: 26px; border-radius: 50%; background: #dcfce7; color: #15803d; display: grid; place-items: center; font-size: 12px; font-weight: 900; }
  .selected-file-text { min-width: 0; flex: 1; }
  .selected-file-text strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; color: #344054; }
  .selected-file-text span { display: block; margin-top: 2px; font-size: 10px; color: #98a2b3; }
  .remove-file { border: 0; background: transparent; color: #667085; font-size: 10px; font-weight: 750; cursor: pointer; }
  .remove-file:hover { color: #dc2626; }
  .error-box { margin-top: 12px; padding: 11px 12px; border-radius: 11px; background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; font-size: 11px; line-height: 1.5; }
  .analyze-button { width: 100%; min-height: 50px; margin-top: 16px; border: 0; border-radius: 13px; background: #111827; color: white; font-weight: 850; font-size: 13px; cursor: pointer; box-shadow: 0 10px 25px rgba(17,24,39,.13); transition: transform .15s ease, background .15s ease; }
  .analyze-button:hover:not(:disabled) { background: #1f2937; transform: translateY(-1px); }
  .analyze-button:disabled { opacity: .45; cursor: not-allowed; box-shadow: none; }
  .analyze-button span { margin-left: 7px; font-size: 17px; vertical-align: -1px; }
  .spinner { display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,.35); border-top-color: white; border-radius: 50%; animation: spin .7s linear infinite; margin-right: 7px; }
  .privacy-note { margin: 12px 0 0; text-align: center; color: #98a2b3; font-size: 10px; line-height: 1.4; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .history-section { min-height: 72vh; padding: 58px 0 78px; background: #f6f8fc; }
  .history-heading { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 26px; }
  .history-heading h1 { margin: 8px 0; color: #101828; font-size: 34px; letter-spacing: -.04em; }
  .history-heading p { margin: 0; color: #667085; font-size: 13px; }
  .history-refresh, .history-back-button { border: 1px solid #dbe3ee; background: white; color: #344054; padding: 10px 14px; border-radius: 10px; font-size: 11px; font-weight: 800; cursor: pointer; }
  .history-refresh:disabled { opacity: .6; cursor: wait; }
  .history-list { display: grid; gap: 12px; }
  .history-record { display: flex; align-items: center; gap: 18px; padding: 18px; border: 1px solid #e2e7ee; border-radius: 16px; background: white; box-shadow: 0 10px 28px rgba(15,23,42,.035); }
  .history-record-main { display: flex; align-items: center; gap: 13px; min-width: 0; flex: 1; }
  .history-activity-icon { width: 42px; height: 42px; flex: 0 0 auto; display: grid; place-items: center; border-radius: 12px; background: #eff6ff; color: #2563eb; font-size: 11px; font-weight: 900; }
  .history-record-info { min-width: 0; }
  .history-record-info h3 { margin: 0; color: #172033; font-size: 14px; }
  .history-record-info p { margin: 4px 0; color: #667085; font-size: 11px; overflow-wrap: anywhere; }
  .history-record-info span, .history-date { color: #98a2b3; font-size: 10px; }
  .history-record-score { min-width: 95px; text-align: center; }
  .history-record-score strong { display: block; color: #172033; font-size: 22px; }
  .history-record-score span { display: block; margin-top: 3px; color: #15803d; font-size: 10px; font-weight: 800; }
  .history-view-button { border: 0; border-radius: 10px; background: #111827; color: white; padding: 11px 14px; font-size: 11px; font-weight: 800; cursor: pointer; white-space: nowrap; }
  .history-view-button:disabled { opacity: .5; cursor: wait; }
  .history-view-button span { margin-left: 5px; }
  .history-empty { padding: 50px 20px; text-align: center; border: 1px dashed #cbd5e1; border-radius: 16px; background: white; color: #667085; font-size: 13px; }
  .history-error { margin-bottom: 16px; padding: 12px 14px; border: 1px solid #fecaca; border-radius: 11px; background: #fef2f2; color: #b91c1c; font-size: 12px; }
  .history-detail-card { padding: 24px; border: 1px solid #e2e7ee; border-radius: 18px; background: white; box-shadow: 0 12px 32px rgba(15,23,42,.04); }
  .history-detail-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 22px; }
  .history-back-button { padding: 8px 11px; }
  .history-detail-title { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding-bottom: 20px; border-bottom: 1px solid #edf0f4; }
  .history-detail-title h2 { margin: 7px 0 4px; color: #172033; font-size: 25px; }
  .history-detail-title p { margin: 0; color: #667085; font-size: 11px; }
  .history-score-large { min-width: 115px; padding: 13px; text-align: center; border-radius: 14px; background: #f0fdf4; }
  .history-score-large strong { color: #15803d; font-size: 28px; }
  .history-score-large > span { color: #64748b; font-size: 11px; }
  .history-score-large small { display: block; margin-top: 3px; color: #166534; font-size: 10px; font-weight: 800; }
  .history-subheading { margin: 24px 0 13px; color: #172033; font-size: 14px; font-weight: 850; }
  .history-subsection { margin-top: 20px; }
  .history-subsection .history-subheading { margin-top: 0; }
  .history-metrics-grid { margin-top: 0; }
  .history-video-panel { margin-top: 20px; padding: 14px; border: 1px solid #e8ecf2; border-radius: 13px; background: #fbfcfe; }
  .history-video-panel h3 { margin: 0 0 10px; color: #344054; font-size: 12px; }
  .history-video-panel video { display: block; width: min(100%, 720px); max-height: 420px; margin: 0 auto; background: #0f172a; border-radius: 10px; }
  .history-id { margin-top: 22px; padding-top: 14px; border-top: 1px solid #edf0f4; color: #98a2b3; font-size: 10px; overflow-wrap: anywhere; }
  .history-loading-note { margin-top: 12px; color: #667085; font-size: 11px; text-align: center; }

  .results-section { background: #f0f4f8; border-top: 1px solid #e5e9ef; padding: 72px 0 78px; }
  .analysis-id { min-width: 190px; padding: 11px 13px; border: 1px solid #dde3eb; background: white; border-radius: 12px; }
  .analysis-id span { display: block; color: #98a2b3; font-size: 9px; text-transform: uppercase; letter-spacing: .08em; font-weight: 850; }
  .analysis-id strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 4px; color: #475467; font-size: 10px; }
  .no-reps-banner { display: flex; gap: 13px; padding: 15px 17px; border: 1px solid #fed7aa; background: #fffaf2; border-radius: 15px; margin-bottom: 18px; }
  .no-reps-icon { width: 28px; height: 28px; flex: 0 0 auto; display: grid; place-items: center; border-radius: 50%; background: #ffedd5; color: #c2410c; font-weight: 900; }
  .no-reps-banner strong { color: #9a3412; font-size: 12px; }
  .no-reps-banner p { margin: 4px 0 0; color: #b45309; font-size: 11px; line-height: 1.5; }
  .score-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
  .stat-card { border-radius: 17px; padding: 18px; overflow: hidden; position: relative; }
  .stat-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #2563eb; }
  .stat-green::before { background: #16a34a; }
  .stat-purple::before { background: #7c3aed; }
  .stat-orange::before { background: #ea580c; }
  .stat-top { display: flex; align-items: center; justify-content: space-between; color: #667085; font-size: 10px; font-weight: 800; }
  .stat-top i { width: 8px; height: 8px; border-radius: 50%; background: #bfdbfe; }
  .stat-green .stat-top i { background: #bbf7d0; }
  .stat-purple .stat-top i { background: #ddd6fe; }
  .stat-orange .stat-top i { background: #fed7aa; }
  .stat-card strong { display: block; margin-top: 13px; color: #111827; font-size: 25px; letter-spacing: -.04em; }
  .stat-card small { display: block; margin-top: 4px; color: #98a2b3; font-size: 10px; }
  .results-two-column { display: grid; grid-template-columns: 1.3fr .7fr; gap: 18px; margin-bottom: 18px; }
  .result-card { border-radius: 18px; padding: 22px; margin-bottom: 18px; }
  .result-card:last-child { margin-bottom: 0; }
  .card-title { margin-bottom: 18px; }
  .card-title h3 { margin: 0; font-size: 16px; color: #172033; letter-spacing: -.02em; }
  .card-title p { margin: 4px 0 0; color: #98a2b3; font-size: 11px; }
  .video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .video-panel { border: 1px solid #e6eaf0; border-radius: 13px; overflow: hidden; background: #f8fafc; }
  .video-panel-title { padding: 10px 12px; background: white; border-bottom: 1px solid #e6eaf0; font-size: 10px; color: #475467; font-weight: 850; }
  .video-panel video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #0f172a; object-fit: contain; }
  .video-placeholder { aspect-ratio: 16 / 9; display: grid; place-items: center; color: #98a2b3; font-size: 11px; }
  .score-bars { display: grid; gap: 16px; }
  .score-row-top { display: flex; justify-content: space-between; margin-bottom: 7px; color: #475467; font-size: 11px; }
  .score-row-top strong { color: #111827; }
  .score-track { height: 7px; border-radius: 99px; background: #edf1f5; overflow: hidden; }
  .score-fill { height: 100%; border-radius: inherit; background: #2563eb; transition: width .5s ease; }
  .score-unavailable-track { display: block; width: 100%; height: 100%; background: repeating-linear-gradient(135deg, #e5e7eb, #e5e7eb 5px, #f8fafc 5px, #f8fafc 10px); }
  .score-row-unavailable .score-row-top strong { color: #98a2b3; font-size: 10px; }
  .squat-assessment-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 22px; margin: 14px 0 20px; padding: 17px 19px; border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; }
  .squat-summary-item span { display: block; color: #7b8494; font-size: 10px; font-weight: 750; text-transform: uppercase; letter-spacing: .06em; }
  .squat-summary-item strong { display: block; margin-top: 5px; color: #172033; font-size: 18px; }
  .squat-summary-item .status-provisional { color: #b45309; }
  .squat-assessment-summary p { grid-column: 1 / -1; margin: 0; padding-top: 11px; border-top: 1px solid #edf0f4; color: #667085; font-size: 11px; line-height: 1.6; }
  .chart-wrap { width: 100%; height: 330px; }
  .empty-chart, .empty-state { min-height: 130px; display: grid; place-items: center; border: 1px dashed #d7dee8; border-radius: 12px; color: #98a2b3; font-size: 12px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .movement-breakdown-note { margin: 14px 0 0; color: #7b8494; font-size: 10px; line-height: 1.55; }
  .metric-card { border: 1px solid #e8ecf2; border-radius: 12px; padding: 13px; background: #fbfcfe; min-height: 72px; }
  .metric-card span { display: block; color: #98a2b3; font-size: 9px; line-height: 1.35; }
  .metric-card strong { display: block; margin-top: 8px; color: #172033; font-size: 15px; }
  .table-scroll { overflow-x: auto; border: 1px solid #e8ecf2; border-radius: 12px; }
  table { width: 100%; min-width: 760px; border-collapse: collapse; font-size: 10px; }
  th { padding: 11px 12px; text-align: left; background: #f8fafc; color: #667085; font-size: 9px; text-transform: uppercase; letter-spacing: .05em; }
  td { padding: 12px; color: #667085; border-top: 1px solid #edf0f4; }
  td strong { color: #344054; }
  .feedback-list { display: grid; gap: 9px; }
  .feedback-item { display: flex; align-items: start; gap: 10px; padding: 12px 13px; border: 1px solid #e6eaf0; background: #fbfcfe; border-radius: 11px; }
  .feedback-item > span { width: 21px; height: 21px; flex: 0 0 auto; display: grid; place-items: center; border-radius: 50%; background: #dcfce7; color: #15803d; font-size: 10px; font-weight: 900; }
  .feedback-item p { margin: 2px 0 0; color: #475467; font-size: 11px; line-height: 1.5; }
  .prototype-note { margin-top: 18px; padding: 17px 19px; border: 1px solid #c7d2fe; background: #eef2ff; border-radius: 15px; }
  .prototype-note strong { color: #3730a3; font-size: 11px; }
  .prototype-note p { margin: 5px 0 0; color: #4f46a5; font-size: 10px; line-height: 1.6; }

  .how-section { background: white; padding: 72px 0 78px; border-top: 1px solid #e6eaf0; }
  .center-heading { text-align: center; max-width: 600px; margin: 0 auto 38px; }
  .pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .pipeline-step { padding: 20px 17px; border: 1px solid #e5e9ef; border-radius: 16px; background: #fbfcfe; }
  .pipeline-number { color: #2563eb; font-size: 10px; font-weight: 900; letter-spacing: .08em; }
  .pipeline-step h3 { margin: 13px 0 6px; font-size: 13px; color: #172033; }
  .pipeline-step p { margin: 0; color: #7b8494; font-size: 10px; line-height: 1.55; }
  .footer { background: #111827; color: white; padding: 25px 0; }
  .footer-inner { display: flex; justify-content: space-between; gap: 20px; align-items: center; }
  .footer strong { display: block; font-size: 14px; }
  .footer span { display: block; margin-top: 3px; color: #9ca3af; font-size: 10px; }
  .footer-inner > span { margin: 0; text-align: right; }

  @media (max-width: 950px) {
    .hero-grid, .analysis-layout, .results-two-column { grid-template-columns: 1fr; }
    .hero-mini-card { max-width: 420px; }
    .score-grid { grid-template-columns: repeat(2, 1fr); }
    .metrics-grid { grid-template-columns: repeat(3, 1fr); }
    .pipeline { grid-template-columns: repeat(2, 1fr); }
  }

  @media (max-width: 650px) {
    .container { width: min(100% - 24px, 1180px); }
    .topbar-badge { display: none; }
    .topbar-actions { gap: 6px; }
    .nav-button { padding: 8px 9px; font-size: 10px; }
    .history-heading, .history-record, .history-detail-title { align-items: stretch; flex-direction: column; }
    .history-record-score { text-align: left; }
    .history-view-button { width: 100%; }
    .history-detail-card { padding: 17px; }
    .history-section { padding: 42px 0 55px; }
    .hero { padding: 48px 0 52px; }
    .hero h1 { font-size: 43px; }
    .hero-grid { gap: 30px; }
    .section-heading, .results-header { align-items: start; flex-direction: column; }
    .step-indicator { display: none; }
    .guidelines-card, .upload-card, .result-card { padding: 18px; }
    .guide-grid, .video-grid, .score-grid, .metrics-grid, .pipeline { grid-template-columns: 1fr; }
    .guide-good-list { grid-template-columns: 1fr; }
    .analysis-section, .results-section, .how-section { padding: 48px 0; }
    .footer-inner { align-items: start; flex-direction: column; }
    .footer-inner > span { text-align: left; }
    .chart-wrap { height: 270px; }
  }

  .athlete-dashboard-section { min-height: 76vh; padding: 48px 0 72px; background: #f6f8fc; }
  .dashboard-heading { display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:24px; }
  .dashboard-heading h1 { margin:8px 0; color:#101828; font-size:34px; letter-spacing:-.04em; }
  .dashboard-heading p { margin:0; color:#667085; font-size:13px; }
  .dashboard-actions { display:flex; gap:9px; }
  .dashboard-primary { border:0; border-radius:10px; padding:10px 14px; background:#111827; color:#fff; font-size:11px; font-weight:800; cursor:pointer; }
  .dashboard-stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; }
  .dashboard-stat { background:#fff; border:1px solid #e2e7ee; border-radius:16px; padding:19px; box-shadow:0 10px 28px rgba(15,23,42,.035); }
  .dashboard-stat span { display:block; color:#667085; font-size:11px; font-weight:750; }
  .dashboard-stat strong { display:block; margin:10px 0 5px; color:#101828; font-size:25px; letter-spacing:-.03em; }
  .dashboard-stat small { color:#98a2b3; font-size:10px; }
  .dashboard-stat-focus strong { color:#b45309; }
  .dashboard-content-grid { display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr); gap:16px; margin-bottom:16px; }
  .dashboard-card { background:#fff; border:1px solid #e2e7ee; border-radius:17px; padding:20px; box-shadow:0 10px 28px rgba(15,23,42,.035); min-width:0; }
  .dashboard-card-heading { display:flex; align-items:start; justify-content:space-between; gap:12px; margin-bottom:12px; }
  .dashboard-card-heading h2 { margin:0; color:#172033; font-size:16px; letter-spacing:-.02em; }
  .dashboard-card-heading p { margin:5px 0 0; color:#98a2b3; font-size:11px; }
  .dashboard-radar-wrap { width:100%; height:310px; }
  .dashboard-empty-chart { min-height:150px; display:grid; place-items:center; padding:24px; text-align:center; border:1px dashed #d7dee8; border-radius:12px; color:#98a2b3; font-size:12px; line-height:1.5; }
  .exercise-score-list { display:grid; gap:15px; padding:8px 0; }
  .exercise-score-row { display:grid; grid-template-columns:125px minmax(0,1fr) 44px; align-items:center; gap:10px; }
  .exercise-score-label { display:flex; align-items:center; gap:8px; min-width:0; }
  .exercise-score-label strong { color:#344054; font-size:11px; }
  .exercise-code { width:27px; height:27px; display:grid; place-items:center; border-radius:8px; background:#eff6ff; color:#2563eb; font-size:9px; font-weight:900; flex:0 0 auto; }
  .exercise-score-track { height:8px; border-radius:99px; background:#edf1f6; overflow:hidden; }
  .exercise-score-track div { height:100%; border-radius:99px; background:linear-gradient(90deg,#60a5fa,#2563eb); }
  .exercise-score-number { text-align:right; color:#172033; font-size:11px; }
  .dashboard-summary-insights { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .dashboard-summary-item { padding:16px; border-radius:13px; border:1px solid #e5eaf1; background:#fbfcfe; }
  .dashboard-summary-item span { display:block; font-size:10px; font-weight:850; text-transform:uppercase; letter-spacing:.08em; }
  .summary-positive span { color:#15803d; } .summary-focus span { color:#b45309; }
  .dashboard-summary-item strong { display:block; margin:8px 0 5px; color:#172033; font-size:16px; }
  .dashboard-summary-item p { margin:0; color:#667085; font-size:11px; line-height:1.55; }
  .recent-list { display:grid; }
  .recent-row { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:12px 0; border-bottom:1px solid #edf0f4; }
  .recent-row:last-child { border-bottom:0; }
  .recent-row div strong { display:block; color:#344054; font-size:12px; }
  .recent-row div span { display:block; margin-top:4px; color:#98a2b3; font-size:10px; }
  .recent-score { color:#15803d; font-size:14px; }
  .text-link-button { border:0; background:transparent; color:#2563eb; font-size:11px; font-weight:800; cursor:pointer; }
  .insights-panel { margin:18px 0 22px; padding:21px; border:1px solid #dbe5f2; border-radius:18px; background:linear-gradient(135deg,#fff 0%,#f7faff 100%); box-shadow:0 12px 30px rgba(15,23,42,.035); }
  .insights-heading { display:flex; align-items:start; justify-content:space-between; gap:16px; margin-bottom:15px; }
  .insights-heading h3 { margin:6px 0; color:#172033; font-size:19px; letter-spacing:-.025em; }
  .insights-heading p { margin:0; color:#667085; font-size:11px; }
  .insights-badge { white-space:nowrap; padding:6px 9px; border:1px solid #dbeafe; border-radius:99px; background:#eff6ff; color:#2563eb; font-size:9px; font-weight:850; }
  .insights-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
  .insight-card { display:flex; gap:11px; padding:13px; border:1px solid #e5eaf1; border-radius:13px; background:#fff; }
  .insight-icon { width:27px; height:27px; flex:0 0 auto; display:grid; place-items:center; border-radius:8px; background:#f1f5f9; color:#475569; font-size:12px; font-weight:900; }
  .insight-strength .insight-icon { background:#dcfce7; color:#15803d; } .insight-focus .insight-icon { background:#ffedd5; color:#c2410c; }
  .insight-card strong { display:block; color:#344054; font-size:11px; }
  .insight-card p { margin:5px 0 0; color:#667085; font-size:10px; line-height:1.55; }
  @media(max-width:900px) { .dashboard-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .dashboard-content-grid { grid-template-columns:1fr; } }
  @media(max-width:600px) { .dashboard-heading { align-items:stretch; flex-direction:column; } .dashboard-actions { width:100%; } .dashboard-actions button { flex:1; } .dashboard-stat-grid { grid-template-columns:1fr 1fr; gap:9px; } .dashboard-stat { padding:14px; } .dashboard-stat strong { font-size:20px; } .exercise-score-row { grid-template-columns:100px minmax(0,1fr) 35px; gap:7px; } .exercise-score-label strong { font-size:10px; } .dashboard-summary-insights,.insights-grid { grid-template-columns:1fr; } .insights-heading { flex-direction:column; } }
`;

export default App;