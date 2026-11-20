
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000/api";

const API_URL = `${API_BASE}/results`;
const STATUS_URL = `${API_BASE}/tests/status`;
const ML_STATUS_URL = `${API_BASE}/ml/status`;

function App() {
  const [activePage, setActivePage] = useState("Dashboard");

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [testStatus, setTestStatus] = useState({});
  const [startingTest, setStartingTest] = useState("");
  const [testMessage, setTestMessage] = useState("");
  const [testError, setTestError] = useState("");

  const [mlStatus, setMlStatus] = useState(null);
  const [mlLoading, setMlLoading] = useState(true);
  const [mlError, setMlError] = useState("");

  // =========================================================
  // LOAD TEST RESULTS
  // =========================================================

  const loadResults = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL);

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(
        "Backend se connect nahi ho pa raha. Flask server port 5000 par check karo."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // =========================================================
  // LOAD ML MODEL STATUS AND METRICS
  // =========================================================

  const loadMLStatus = useCallback(async () => {
    setMlLoading(true);
    setMlError("");

    try {
      const response = await fetch(ML_STATUS_URL);

      if (!response.ok) {
        throw new Error(`ML API returned status ${response.status}`);
      }

      const data = await response.json();
      setMlStatus(data);
    } catch (err) {
      setMlError("Unable to load ML model information.");
    } finally {
      setMlLoading(false);
    }
  }, []);

  // =========================================================
  // LOAD TEST STATUS
  // =========================================================

  const loadTestStatus = useCallback(async () => {
    try {
      const response = await fetch(STATUS_URL);

      if (!response.ok) {
        throw new Error("Unable to retrieve test status");
      }

      const data = await response.json();
      setTestStatus(data);
    } catch (err) {
      console.error("Test status error:", err);
    }
  }, []);

  // =========================================================
  // INITIAL DATA LOADING
  // =========================================================

  useEffect(() => {
    loadResults();
    loadTestStatus();
    loadMLStatus();
  }, [loadResults, loadTestStatus, loadMLStatus]);

  // =========================================================
  // AUTOMATIC TEST STATUS CHECKING
  // =========================================================

  useEffect(() => {
    const hasRunningTest = Object.values(testStatus).includes("running");

    if (!hasRunningTest) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(STATUS_URL);

        if (!response.ok) return;

        const data = await response.json();

        const wasRunning = Object.values(testStatus).includes("running");
        const isRunning = Object.values(data).includes("running");

        setTestStatus(data);

        if (wasRunning && !isRunning) {
          setTestMessage(
            "Test finished. Refreshing your saved results..."
          );

          loadResults();
        }
      } catch (err) {
        console.error("Status polling error:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [testStatus, loadResults]);

  // =========================================================
  // START WEBCAM TEST
  // =========================================================

  const startTest = async (testName) => {
    setStartingTest(testName);
    setTestMessage("");
    setTestError("");

    try {
      const response = await fetch(
        `${API_BASE}/tests/start/${testName}`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || "Unable to start test.");
      }

      setTestStatus((previous) => ({
        ...previous,
        [testName]: "running",
      }));

      setTestMessage(
        `${
          testName === "eye-blink"
            ? "Eye Blink"
            : "Finger Stability"
        } test started. Check your webcam window.`
      );
    } catch (err) {
      setTestError(err.message);
    } finally {
      setStartingTest("");
    }
  };

  // =========================================================
  // DATA
  // =========================================================

  const eyeRecords = results?.eye_blink?.records ?? [];
  const fingerRecords = results?.finger_stability?.records ?? [];
  const combinedRecords = results?.combined_report?.records ?? [];

  const latestEye = eyeRecords[eyeRecords.length - 1];
  const latestFinger = fingerRecords[fingerRecords.length - 1];

  const anyTestRunning = Object.values(testStatus).includes("running");

  const navItems = [
    { name: "Dashboard", icon: "▦" },
    { name: "Screening Tests", icon: "◎" },
    { name: "Test Results", icon: "▤" },
  ];

  // =========================================================
  // METRIC CARD
  // =========================================================

  const MetricCard = ({ icon, label, value, detail }) => (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <p>{label}</p>
      <h2>{value}</h2>
      <span>{detail}</span>
    </div>
  );

  // =========================================================
  // ML METRICS SECTION
  // =========================================================

  const renderMLMetrics = () => {
    const metrics = mlStatus?.metrics ?? {};

    if (mlLoading) {
      return (
        <div className="status-box">
          Loading machine-learning model information...
        </div>
      );
    }

    if (mlError) {
      return (
        <div className="status-box error-box">
          {mlError}
          <button className="secondary-button" onClick={loadMLStatus}>
            Retry
          </button>
        </div>
      );
    }

    if (!mlStatus?.model_loaded) {
      return (
        <div className="notice-box">
          <strong>ML model unavailable</strong>
          <p>
            The trained model could not be loaded. Check the model file
            and Flask terminal.
          </p>
        </div>
      );
    }

    return (
      <section className="content-panel results-panel">
        <div className="section-title-row">
          <div>
            <h2>Machine Learning Model Evaluation</h2>
            <p className="muted">
              Research dataset evaluation results
            </p>
          </div>

          <span className="mode-badge">● Model Loaded</span>
        </div>

        <div className="metrics-grid">
          <MetricCard
            icon="◎"
            label="F1 Score"
            value={
              metrics.f1_diagnosed != null
                ? `${(metrics.f1_diagnosed * 100).toFixed(2)}%`
                : "N/A"
            }
            detail="Diagnosed class"
          />

          <MetricCard
            icon="✓"
            label="Precision"
            value={
              metrics.precision_diagnosed != null
                ? `${(metrics.precision_diagnosed * 100).toFixed(2)}%`
                : "N/A"
            }
            detail="Diagnosed class"
          />

          <MetricCard
            icon="◷"
            label="Recall"
            value={
              metrics.recall_diagnosed != null
                ? `${(metrics.recall_diagnosed * 100).toFixed(2)}%`
                : "N/A"
            }
            detail="Diagnosed class"
          />

          <MetricCard
            icon="▤"
            label="ROC-AUC"
            value={
              metrics.roc_auc != null
                ? Number(metrics.roc_auc).toFixed(4)
                : "N/A"
            }
            detail="Model evaluation score"
          />
        </div>

        <div className="metrics-grid" style={{ marginTop: "20px" }}>
          <MetricCard
            icon="▦"
            label="Dataset Records"
            value={metrics.dataset_rows ?? "N/A"}
            detail="Total research records"
          />

          <MetricCard
            icon="⚙"
            label="Features"
            value={metrics.feature_count ?? "N/A"}
            detail="Input features used"
          />

          <MetricCard
            icon="▤"
            label="Training Records"
            value={metrics.train_rows ?? "N/A"}
            detail="Training dataset"
          />

          <MetricCard
            icon="◷"
            label="Testing Records"
            value={metrics.test_rows ?? "N/A"}
            detail="Holdout test dataset"
          />
        </div>

        <h3 style={{ marginTop: "28px" }}>
          Confusion Matrix
        </h3>

        {Array.isArray(metrics.confusion_matrix) ? (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Actual / Predicted</th>
                  <th>Control - No</th>
                  <th>Diagnosed - Yes</th>
                </tr>
              </thead>

              <tbody>
                {metrics.confusion_matrix.map((row, index) => (
                  <tr key={index}>
                    <td>
                      {index === 0
                        ? "Actual Control"
                        : "Actual Diagnosed"}
                    </td>

                    {row.map((value, cellIndex) => (
                      <td key={cellIndex}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">Confusion matrix not available.</p>
        )}

        <div className="notice-box" style={{ marginTop: "20px" }}>
          <strong>Important Model Limitations</strong>

          <p>
            {mlStatus.message}
          </p>

          <p>
            This model uses research-dataset features and is not
            directly compatible with the current webcam measurements.
            It has not been clinically validated.
          </p>

          {metrics.warning && (
            <p style={{ marginTop: "10px" }}>
              <strong>Validation warning:</strong> {metrics.warning}
            </p>
          )}

          <p style={{ marginTop: "10px" }}>
            The model output is experimental research information.
            It must not be used to diagnose or rule out Parkinson's disease.
          </p>
        </div>
      </section>
    );
  };

  // =========================================================
  // DASHBOARD
  // =========================================================

  const renderDashboard = () => (
    <>
      <div className="page-heading">
        <div>
          <p className="breadcrumb">Workspace / Dashboard</p>
          <h1>Dashboard</h1>
        </div>

        <span className="mode-badge">● Research Mode</span>
      </div>

      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">COMPUTER VISION PROJECT</span>

          <h2>Parkinson's Screening Research</h2>

          <p>
            A webcam-based research prototype for recording
            experimental eye-blink and finger-movement measurements,
            with a separate research-dataset machine-learning model.
          </p>

          <button
            className="primary-button"
            onClick={() => setActivePage("Screening Tests")}
          >
            Explore Screening Tests <span>→</span>
          </button>
        </div>

        <div className="hero-symbol" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="orbit orbit-three" />
          <div className="symbol-core">N</div>
        </div>
      </section>

      <section className="overview-section">
        <h2>Project Overview</h2>
        <p className="muted">
          Measurements retrieved from your local backend
        </p>

        {loading ? (
          <div className="status-box">
            Loading screening results...
          </div>
        ) : error ? (
          <div className="status-box error-box">
            {error}
            <button className="secondary-button" onClick={loadResults}>
              Retry
            </button>
          </div>
        ) : (
          <div className="metrics-grid">
            <MetricCard
              icon="◎"
              label="Eye Blink Rate"
              value={
                latestEye
                  ? `${latestEye.Blink_Rate_Per_Minute ?? "—"}/min`
                  : "No data"
              }
              detail={
                latestEye
                  ? `${latestEye.Total_Blinks ?? "—"} blinks recorded`
                  : "Run the eye-blink test first"
              }
            />

            <MetricCard
              icon="◷"
              label="Test Duration"
              value={
                latestEye
                  ? `${latestEye.Duration_Seconds ?? "—"} sec`
                  : latestFinger
                    ? `${latestFinger.Actual_Duration_Seconds ?? "—"} sec`
                    : "No data"
              }
              detail="Duration of the latest available test"
            />

            <MetricCard
              icon="▤"
              label="Finger Measurements"
              value={latestFinger?.Total_Measurements ?? "No data"}
              detail={
                latestFinger
                  ? `Average movement: ${
                      latestFinger.Average_Frame_Movement_Pixels ?? "—"
                    } pixels`
                  : "Run the finger-stability test first"
              }
            />

            <MetricCard
              icon="⚙"
              label="ML Research Model"
              value={
                mlLoading
                  ? "Loading..."
                  : mlStatus?.model_loaded
                    ? "Loaded"
                    : "Unavailable"
              }
              detail={
                mlStatus?.model_loaded
                  ? `${mlStatus.model_type} · ${mlStatus.features_required} features`
                  : "Check ML model status"
              }
            />
          </div>
        )}
      </section>

      <section className="recent-section">
        <div className="section-title-row">
          <div>
            <h2>Latest Measurements</h2>
            <p className="muted">
              Results from your saved CSV files
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={loadResults}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "↻ Refresh"}
          </button>
        </div>

        {error ? null : (
          <ResultsTable records={combinedRecords.slice(-4).reverse()} />
        )}
      </section>

      {renderMLMetrics()}
    </>
  );

  // =========================================================
  // SCREENING TESTS
  // =========================================================

  const renderScreeningTests = () => (
    <>
      <div className="page-heading">
        <div>
          <p className="breadcrumb">Workspace / Screening Tests</p>
          <h1>Screening Tests</h1>
        </div>

        <span className="mode-badge">● Local Prototype</span>
      </div>

      <section className="content-panel">
        <h2>Available Computer Vision Tests</h2>

        <p className="muted">
          Start a webcam test directly from this dashboard.
        </p>

        {testMessage && (
          <div className="status-box">{testMessage}</div>
        )}

        {testError && (
          <div className="status-box error-box">{testError}</div>
        )}

        <div className="test-card">
          <div className="test-icon">◎</div>

          <div className="test-card-content">
            <h3>Eye Blink Test</h3>

            <p>
              Records eye-blink count and estimated blink rate
              during a timed webcam session.
            </p>

            <span className="tag">
              Python · MediaPipe · OpenCV
            </span>

            <div style={{ marginTop: "18px" }}>
              <button
                className="primary-button"
                onClick={() => startTest("eye-blink")}
                disabled={anyTestRunning || startingTest !== ""}
              >
                {testStatus["eye-blink"] === "running"
                  ? "Test Running..."
                  : startingTest === "eye-blink"
                    ? "Starting..."
                    : "Start Eye Blink Test"}
              </button>

              {testStatus["eye-blink"] === "running" && (
                <p style={{ marginTop: "10px" }}>
                  ● Webcam test is currently running.
                </p>
              )}
            </div>
          </div>

          <div className="test-card-result">
            <strong>
              {latestEye?.Blink_Rate_Per_Minute ?? "—"}
            </strong>
            <span>blinks/min</span>
          </div>
        </div>

        <div className="test-card">
          <div className="test-icon blue-icon">⌁</div>

          <div className="test-card-content">
            <h3>Finger Stability Test</h3>

            <p>
              Measures frame-to-frame index-finger movement
              during a timed webcam session.
            </p>

            <span className="tag">
              Python · MediaPipe · OpenCV
            </span>

            <div style={{ marginTop: "18px" }}>
              <button
                className="primary-button"
                onClick={() => startTest("finger-stability")}
                disabled={anyTestRunning || startingTest !== ""}
              >
                {testStatus["finger-stability"] === "running"
                  ? "Test Running..."
                  : startingTest === "finger-stability"
                    ? "Starting..."
                    : "Start Finger Stability Test"}
              </button>

              {testStatus["finger-stability"] === "running" && (
                <p style={{ marginTop: "10px" }}>
                  ● Webcam test is currently running.
                </p>
              )}
            </div>
          </div>

          <div className="test-card-result">
            <strong>
              {latestFinger?.Average_Frame_Movement_Pixels ?? "—"}
            </strong>
            <span>average pixels</span>
          </div>
        </div>

        <div className="notice-box">
          <strong>How it works</strong>

          <p>
            Click Start Test to launch the Python webcam script
            through the Flask backend. Allow webcam access if prompted.
            Once the test finishes, saved results refresh automatically.
          </p>
        </div>
      </section>
    </>
  );

  // =========================================================
  // TEST RESULTS PAGE
  // =========================================================

  const renderTestResults = () => (
    <>
      <div className="page-heading">
        <div>
          <p className="breadcrumb">Workspace / Test Results</p>
          <h1>Test Results</h1>
        </div>

        <button
          className="secondary-button"
          onClick={loadResults}
          disabled={loading}
        >
          ↻ Refresh Results
        </button>
      </div>

      {loading ? (
        <div className="status-box">
          Loading saved results...
        </div>
      ) : error ? (
        <div className="status-box error-box">{error}</div>
      ) : (
        <>
          <section className="content-panel results-panel">
            <h2>Eye Blink Test</h2>
            <p className="muted">
              Saved eye-blink measurements
            </p>

            <ResultsTable
              records={eyeRecords}
              columns={[
                ["Timestamp", "Timestamp"],
                ["Duration (sec)", "Duration_Seconds"],
                ["Total blinks", "Total_Blinks"],
                ["Blink rate/min", "Blink_Rate_Per_Minute"],
              ]}
            />
          </section>

          <section className="content-panel results-panel">
            <h2>Finger Stability Test</h2>
            <p className="muted">
              Saved finger movement measurements
            </p>

            <ResultsTable
              records={fingerRecords}
              columns={[
                ["Timestamp", "Timestamp"],
                ["Duration (sec)", "Actual_Duration_Seconds"],
                ["Measurements", "Total_Measurements"],
                [
                  "Average movement (pixels)",
                  "Average_Frame_Movement_Pixels",
                ],
              ]}
            />
          </section>

          <section className="content-panel results-panel">
            <h2>Combined Report</h2>
            <p className="muted">
              Combined measurements from both tests
            </p>

            <ResultsTable
              records={[...combinedRecords].reverse()}
            />
          </section>

          {renderMLMetrics()}
        </>
      )}

      <div className="notice-box">
        These measurements are experimental and are not a diagnosis
        of Parkinson's disease. The prototype has not been clinically validated.
      </div>
    </>
  );

  // =========================================================
  // MAIN APP
  // =========================================================

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">N</div>

          <div>
            <h2>NeuroVision</h2>
            <p>Research Prototype</p>
          </div>
        </div>

        <p className="sidebar-label">WORKSPACE</p>

        <nav>
          {navItems.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${
                activePage === item.name ? "active" : ""
              }`}
              onClick={() => setActivePage(item.name)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className="status-dot" />

          <div>
            <strong>Local Prototype</strong>
            <p>Development environment</p>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {activePage === "Dashboard" && renderDashboard()}

        {activePage === "Screening Tests" && renderScreeningTests()}

        {activePage === "Test Results" && renderTestResults()}
      </main>

      <footer className="disclaimer">
        Research prototype only. The ML model is trained on a separate
        research dataset and is not connected to the webcam measurements.
        This application has not been clinically validated to diagnose
        or rule out Parkinson's disease.
      </footer>
    </div>
  );
}

// =============================================================
// REUSABLE RESULTS TABLE
// =============================================================

function ResultsTable({ records = [], columns }) {
  const tableColumns =
    columns ??
    [
      ["Test", "Test"],
      ["Metric", "Metric"],
      ["Value", "Value"],
      ["Timestamp", "Timestamp"],
    ];

  if (!records.length) {
    return (
      <div className="empty-state">
        No saved measurements available.
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            {tableColumns.map(([label]) => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {records.map((record, index) => (
            <tr key={`${record.Timestamp ?? "record"}-${index}`}>
              {tableColumns.map(([label, key]) => (
                <td key={key}>
                  {record[key] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
