import { useEffect, useMemo, useRef, useState } from "react";
import styled, { createGlobalStyle } from "styled-components";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const presets = {
  public: {
    threshold: 0.25,
    close_k: 7,
    min_area: 0,
    label: "Rover/Public preset",
  },
  private: {
    threshold: 0.9,
    close_k: 7,
    min_area: 0,
    label: "Lunar/Private preset",
  },
};

const resultRows = [
  {
    model: "Model #1 FCN baseline",
    split: "Public val @ iter_5000",
    iou: "69.33%",
    miou: "80.84%",
    aacc: "93.47%",
  },
  {
    model: "Model #1 FCN baseline",
    split: "Private (10)",
    iou: "21.74%",
    miou: "34.97%",
    aacc: "54.72%",
  },
  {
    model: "Model #2 DeepLabV3+ R50 locked final (report)",
    split: "Private report",
    iou: "23.92%",
    miou: "42.24%",
    aacc: "64.90%",
  },
  {
    model:
      "Model #3 DeepLab calibration (same checkpoint, tuned post-processing)",
    split: "Public (1085 val)",
    iou: "63.16%",
    miou: "Not reported",
    aacc: "Not reported",
  },
  {
    model:
      "Model #3 DeepLab calibration (same checkpoint, tuned post-processing)",
    split: "Private (10)",
    iou: "32.18%",
    miou: "Not reported",
    aacc: "Not reported",
  },
];

const chartData = [
  { name: "FCN", public: 69.33, private: 21.74 },
  { name: "DeepLabV3+", public: 63.16, private: 23.92 },
  { name: "DeepLabV3+ Calibrated", public: 63.16, private: 32.18 },
];

function cls(...items) {
  return items.filter(Boolean).join(" ");
}

const Global = createGlobalStyle`
  * { box-sizing: border-box; }
  html, body, #root { margin: 0; min-height: 100%; }
  body {
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    color: #edf2ff;
    background:
      radial-gradient(900px 480px at -5% -10%, rgba(89,128,235,0.2), transparent),
      radial-gradient(900px 520px at 105% -8%, rgba(55,140,226,0.18), transparent),
      linear-gradient(140deg, #08132d 0%, #091a3c 55%, #0a2750 100%);
  }
`;

const AppShell = styled.div`
  min-height: 100vh;
  padding: 18px;
`;

const Wrap = styled.div`
  max-width: 1780px;
  margin: 0 auto;
`;

const Glass = styled.div`
  border: 1px solid rgba(88, 127, 218, 0.52);
  background: rgba(11, 26, 68, 0.58);
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.3);
`;

const Topbar = styled(Glass)`
  height: 76px;
  border-radius: 22px;
  padding: 0 22px;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const Logo = styled.div`
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(140deg, #72baff, #5a69ff);
  display: grid;
  place-items: center;
  box-shadow: 0 8px 22px rgba(94, 103, 252, 0.45);
  span {
    width: 11px;
    height: 11px;
    border-radius: 999px;
    background: #10172b;
  }
`;

const BrandTitle = styled.h1`
  margin: 0;
  font-size: 48px;
  font-weight: 800;
  letter-spacing: -0.02em;
`;

const Tabs = styled.div`
  display: flex;
  gap: 8px;
  padding: 5px;
  border-radius: 14px;
  border: 1px solid rgba(96, 131, 213, 0.34);
  background: rgba(8, 20, 52, 0.7);
`;

const Tab = styled.button`
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: #a7b8d7;
  font-size: 30px;
  font-weight: 620;
  padding: 8px 18px;
  cursor: pointer;
  transition: all 0.2s ease;
  &.active {
    color: #bb99ff;
    background: rgba(86, 72, 170, 0.33);
    box-shadow: 0 0 24px rgba(88, 90, 255, 0.38);
  }
`;

const StudioLayout = styled.div`
  margin-top: 16px;
  display: grid;
  grid-template-columns: 390px 1fr;
  gap: 14px;
`;

const Control = styled(Glass)`
  border-radius: 24px;
  padding: 16px;
`;

const ControlHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  h2 {
    margin: 0;
    font-size: 50px;
    font-weight: 800;
  }
  button {
    border: 1px solid rgba(103, 145, 228, 0.62);
    border-radius: 9px;
    background: rgba(24, 46, 96, 0.72);
    color: #d9e8ff;
    font-size: 13px;
    padding: 5px 8px;
    cursor: pointer;
  }
`;

const UploadBox = styled.label`
  margin-top: 12px;
  min-height: 124px;
  border-radius: 18px;
  border: 1px dashed rgba(102, 147, 234, 0.72);
  background: rgba(20, 40, 88, 0.62);
  display: grid;
  place-items: center;
  text-align: center;
  cursor: pointer;
  input {
    display: none;
  }
  div {
    font-size: 20px;
    color: #e9f2ff;
    padding: 12px;
    word-break: break-word;
  }
`;

const Presets = styled.div`
  margin-top: 10px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  button {
    height: 46px;
    border-radius: 14px;
    border: 1px solid rgba(97, 141, 234, 0.72);
    font-size: 14px;
    font-weight: 700;
    color: #e5efff;
    background: rgba(21, 42, 91, 0.72);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  button.active {
    background: linear-gradient(135deg, #4e8aff, #5e5fff);
  }
`;

const Field = styled.div`
  margin-top: 10px;
  .label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 17px;
    font-weight: 650;
  }
  .value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 16px;
  }
  input[type="range"] {
    margin-top: 8px;
    width: 100%;
    accent-color: #d8e6ff;
  }
`;

const Inputs = styled.div`
  margin-top: 10px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  label {
    display: block;
    color: #abc1e8;
    font-size: 14px;
  }
  input {
    margin-top: 5px;
    width: 100%;
    height: 42px;
    border-radius: 10px;
    border: 1px solid rgba(89, 132, 230, 0.76);
    background: rgba(14, 28, 67, 0.93);
    color: #f3f8ff;
    padding: 0 10px;
    font-size: 17px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
`;

const RunBtn = styled.button`
  margin-top: 12px;
  width: 100%;
  height: 56px;
  border: 0;
  border-radius: 14px;
  background: linear-gradient(135deg, #3f90ff, #6157ff);
  color: #f3f9ff;
  font-size: 15px;
  line-height: 1;
  font-weight: 760;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: pointer;
  transition: all 0.2s ease;
  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(58, 115, 234, 0.35);
  }
  &:disabled {
    opacity: 0.74;
  }
`;

const Health = styled.div`
  margin-top: 10px;
  font-size: 16px;
  color: #b4c8ee;
  .ok {
    color: #50d381;
    font-weight: 700;
  }
  .bad {
    color: #ff8293;
    font-weight: 700;
  }
`;

const Error = styled.div`
  margin-top: 8px;
  border-radius: 10px;
  border: 1px solid rgba(255, 125, 146, 0.55);
  padding: 8px;
  font-size: 13px;
  color: #ffd8de;
`;

const Panels = styled.div`
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
`;

const Panel = styled(Glass)`
  border-radius: 24px;
  min-height: 460px;
  overflow: hidden;
`;

const PanelTop = styled.div`
  height: 52px;
  padding: 0 12px;
  border-bottom: 1px solid rgba(87, 126, 216, 0.52);
  background: rgba(32, 53, 113, 0.5);
  display: flex;
  align-items: center;
  justify-content: space-between;
  h3 {
    margin: 0;
    font-size: 19px;
    font-weight: 760;
  }
  a {
    border-radius: 9px;
    border: 1px solid rgba(123, 167, 252, 0.72);
    padding: 4px 9px;
    font-size: 13px;
    color: #e8f1ff;
    text-decoration: none;
    background: rgba(17, 37, 82, 0.78);
    transition: all 0.2s ease;
  }
  a:hover {
    box-shadow: 0 0 14px rgba(88, 142, 255, 0.3);
  }
  a.disabled {
    opacity: 0.45;
    pointer-events: none;
  }
`;

const Placeholder = styled.div`
  height: calc(100% - 52px);
  display: grid;
  place-items: center;
  color: #88a5d8;
  font-size: 19px;
  text-align: center;
  padding: 16px;
`;

const Img = styled.img`
  width: 100%;
  height: calc(100% - 52px);
  object-fit: contain;
  background: #0f182a;
`;

const Section = styled(Glass)`
  margin-top: 16px;
  border-radius: 24px;
  padding: 20px;
`;

const Headline = styled.h2`
  margin: 0;
  font-size: 62px;
  font-weight: 800;
  letter-spacing: -0.02em;
`;

const Subline = styled.p`
  margin: 8px 0 0;
  color: #9db4dd;
  font-size: 22px;
  line-height: 1.35;
  max-width: 940px;
`;

const MetricCards = styled.div`
  margin-top: 16px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
`;

const MetricCard = styled(Glass)`
  border-radius: 16px;
  padding: 14px;
  h3 {
    margin: 0;
    font-size: 18px;
  }
  .value {
    margin-top: 6px;
    font-size: 42px;
    font-weight: 800;
  }
  .line {
    margin-top: 3px;
    font-size: 13px;
    color: #9bb3dd;
  }
`;

const TableCard = styled(Glass)`
  margin-top: 14px;
  border-radius: 16px;
  padding: 14px;
  h3 {
    margin: 0;
    font-size: 24px;
  }
  .wrap {
    margin-top: 8px;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    min-width: 840px;
  }
  th,
  td {
    border-bottom: 1px solid rgba(95, 136, 224, 0.3);
    text-align: left;
    padding: 8px 6px;
    font-size: 13px;
  }
  th {
    color: #d4e4ff;
  }
  td {
    color: #bbd0f1;
  }
`;

const ChartCard = styled(Glass)`
  margin-top: 14px;
  border-radius: 16px;
  padding: 14px;
  h3 {
    margin: 0;
    font-size: 24px;
  }
  .chart {
    margin-top: 8px;
    height: 260px;
  }
`;

const MethodGrid = styled.div`
  margin-top: 16px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
`;

const MethodCard = styled(Glass)`
  border-radius: 16px;
  padding: 14px;
  h3 {
    margin: 0;
    font-size: 18px;
  }
  p {
    margin: 8px 0 0;
    color: #abc0e3;
    font-size: 14px;
    line-height: 1.4;
  }
`;

const HealthLine = styled.div`
  margin-top: 6px;
`;

function StudioPanel({ title, src, downloadName }) {
  return (
    <Panel>
      <PanelTop>
        <h3>{title}</h3>
        <a
          className={cls(!src && "disabled")}
          href={src || "#"}
          download={downloadName}
        >
          Download
        </a>
      </PanelTop>
      {src ? (
        <Img src={src} alt={title} />
      ) : (
        <Placeholder>Run segmentation to generate this panel.</Placeholder>
      )}
    </Panel>
  );
}

export default function App() {
  const [tab, setTab] = useState("studio");
  const [preset, setPreset] = useState("public");
  const [threshold, setThreshold] = useState(0.25);
  const [closeK, setCloseK] = useState(7);
  const [minArea, setMinArea] = useState(0);
  const [health, setHealth] = useState({ status: "ok", env: "shadowseg" });
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [file, setFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState("");
  const [maskUrl, setMaskUrl] = useState("");
  const [overlayUrl, setOverlayUrl] = useState("");
  const previewRef = useRef("");

  useEffect(() => {
    fetch("/health")
      .then(async (r) => ({ ok: r.ok, data: await r.json() }))
      .then(({ ok, data }) => {
        if (!ok) {
          setHealth({ status: "error", env: data?.env || "shadowseg" });
          return;
        }
        setHealth({
          status: data.status || "ok",
          env: data.env || "shadowseg",
        });
      })
      .catch(() => setHealth({ status: "error", env: "shadowseg" }));
  }, []);

  const thresholdText = useMemo(() => threshold.toFixed(2), [threshold]);

  const applyPreset = (key) => {
    const p = presets[key];
    setPreset(key);
    setThreshold(p.threshold);
    setCloseK(p.close_k);
    setMinArea(p.min_area);
  };

  const selectFile = (img) => {
    if (!img) return;
    if (!img.type.startsWith("image/")) {
      setError("Please upload JPG or PNG image.");
      return;
    }
    setError("");
    setFile(img);
    setMaskUrl("");
    setOverlayUrl("");
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = URL.createObjectURL(img);
    previewRef.current = url;
    setOriginalUrl(url);
  };

  const run = async () => {
    if (!file) {
      setError("Upload image first.");
      return;
    }
    setError("");
    setRunning(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("threshold", threshold.toFixed(2));
      form.append("close_k", String(closeK));
      form.append("min_area", String(minArea));
      const response = await fetch("/api/run", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok || data.status !== "ok") {
        throw new Error(data.detail || data.message || "Segmentation failed.");
      }
      setMaskUrl(data.mask_url || "");
      setOverlayUrl(data.overlay_url || "");
      if (data.original_url) setOriginalUrl(data.original_url);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  const resetStudio = () => {
    setFile(null);
    setOriginalUrl("");
    setMaskUrl("");
    setOverlayUrl("");
    setError("");
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    }
  };

  return (
    <>
      <Global />
      <AppShell>
        <Wrap>
          <Topbar>
            <Brand>
              <Logo>
                <span />
              </Logo>
              <BrandTitle>ShadowSeg</BrandTitle>
            </Brand>
            <Tabs>
              <Tab
                className={cls(tab === "studio" && "active")}
                onClick={() => setTab("studio")}
              >
                Studio
              </Tab>
              <Tab
                className={cls(tab === "results" && "active")}
                onClick={() => setTab("results")}
              >
                Results
              </Tab>
              <Tab
                className={cls(tab === "methodology" && "active")}
                onClick={() => setTab("methodology")}
              >
                Methodology
              </Tab>
            </Tabs>
          </Topbar>

          {tab === "studio" && (
            <StudioLayout>
              <Control>
                <ControlHeader>
                  <h2>Studio</h2>
                  <button onClick={resetStudio}>Reset</button>
                </ControlHeader>
                <UploadBox>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={(e) => selectFile(e.target.files?.[0])}
                  />
                  <div>{file ? file.name : "Upload JPG or PNG image"}</div>
                </UploadBox>
                <Presets>
                  <button
                    className={cls(preset === "public" && "active")}
                    onClick={() => applyPreset("public")}
                  >
                    {presets.public.label}
                  </button>
                  <button
                    className={cls(preset === "private" && "active")}
                    onClick={() => applyPreset("private")}
                  >
                    {presets.private.label}
                  </button>
                </Presets>
                <Field>
                  <div className="label">
                    <span>Threshold</span>
                    <span className="value">{thresholdText}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => {
                      setPreset("");
                      setThreshold(Number(e.target.value));
                    }}
                  />
                </Field>
                <Inputs>
                  <label>
                    close_k
                    <input
                      type="number"
                      min="0"
                      step="1"
                      value={closeK}
                      onChange={(e) => {
                        setPreset("");
                        setCloseK(Math.max(0, Number(e.target.value || 0)));
                      }}
                    />
                  </label>
                  <label>
                    min_area
                    <input
                      type="number"
                      min="0"
                      step="1"
                      value={minArea}
                      onChange={(e) => {
                        setPreset("");
                        setMinArea(Math.max(0, Number(e.target.value || 0)));
                      }}
                    />
                  </label>
                </Inputs>
                <RunBtn onClick={run} disabled={running}>
                  {running ? "Running segmentation..." : "Run segmentation"}
                </RunBtn>
                <Health>
                  <HealthLine>
                    Health:{" "}
                    <span className={health.status === "ok" ? "ok" : "bad"}>
                      {health.status}
                    </span>{" "}
                    env:{health.env}
                  </HealthLine>
                </Health>
                {error ? <Error>{error}</Error> : null}
              </Control>
              <Panels>
                <StudioPanel
                  title="Original"
                  src={originalUrl}
                  downloadName="original.png"
                />
                <StudioPanel
                  title="Predicted Mask"
                  src={maskUrl}
                  downloadName="mask.png"
                />
                <StudioPanel
                  title="Overlay"
                  src={overlayUrl}
                  downloadName="overlay.png"
                />
              </Panels>
            </StudioLayout>
          )}

          {tab === "results" && (
            <Section>
              <Headline>Performance Metrics</Headline>
              <Subline>
                Benchmarking our shadow segmentation model against standard
                datasets and real-world rover telemetry.
              </Subline>
              <MetricCards>
                <MetricCard>
                  <h3>Public (1085 val)</h3>
                  <div className="value">63.16%</div>
                  <div className="line">thr=0.25, close_k=7, min_area=0</div>
                </MetricCard>
                <MetricCard>
                  <h3>Private (10)</h3>
                  <div className="value">32.18%</div>
                  <div className="line">thr=0.90, close_k=7, min_area=0</div>
                </MetricCard>
              </MetricCards>
              <TableCard>
                <h3>Baseline Comparison</h3>
                <div className="wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Split</th>
                        <th>IoU_shadow</th>
                        <th>mIoU</th>
                        <th>aAcc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {resultRows.map((row) => (
                        <tr key={`${row.model}-${row.split}`}>
                          <td>{row.model}</td>
                          <td>{row.split}</td>
                          <td>{row.iou}</td>
                          <td>{row.miou}</td>
                          <td>{row.aacc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </TableCard>
              <ChartCard>
                <h3>Model Comparison</h3>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      margin={{ top: 6, right: 6, left: 0, bottom: 4 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(131,161,218,0.24)"
                      />
                      <XAxis
                        dataKey="name"
                        stroke="#a8bfdf"
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis
                        stroke="#a8bfdf"
                        domain={[0, 100]}
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#132341",
                          border: "1px solid rgba(121,161,232,0.44)",
                          borderRadius: 10,
                        }}
                      />
                      <Bar
                        dataKey="public"
                        fill="#8d89ff"
                        radius={[5, 5, 0, 0]}
                      />
                      <Bar
                        dataKey="private"
                        fill="#10b678"
                        radius={[5, 5, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>
            </Section>
          )}

          {tab === "methodology" && (
            <Section>
              <Headline>Methodology</Headline>
              <Subline>
                Practical calibration from public SBU-like shadows to private
                rover-like imagery.
              </Subline>
              <MethodGrid>
                <MethodCard>
                  <h3>Domain shift: SBU to Rover-like imagery</h3>
                  <p>
                    The public training data captures everyday scenes, while the
                    private 10-image rover-like set has distinct lighting,
                    texture, and contrast patterns.
                  </p>
                </MethodCard>
                <MethodCard>
                  <h3>Calibration: threshold + morphological closing</h3>
                  <p>
                    The model checkpoint remains fixed. Calibration changes only
                    threshold and post-processing to improve mask reliability
                    across domains.
                  </p>
                </MethodCard>
                <MethodCard>
                  <h3>close_k: fills small holes, connects fragments</h3>
                  <p>
                    Morphological closing links broken shadow components and
                    smooths local gaps created by uneven terrain and imaging
                    noise.
                  </p>
                </MethodCard>
                <MethodCard>
                  <h3>min_area: removes tiny regions (0 keeps all)</h3>
                  <p>
                    min_area controls connected-component filtering. With
                    min_area set to 0, all detected shadow components are
                    retained.
                  </p>
                </MethodCard>
              </MethodGrid>
            </Section>
          )}
        </Wrap>
      </AppShell>
    </>
  );
}
