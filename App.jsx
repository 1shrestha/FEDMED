import React, { useEffect, useRef, useState } from "react";
import ConvergenceChart from "./components/ConvergenceChart.jsx";
import NodeStatusPanel from "./components/NodeStatusPanel.jsx";

const METRICS_WS_URL = import.meta.env.VITE_METRICS_WS_URL || "ws://localhost:8090/ws/metrics";

export default function App() {
  const [rounds, setRounds] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(METRICS_WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000); // auto-reconnect
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        const entry = JSON.parse(event.data);
        setRounds((prev) => {
          const next = [...prev];
          const idx = next.findIndex((r) => r.round === entry.round);
          if (idx >= 0) next[idx] = { ...next[idx], ...entry };
          else next.push(entry);
          return next.sort((a, b) => a.round - b.round);
        });
      };
    }
    connect();
    return () => wsRef.current && wsRef.current.close();
  }, []);

  const latest = rounds[rounds.length - 1];

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>FedMed Training Dashboard</h1>
          <p style={styles.subtitle}>
            Cross-silo federated learning · 3D U-Net · brain tumor segmentation
          </p>
        </div>
        <span style={{ ...styles.badge, background: connected ? "#16a34a" : "#dc2626" }}>
          {connected ? "live" : "reconnecting…"}
        </span>
      </header>

      <div style={styles.statsRow}>
        <StatCard label="Round" value={latest?.round ?? "—"} />
        <StatCard label="Global Val Dice" value={fmt(latest?.val_dice)} />
        <StatCard label="Global Val Loss" value={fmt(latest?.val_loss)} />
        <StatCard label="Nodes Reporting" value={latest?.n_clients ?? "—"} />
      </div>

      <div style={styles.grid}>
        <ConvergenceChart rounds={rounds} />
        <NodeStatusPanel latest={latest} />
      </div>
    </div>
  );
}

function fmt(v) {
  return typeof v === "number" ? v.toFixed(4) : "—";
}

function StatCard({ label, value }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={styles.cardValue}>{value}</div>
    </div>
  );
}

const styles = {
  page: { fontFamily: "Inter, system-ui, sans-serif", color: "#e5e7eb", padding: "24px 32px" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 },
  title: { margin: 0, fontSize: 24 },
  subtitle: { margin: "4px 0 0", color: "#9ca3af", fontSize: 14 },
  badge: { padding: "4px 10px", borderRadius: 999, fontSize: 12, fontWeight: 600, color: "white", height: "fit-content" },
  statsRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 },
  card: { background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: "16px 20px" },
  cardLabel: { fontSize: 12, color: "#9ca3af", marginBottom: 6 },
  cardValue: { fontSize: 28, fontWeight: 700 },
  grid: { display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 },
};
