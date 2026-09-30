import React from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";

export default function ConvergenceChart({ rounds }) {
  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>Global Model Convergence</h3>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={rounds} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="round" stroke="#9ca3af" label={{ value: "Round", position: "insideBottom", offset: -2, fill: "#9ca3af" }} />
          <YAxis stroke="#9ca3af" />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
          <Legend />
          <Line type="monotone" dataKey="val_dice" name="Val Dice" stroke="#22c55e" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#f97316" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles = {
  panel: { background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 20 },
  heading: { margin: "0 0 12px", fontSize: 15, fontWeight: 600 },
};
