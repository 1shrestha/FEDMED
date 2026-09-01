import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function TrainingChart({ history, baselineDice }) {
  const latest = history[history.length - 1]

  return (
    <div className="panel">
      <h2>Global Model Convergence</h2>
      <p className="sub">Aggregated across nodes each FedAvg round — server never sees raw MRI data.</p>

      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <LineChart data={history} margin={{ top: 4, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#1f2c46" strokeDasharray="3 3" />
            <XAxis dataKey="round" stroke="#7a8aa8" fontSize={12} tickLine={false} />
            <YAxis stroke="#7a8aa8" fontSize={12} tickLine={false} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ background: '#111a2c', border: '1px solid #1f2c46', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#7a8aa8' }}
            />
            <Line type="monotone" dataKey="dice" stroke="#3ddbd0" strokeWidth={2} dot={false} name="Global Dice" />
            {baselineDice != null && (
              <Line
                type="monotone"
                dataKey={() => baselineDice}
                stroke="#f0b155"
                strokeDasharray="5 4"
                strokeWidth={1.5}
                dot={false}
                name="Centralized Baseline"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="metric-row">
        <div className="metric">
          <div className="value">{latest ? latest.dice.toFixed(3) : '—'}</div>
          <div className="label">Global Dice (round {latest?.round ?? 0})</div>
        </div>
        <div className="metric">
          <div className="value">{baselineDice != null ? baselineDice.toFixed(3) : '—'}</div>
          <div className="label">Centralized Baseline</div>
        </div>
        <div className="metric">
          <div className="value">{history.length}</div>
          <div className="label">Rounds Completed</div>
        </div>
      </div>
    </div>
  )
}
