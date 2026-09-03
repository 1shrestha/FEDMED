import React, { useEffect, useRef, useState } from 'react'
import NodeStatusPanel from './components/NodeStatusPanel.jsx'
import TrainingChart from './components/TrainingChart.jsx'
import SegmentationPreview from './components/SegmentationPreview.jsx'

const WS_URL = 'ws://localhost:8765'
const BASELINE_DICE = 0.81 // replace with the real number printed by central_baseline/train_baseline.py

export default function App() {
  const [history, setHistory] = useState([])
  const [connected, setConnected] = useState(false)
  const [nodes, setNodes] = useState([
    { id: 0, name: 'Hospital Node A', samples: 20, online: true, lastRound: null },
    { id: 1, name: 'Hospital Node B', samples: 20, online: true, lastRound: null },
    { id: 2, name: 'Hospital Node C', samples: 20, online: true, lastRound: null },
  ])
  const wsRef = useRef(null)

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        setTimeout(connect, 2000) // auto-reconnect — training server / ws server may start after the dashboard
      }
      ws.onmessage = (event) => {
        const entry = JSON.parse(event.data)
        setHistory((prev) => [...prev, entry])
        setNodes((prev) => prev.map((n) => ({ ...n, lastRound: entry.round })))
      }
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  return (
    <div className="app">
      <div className="app-header">
        <div>
          <p className="eyebrow">FedMed / Cross-Silo Federated Learning</p>
          <h1>Training Dashboard</h1>
        </div>
        <span className="status-pill">
          <span className="dot" />
          {connected ? 'live · ws://localhost:8765' : 'waiting for metrics stream…'}
        </span>
      </div>

      <div className="grid">
        <TrainingChart history={history} baselineDice={BASELINE_DICE} />
        <NodeStatusPanel nodes={nodes} />
      </div>

      <div className="grid" style={{ marginTop: 20 }}>
        <SegmentationPreview />
        <div className="panel">
          <h2>Privacy Layer</h2>
          <p className="sub">What never leaves each hospital, and what does.</p>
          <div className="node-row">
            <span className="node-name">Raw MRI volumes</span>
            <span className="badge offline">NEVER TRANSMITTED</span>
          </div>
          <div className="node-row">
            <span className="node-name">Weight deltas</span>
            <span className="badge online">HE-ENCRYPTED (CKKS)</span>
          </div>
          <div className="node-row">
            <span className="node-name">Aggregation</span>
            <span className="badge online">DP-NOISED FEDAVG</span>
          </div>
        </div>
      </div>

      <p className="footer-note">
        history has {history.length} logged round(s) · start federated/server.py + federated/ws_metrics_server.py, then `npm run dev`
      </p>
    </div>
  )
}
