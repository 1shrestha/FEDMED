import React from 'react'

// A "sealed vault" cipher-strip animation stands in for "this hospital's raw
// data never leaves the building — only an encrypted delta does."
function CipherStrip() {
  const glyphs = '0123456789abcdef'
  const str = Array.from({ length: 60 }, () => glyphs[Math.floor(Math.random() * glyphs.length)]).join('')
  return <div className="cipher-strip">{str}</div>
}

export default function NodeStatusPanel({ nodes }) {
  return (
    <div className="panel">
      <h2>Hospital Nodes</h2>
      <p className="sub">Raw patient data stays local — only encrypted weight deltas leave each site.</p>
      {nodes.map((node) => (
        <div className="node-row" key={node.id}>
          <div>
            <div className="node-name">{node.name}</div>
            <div className="node-detail">{node.samples} local samples · round {node.lastRound ?? '—'}</div>
            {node.online && <CipherStrip />}
          </div>
          <span className={`badge ${node.online ? 'online' : 'offline'}`}>
            {node.online ? 'TRANSMITTING' : 'OFFLINE'}
          </span>
        </div>
      ))}
    </div>
  )
}
