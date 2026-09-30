import React from 'react'

// Lightweight stand-in for a real slice viewer. Swap the <circle> layers for
// an actual canvas render of a predicted mask slice once you're pulling real
// inference output (e.g. export a middle axial slice as PNG from
// central_baseline / federated eval and fetch it here).
export default function SegmentationPreview() {
  return (
    <div className="panel">
      <h2>Segmentation Preview</h2>
      <p className="sub">Middle axial slice · global model, latest round</p>
      <svg viewBox="0 0 200 200" style={{ width: '100%', maxWidth: 220, display: 'block', margin: '0 auto' }}>
        <rect width="200" height="200" rx="8" fill="#0a0f1a" stroke="#1f2c46" />
        <circle cx="100" cy="100" r="55" fill="#3ddbd0" opacity="0.12" />
        <circle cx="100" cy="100" r="34" fill="#3ddbd0" opacity="0.28" />
        <circle cx="100" cy="100" r="16" fill="#f0576b" opacity="0.55" />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginTop: 8, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--muted)' }}>
        <span><span style={{ color: '#3ddbd0' }}>■</span> Whole Tumor</span>
        <span><span style={{ color: '#3ddbd0', opacity: 0.7 }}>■</span> Tumor Core</span>
        <span><span style={{ color: '#f0576b' }}>■</span> Enhancing</span>
      </div>
    </div>
  )
}
