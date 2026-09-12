import React from 'react'

// Renders the REAL segmentation preview PNG that federated/server.py produces
// every round (base64-encoded, riding along on the same WebSocket metrics
// message as loss/dice — see render_segmentation_png() in server.py). Falls
// back to a schematic placeholder only until the first round arrives, so the
// panel never renders blank before training starts.
export default function SegmentationPreview({ imageBase64, round }) {
  return (
    <div className="panel">
      <h2>Segmentation Preview</h2>
      <p className="sub">
        {imageBase64 ? `Middle axial slice · global model, round ${round}` : 'Waiting for first round…'}
      </p>
      {imageBase64 ? (
        <img
          src={`data:image/png;base64,${imageBase64}`}
          alt={`Predicted tumor segmentation, round ${round}`}
          style={{ width: '100%', maxWidth: 220, display: 'block', margin: '0 auto', borderRadius: 8, border: '1px solid var(--panel-border)' }}
        />
      ) : (
        <svg viewBox="0 0 200 200" style={{ width: '100%', maxWidth: 220, display: 'block', margin: '0 auto' }}>
          <rect width="200" height="200" rx="8" fill="#0a0f1a" stroke="#1f2c46" />
          <circle cx="100" cy="100" r="55" fill="#3ddbd0" opacity="0.12" />
          <circle cx="100" cy="100" r="34" fill="#3ddbd0" opacity="0.28" />
          <circle cx="100" cy="100" r="16" fill="#f0576b" opacity="0.55" />
        </svg>
      )}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginTop: 8, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--muted)' }}>
        <span><span style={{ color: '#6fdc7d' }}>■</span> Whole Tumor</span>
        <span><span style={{ color: '#5b9bf5' }}>■</span> Tumor Core</span>
        <span><span style={{ color: '#f0576b' }}>■</span> Enhancing</span>
      </div>
    </div>
  )
}
