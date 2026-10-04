// A ruler with ticks every 5% (major at 0/50/100). Used for confidence (needle + abstain mark) and for progress (fill only).
export function Ruler({ value, needle, threshold, labels, compact, ariaLabel }: {
  value: number; needle?: boolean; threshold?: number; labels?: boolean; compact?: boolean; ariaLabel: string;
}) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={`ruler${compact ? ' compact' : ''}`} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} aria-label={ariaLabel}>
      <div className="base" />
      <div className="fill" style={{ width: `${pct}%` }} />
      {Array.from({ length: 21 }, (_, i) => (
        <span key={i} className={`tk${i % 10 === 0 ? ' major' : i % 2 === 0 ? ' mid' : ''}`} style={{ left: `${i * 5}%` }} />
      ))}
      {threshold != null && <div className="thr" style={{ left: `${threshold * 100}%` }} />}
      {needle && <div className="needle" style={{ left: `${pct}%` }} />}
      {labels && (
        <>
          <span className="rl" style={{ left: 0 }}>0</span>
          <span className="rl" style={{ right: 0 }}>100</span>
          {threshold != null && <span className="rl" style={{ left: `${threshold * 100}%`, transform: 'translateX(-50%)' }}>Abstain below {Math.round(threshold * 100)}</span>}
        </>
      )}
    </div>
  );
}
