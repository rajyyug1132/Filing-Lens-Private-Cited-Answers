'use client';

import { Fragment } from 'react';
import { CITE_SOURCE, parseCitations } from '@/lib/cite';
import type { Outcome } from '@/lib/pipeline';

// Same pattern lib/cite.ts validates with, so every accepted citation gets a chip.
const CITE_SPLIT = new RegExp(`(${CITE_SOURCE.replace('(\\d+', '(?:\\d+')})`, 'gi');
const IS_CITE = new RegExp(`^${CITE_SOURCE}$`, 'i');

function Cited({ text, onPage }: { text: string; onPage: (p: number) => void }) {
  return (
    <p className="answer-text" data-testid="answer-text">
      {text.split(CITE_SPLIT).map((part, i) => {
        if (!IS_CITE.test(part)) return <Fragment key={i}>{part}</Fragment>;
        const pages = Array.from(new Set(parseCitations(part)));
        return (
          <Fragment key={i}>
            {pages.map((p) => (
              <button key={p} className="cite" onClick={() => onPage(p)} data-testid="cite-chip" aria-label={`Open page ${p}`}>
                p.{p}
              </button>
            ))}
          </Fragment>
        );
      })}
    </p>
  );
}

export function ConfidenceMeter({ value, threshold }: { value: number; threshold: number }) {
  const pct = Math.min(99, Math.max(1, Math.round(value * 100))); // never show 0% or 100% certainty
  const band = value >= 0.75 ? 'good' : value >= threshold ? 'warn' : 'crit';
  const label = value >= 0.75 ? 'Well supported' : value >= threshold ? 'Partly supported, check the page' : 'Not supported';
  return (
    <div className="meter" data-testid="confidence" data-value={value.toFixed(3)}>
      <div className="meter-row">
        <span className="small strong">Confidence {pct}%</span>
        <span className={`small band-${band}`}>{band === 'good' ? '✓' : band === 'warn' ? '!' : '✕'} {label}</span>
      </div>
      <div className="meter-track" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Calibrated confidence">
        <div className={`meter-fill fill-${band}`} style={{ width: `${pct}%` }} />
        <div className="meter-tick" style={{ left: `${threshold * 100}%` }} title={`Abstain below ${Math.round(threshold * 100)}%`} />
      </div>
      <div className="muted xsmall">Calibrated probability the retrieved pages support an answer. Abstains below {Math.round(threshold * 100)}%.</div>
    </div>
  );
}

const REASONS = {
  low_confidence: 'The retrieved pages don’t look like they answer this, so the model wasn’t asked.',
  verifier_abstained: 'The on-device model read the closest sentences and found no answer in them.',
  empty_answer: 'No sentence in the closest pages matched the question.',
  citation_check_failed: 'The draft answer didn’t cite the pages it was given, so it was discarded.',
} as const;

export function AnswerCard({ outcome, threshold, onPage }: { outcome: Outcome; threshold: number; onPage: (p: number) => void }) {
  const t = outcome.timings;
  const total = t.embedMs + t.retrieveMs + t.decideMs + t.verifyMs + t.generateMs;
  const pages = Array.from(new Set(outcome.hits.map((h) => h.chunk.page)));
  if (outcome.kind === 'abstain') {
    return (
      <div className="card abstain" data-testid="abstain-card">
        <div className="abstain-head">
          <span className="abstain-icon" aria-hidden>⊘</span>
          <div>
            <div className="strong">I can’t answer that from this filing</div>
            <div className="muted small">{REASONS[outcome.reason]}</div>
          </div>
        </div>
        <ConfidenceMeter value={outcome.confidence} threshold={threshold} />
        <div className="small muted">Closest pages, if you want to check yourself:</div>
        <div className="chips">
          {pages.map((p) => (
            <button key={p} className="cite" onClick={() => onPage(p)}>p.{p}</button>
          ))}
        </div>
        <div className="xsmall muted">
          {outcome.reason === 'low_confidence' ? 'Stopped at the feature gate' : 'Gate passed, then stopped by the verifier step'} · {Math.round(total)} ms on-device
        </div>
      </div>
    );
  }
  return (
    <div className="card answer" data-testid="answer-card">
      <Cited text={outcome.text} onPage={onPage} />
      <ConfidenceMeter value={outcome.confidence} threshold={threshold} />
      <div className="xsmall muted">
        {outcome.engine} · {outcome.verified ? `verifier: ANSWER (${outcome.evidencePage ?? '?'})` : 'no verifier'} · {Math.round(total)} ms · citation coverage{' '}
        {Math.round(outcome.citations.coverage * 100)}% · sources: {outcome.citations.cited.map((p) => `p.${p}`).join(', ')}
      </div>
    </div>
  );
}
