'use client';

import { Fragment } from 'react';
import { CITE_SOURCE, parseCitations } from '@/lib/cite';
import type { Outcome } from '@/lib/pipeline';
import { Ruler } from './Ruler';

// Same pattern lib/cite.ts validates with, so every accepted citation gets a page tab.
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
                <span className="tab">p.{p}</span>
              </button>
            ))}
          </Fragment>
        );
      })}
    </p>
  );
}

export function ConfidenceMeter({ value, threshold, stateLabel }: { value: number; threshold: number; stateLabel?: string }) {
  const pct = Math.min(99, Math.max(1, Math.round(value * 100))); // never show 0% or 100% certainty
  const label = stateLabel ?? (value >= 0.75 ? 'Well supported' : value >= threshold ? 'Partly supported · check the page' : 'Not supported');
  return (
    <div className="meter" data-testid="confidence" data-value={value.toFixed(3)}>
      <div className="meter-head">
        <span className="meter-num" aria-label={`Confidence ${pct} percent`}>{pct}<small>%</small></span>
        <span className="meter-state">{label}</span>
      </div>
      <Ruler value={pct / 100} needle threshold={threshold} labels ariaLabel="Calibrated confidence" />
      <p className="cap">Calibrated probability that the retrieved pages support an answer.</p>
    </div>
  );
}

const REASONS = {
  low_confidence: 'The retrieved pages don’t look like they answer this, so the model wasn’t asked.',
  verifier_abstained: 'The on-device model read the closest sentences and found no answer in them.',
  empty_answer: 'No sentence in the closest pages matched the question.',
  citation_check_failed: 'The draft answer didn’t cite the pages it was given, so it was discarded.',
} as const;

// The gate's probability stays visible, but it must not read "well supported" under a stamp that says the opposite.
const LATER: Partial<Record<keyof typeof REASONS, string>> = {
  verifier_abstained: 'Gate passed · verifier said no',
  empty_answer: 'Gate passed · nothing to quote',
  citation_check_failed: 'Gate passed · draft discarded',
};

export function AnswerCard({ outcome, threshold, onPage, onForce }: { outcome: Outcome; threshold: number; onPage: (p: number) => void; onForce?: () => void }) {
  const t = outcome.timings;
  const total = t.embedMs + t.retrieveMs + t.decideMs + t.verifyMs + t.generateMs;
  const pages = Array.from(new Set(outcome.hits.map((h) => h.chunk.page)));
  if (outcome.kind === 'abstain') {
    return (
      <div className="entry abstain" data-testid="abstain-card">
        <div className="stamp" role="note">Not in this filing</div>
        <p className="reason">{REASONS[outcome.reason]}</p>
        <ConfidenceMeter value={outcome.confidence} threshold={threshold} stateLabel={LATER[outcome.reason]} />
        <div className="pages">
          <span className="label">Closest pages, if you want to check yourself</span>
          <div className="tabs">
            {pages.map((p) => (
              <button key={p} className="tabbtn" onClick={() => onPage(p)} aria-label={`Open page ${p}`}><span className="tab">p.{p}</span></button>
            ))}
          </div>
        </div>
        {onForce && !outcome.forced && (outcome.reason === 'low_confidence' || outcome.reason === 'verifier_abstained') && (
          <p className="anyway">
            <button className="textbtn" onClick={onForce} data-testid="answer-anyway">Answer anyway</button>
            <span className="cap">Uses the closest pages. Unverified.</span>
          </p>
        )}
        <p className="meta">
          {outcome.reason === 'low_confidence' ? 'Stopped at the feature gate' : outcome.reason === 'verifier_abstained' ? 'Gate passed, then stopped by the verifier step (strict mode)' : 'Gate passed, then stopped by the citation check'} · {Math.round(total)} ms on-device
        </p>
      </div>
    );
  }
  return (
    <div className="entry answer" data-testid="answer-card">
      <Cited text={outcome.text} onPage={onPage} />
      {outcome.unverified ? (
        <p className="unv" data-testid="unverified">Below the gate: unverified</p>
      ) : (
        <ConfidenceMeter value={outcome.confidence} threshold={threshold} />
      )}
      <p className="meta">
        {outcome.engine} · {outcome.verified ? `verifier: ANSWER (${outcome.evidencePage ?? '?'})` : 'no verifier'} · {Math.round(total)} ms · citation coverage{' '}
        {Math.round(outcome.citations.coverage * 100)}% · sources: {outcome.citations.cited.map((p) => `p.${p}`).join(', ')}
      </p>
    </div>
  );
}
